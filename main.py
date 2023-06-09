import argparse
from infrastructure.utils import set_config_logdir, OptimizerSpec, PiecewiseSchedule, sample_trajectories
from infrastructure.rl_trainer import DiffustionRLTrainer, DiffusionQTrainer, DiffusionQRDQNTrainer
from configs.config import DiffusionConfig, DiffusionQConfig, DiffusionQRDQNConfig
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from config import Configurations
from models.model import load_generator_discriminator
from utils.log import make_logger
from utils.ckpt import load_ckpt
from functools import partial
from diffusers import DPMSolverMultistepScheduler


class MyDPMScheduler(DPMSolverMultistepScheduler):
    def set_timesteps(self, device = None, timesteps=None):
        self.timesteps = torch.from_numpy(timesteps)
        self.timesteps = (self.timesteps == 1).nonzero().reshape(-1).flip((0,))
        self.num_inference_steps = len(self.timesteps)
        self.timesteps = self.timesteps.to(device)
        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0


class FFN(nn.Module):
    def __init__(self, trans_dim = 256, hidden_dim = 512, dropout=0.1) -> None:
        super().__init__()
        self.l1 = nn.Linear(trans_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, trans_dim)
        self.activation = nn.ELU()
        self.norm = nn.LayerNorm(trans_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.drop(self.activation(self.l1(x)))
        x1 = self.drop(self.activation(self.l2(x1)))
        return self.norm(x1 + x)


class Actor(nn.Module):
    def __init__(self, timesteps=1000) -> None:
        super().__init__()
        self.embedding = nn.Linear(timesteps, 256)
        self.ffn = FFN(256, 512)
        self.linear = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor):
        # somet times x:refer to scalar while another time 
        # time_embed = F.one_hot(x.long(), num_classes=self.num_class)
        # print(self.embedding_weights.weight)
        return self.linear(self.ffn(self.ffn(self.embedding(x))))


class Critic(nn.Module):
    def __init__(self, n_quantile=1, ac_dim=2, timesteps=1000) -> None:
        super().__init__()
        self.n_quantile = n_quantile
        self.ac_dim = ac_dim
        self.embedding = nn.Linear(timesteps, 256)
        self.ffn = FFN(256, 512)
        self.linear = nn.Linear(256, n_quantile*ac_dim)
    
    def forward(self, x: torch.Tensor):
        if self.n_quantile == 1:
            return self.linear(self.ffn(self.ffn(self.embedding(x))))
        else:
            return self.linear(self.ffn(self.ffn(self.embedding(x)))).reshape(-1, self.n_quantile, self.ac_dim)


cfg = Configurations('configs/CIFAR10/DCGAN.yaml')
cfg.RUN.mixed_precision = False
cfg.RUN.distributed_data_parallel = False
cfg.RUN.eval_metrics = []
cfg.RUN.load_data_in_memory = False
cfg.RUN.langevin_sampling = False
cfg.RUN.freezeD = 1
cfg.RUN.ref_dataset = 'test'
cfg.RUN.train = 0
cfg.RUN.standing_statistics = 0
cfg.RUN.vis_fake_images = 0
cfg.RUN.k_nearest_neighbor = 0
cfg.RUN.interpolation = 0
cfg.RUN.intra_class_fid = 0
cfg.RUN.GAN_train = 0
cfg.RUN.GAN_test = 0
cfg.RUN.eval_metric = "none"
cfg.RUN.semantic_factorization = 0
cfg.RUN.ckpt_dir = '.'
cfg.RUN.eval_backbone = "InceptionV3_torch"
cfg.RUN.post_resizer = "legacy"
cfg.RUN.data_dir = '...'
cfg.RUN.batch_statistics = 0
cfg.RUN.standing_statistics = 0
cfg.RUN.save_freq = 100
cfg.RUN.print_freq = 50
cfg.RUN.pre_resizer = "nearest"
cfg.OPTIMIZATION.world_size = 1
cfg.check_compatability()
logger = make_logger('.', 'test', None)
_, _, _, DIS, *_ = load_generator_discriminator(cfg.DATA, cfg.OPTIMIZATION, cfg.MODEL, cfg.STYLEGAN, cfg.MODULES, cfg.RUN, 'cpu', logger)


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='todo')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_itr', type=int, default=1000)
parser.add_argument('--penalty', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--ibs', type=int, default=512)
parser.add_argument('--algo', type=str, default='q')
parser.add_argument('--learning_start', type=int, default=10000)
parser.add_argument('--loc', type=float, default=0.0)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--rlbs', type=int, default=512)
parser.add_argument('--diffuser_dir', type=str, required=True)
parser.add_argument('--gan_dir', type=str, default='models/model=D-best-weights-step=18000.pth')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'celebahq'])
parser.add_argument('--cpu_debug', default=False, type=bool)
arg_cmd = parser.parse_args()


# 'models/model=D-best-weights-step=18000.pth'
load_ckpt(DIS, None, arg_cmd.gan_dir, True, False, False, True)
ob_dim, ac_dim = 1000, 2
logits_na = Actor()
actor_optim_spec = OptimizerSpec(constructor=Adam, optim_kwargs={'lr': arg_cmd.lr}, learning_rate_schedule=None)
critic_optim_spec = OptimizerSpec(constructor=Adam, optim_kwargs={'lr': arg_cmd.lr}, learning_rate_schedule=None)
explor_sche = PiecewiseSchedule([(0, 0.5), (arg_cmd.n_itr // 10, 0.05), (arg_cmd.n_itr, 0.0)])


if arg_cmd.algo == 'pg':
    args = DiffusionConfig('', arg_cmd.n_itr, exp_name=arg_cmd.exp_name, scalar_log_freq=10, seed=arg_cmd.seed,
                        actor_optim_spec=actor_optim_spec, standardize_advantages=True, inference_batch_size=arg_cmd.ibs,
                        reward_to_go=True, logits_na=logits_na, gamma=0.9, which_gpu=arg_cmd.gpu_id, no_gpu=not arg_cmd.cpu_debug,
                        save_params=False, penalty=arg_cmd.penalty, dis=DIS, diffuser_scheduler=MyDPMScheduler, batch_size=arg_cmd.rlbs,
                        diffuser_dir=arg_cmd.diffuser_dir)
    set_config_logdir(args)
    param = vars(args)
    trainer = DiffustionRLTrainer(param)
    trainer.run_training_loop(args.time_steps, trainer.agent.actor, trainer.agent.actor)
elif arg_cmd.algo == 'q':
    q_fun = lambda: Critic()
    args_q = DiffusionQConfig('',arg_cmd.n_itr, exp_name=arg_cmd.exp_name, batch_size=arg_cmd.rlbs,
                              which_gpu=arg_cmd.gpu_id, seed=arg_cmd.seed,
                              dis=DIS, penalty=arg_cmd.penalty, diffuser_scheduler=MyDPMScheduler, 
                              gamma=0.99, scalar_log_freq=5000, learning_start=arg_cmd.learning_start,
                              env_wrappers=lambda env: env, q_func=q_fun, q2_func=q_fun, 
                              clipped_q=True, double_q=True, loc=arg_cmd.loc, scale=arg_cmd.scale,
                              exploration_schedule=explor_sche, learning_freq=1000,
                              q_net_spec=critic_optim_spec, num_agent_train_steps_per_iter=50,
                              inference_batch_size=arg_cmd.ibs)
    set_config_logdir(args_q)
    param = vars(args_q)
    trainer = DiffusionQTrainer(param)
    trainer.run_training_loop(args_q.time_steps, trainer.agent.actor, trainer.agent.actor)
elif arg_cmd.algo == 'qr-dqn':
    quantile_fun = lambda: Critic(n_quantile=5)
    args_qr_dqn = DiffusionQRDQNConfig('', arg_cmd.n_itr, exp_name=arg_cmd.exp_name, batch_size=arg_cmd.rlbs, 
                                       which_gpu=arg_cmd.gpu_id, seed=arg_cmd.seed,
                                       dis=DIS, penalty=arg_cmd.penalty, diffuser_scheduler=MyDPMScheduler, 
                                       gamma=0.99, scalar_log_freq=5000, learning_start=arg_cmd.learning_start,
                                       env_wrappers=lambda env: env, quantile_func=quantile_fun, 
                                       double_q=True, loc=arg_cmd.loc, scale=arg_cmd.scale,
                                       exploration_schedule=explor_sche, 
                                       quantile_net_spec=critic_optim_spec, inference_batch_size=arg_cmd.ibs)
    set_config_logdir(args_qr_dqn)
    param = vars(args_qr_dqn)
    trainer = DiffusionQRDQNTrainer(param)
    trainer.run_training_loop(args_qr_dqn.time_steps, trainer.agent.actor, trainer.agent.actor)
path, _ = sample_trajectories(trainer.env, trainer.agent.actor, 1000, 1000)
print(path[0]['action'].nonzero()[0][::-1])
