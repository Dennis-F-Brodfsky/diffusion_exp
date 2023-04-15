import argparse
from infrastructure.utils import set_config_logdir, OptimizerSpec
from infrastructure.rl_trainer import DiffustionRLTrainer, DiffusionQTrainer
from configs.config import DiffustionConfig, DiffusionQConfig
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from config import Configurations
from models.model import load_generator_discriminator
from utils.log import make_logger
from utils.ckpt import load_ckpt
from diffusers import DPMSolverMultistepScheduler


class MyDPMScheduler(DPMSolverMultistepScheduler):
    def set_timesteps(self, device = None, timesteps=None):
        self.timesteps = torch.from_numpy(timesteps)
        self.timesteps = len(timesteps) - (self.timesteps == 1).nonzero().reshape(-1) - 1
        self.num_inference_steps = len(self.timesteps)
        self.timesteps = self.timesteps.to(device)
        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0


class Actor(nn.Module):
    def __init__(self, timesteps=1000) -> None:
        super().__init__()
        self.embedding = nn.Linear(timesteps, 256)
        self.l2 = nn.Linear(256, 2)
        self.activation = nn.ReLU()
        self.num_class = timesteps

    def forward(self, x: torch.Tensor):
        # somet times x:refer to scalar while another time 
        # time_embed = F.one_hot(x.long(), num_classes=self.num_class)
        # print(self.embedding_weights.weight)
        return self.activation(self.l2(self.activation(self.embedding(x.float()))))


class Critic(nn.Module):
    def __init__(self, timesteps=1000) -> None:
        super().__init__()
        self.embedding = nn.Linear(timesteps, 256)
        self.l2 = nn.Linear(256, 2)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        return self.activation(self.l2(self.activation(self.embedding(x.float()))))


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
load_ckpt(DIS, None, 'models/model=D-best-weights-step=18000.pth', True, False, False, True)


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='todo')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--timestep', type=int, default=1000)
parser.add_argument('--penalty', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--ibs', type=int, default=128)
parser.add_argument('--algo', type=str, default='q')
arg_cmd = parser.parse_args()


ob_dim, ac_dim = 1000, 2
logits_na = Actor()
c1 = Critic()
c2 = Critic()
actor_optim_spec = OptimizerSpec(constructor=Adam, optim_kwargs={'lr': arg_cmd.lr}, learning_rate_schedule=None)

if arg_cmd.algo == 'pg':
    args = DiffustionConfig('', arg_cmd.timestep, exp_name=arg_cmd.exp_name, no_gpu=False, scalar_log_freq=10, seed=arg_cmd.seed,
                        actor_optim_spec=actor_optim_spec, standardize_advantages=True, inference_batch_size=arg_cmd.ibs,
                        reward_to_go=True, logits_na=logits_na, gamma=0.9, which_gpu=arg_cmd.gpu_id,
                        save_params=False, penalty=arg_cmd.penalty, dis=DIS, diffuser_scheduler=MyDPMScheduler)
    set_config_logdir(args)
    param = vars(args)
    trainer = DiffustionRLTrainer(param)
    trainer.run_training_loop(args.time_steps, trainer.agent.actor, trainer.agent.actor)
elif arg_cmd.algo == 'q':
    args_q = DiffusionQConfig('', arg_cmd.timestep, exp_name=arg_cmd.exp_name)
    set_config_logdir(args_q)
    param = vars(args_q)
    trainer = DiffusionQTrainer(param)
    trainer.run_training_loop(args_q.time_steps, trainer.agent.actor, trainer.agent.actor)
