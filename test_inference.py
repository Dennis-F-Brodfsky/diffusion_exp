from envs.diffusion_env import DiffusionEnv
import infrastructure.pytorch_util as ptu
from config import Configurations
from models.model import load_generator_discriminator
from utils.log import make_logger
from utils.ckpt import load_ckpt
import torch
from diffusers import DPMSolverMultistepScheduler


class MyDPMScheduler(DPMSolverMultistepScheduler):
    def set_timesteps(self, device = None, timesteps: list=None):
        self.timesteps = torch.tensor(timesteps)
        self.timesteps = len(timesteps) - (self.timesteps == 1).nonzero().reshape(-1) - 1
        self.num_inference_steps = len(self.timesteps)
        self.timesteps = self.timesteps.to(device)
        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0


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


ptu.init_gpu(gpu_id=1)
env = DiffusionEnv(params={'dis': DIS, 
                           'logdir': 'logs', 
                           'num_inference_steps': 1000, 
                           'inference_batch_size': 128,
                           'image_size': (3, 32, 32),
                           'penalty': -0.005, 'loc': 0, 'scale': 1,
                           'diffuser_scheduler': MyDPMScheduler}, is_eval=True)


for i in range(96):
    env.reset()
    print(i)
    for i in range(1000):
        env.step(int(i % 4 == 0))
