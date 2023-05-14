# from envs.diffusion_env import DiffusionEnv
import infrastructure.pytorch_util as ptu
from config import Configurations
from models.model import load_generator_discriminator
import argparse
from utils.log import make_logger
from utils.ckpt import load_ckpt
import torch
from diffusers import DPMSolverMultistepScheduler, DDPMPipeline
import seaborn as sns


class MyDPMScheduler(DPMSolverMultistepScheduler):
    def set_timesteps(self, device = None, timesteps=None):
        self.timesteps = torch.tensor(timesteps)
        self.timesteps = (self.timesteps == 1).nonzero().reshape(-1).flip((0,))
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
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--inf_way', type=int, nargs='+', default=[25, 50, 125, 200, 300, 425, 600, 800, 999])
parser.add_argument('--to', type=str, default='img/dist.jpg')
args = parser.parse_args()


ptu.init_gpu(gpu_id=args.gpu_id)
DIS.to(ptu.device)
ddpm = DDPMPipeline.from_pretrained('models/ddpm-cifar10-32').to(ptu.device)
s1 = MyDPMScheduler.from_config(ddpm.scheduler.config)
ts = [0]*1000
inf_way = args.inf_way
for i in inf_way:
    ts[i] = 1
s1.set_timesteps(ptu.device, ts)
s2 = DPMSolverMultistepScheduler.from_config(ddpm.scheduler.config)
s2.set_timesteps(len(inf_way), ptu.device)
res1, res2 = [], []
with torch.no_grad():
    for _ in range(500):
        img = torch.randn((512, 3, 32, 32))
        img_1 = img[:]
        for t in s1.timesteps:
            model_output = ddpm.unet(img.to(ptu.device), t).sample
            img = s1.step(model_output, t, img.to(ptu.device)).prev_sample
        for t in s2.timesteps:
            model_output = ddpm.unet(img_1.to(ptu.device), t).sample
            img_1 = s2.step(model_output, t, img_1.to(ptu.device)).prev_sample
        res1.extend(DIS(img.to(ptu.device), 1)['adv_output'].to('cpu').tolist())
        res2.extend(DIS(img_1.to(ptu.device), 1)['adv_output'].to('cpu').tolist())

group = ['ours']*len(res1) + ['benchmark']*len(res2)
visual_data = {'reward': res1 + res2, 'group': group}
fig = sns.displot(data=visual_data, x = 'reward', hue='group', kind='kde')
fig.savefig(args.to)
