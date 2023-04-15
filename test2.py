import config
from models.model import load_generator_discriminator
from utils.log import make_logger
from utils.ckpt import load_ckpt
import torch
import torchvision
from PIL import Image
from diffusers import DDPMPipeline, DDIMScheduler


cfg = config.Configurations('configs/CIFAR10/DCGAN.yaml')
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


diff = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32').to('cuda:3')
scheduler = DDIMScheduler.from_config(diff.scheduler.config)
diff.scheduler = scheduler
_, _, _, DIS, *_ = load_generator_discriminator(cfg.DATA, cfg.OPTIMIZATION, cfg.MODEL, cfg.STYLEGAN, cfg.MODULES, cfg.RUN, 3, logger)
load_ckpt(DIS, None, 'model=D-best-weights-step=18000.pth', True, False, False, True)
for i in [5, 50, 100, 200, 500, 1000]:
    res = []
    cnt = 0
    for batch in range(10):
        imgs = diff(batch_size=64, num_inference_steps=i, output_type=None).images
        #for img in imgs:
        #    img.save(f'img/test2/steps_{i}_img_{cnt}.jpg')
        with torch.no_grad():
            res.append(torch.sigmoid(DIS(2*torch.tensor(imgs).permute(0, 3, 1, 2).to('cuda:3')-1, 1)['adv_output']).mean().item())
    print(torch.mean(torch.tensor(res)))
# img = torch.stack([torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(torchvision.transforms.ToTensor()(Image.open(f'img/cifar-10/{i}.jpg'))) 
#    for i in range(10)], dim=0).to('cuda:0')
# img_2 = 2*torch.randn((10, 3, 32, 32)).to('cuda:0')-1
# print(img.shape)
# print(DIS)
