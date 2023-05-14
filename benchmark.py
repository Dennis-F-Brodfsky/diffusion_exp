from diffusers import DDPMPipeline, DPMSolverMultistepScheduler
import os


pipe1 = DDPMPipeline.from_pretrained('models/ddpm-cifar10-32').to('cuda:3')
sche = DPMSolverMultistepScheduler.from_config(pipe1.scheduler.config)
pipe1.scheduler = sche
for j in range(3, 15):
    cnt = 0
    if not os.path.exists(f'img/dpm_test_{j}'):
        os.mkdir(f'img/dpm_test_{j}')
    for i in range(64):
        imgs = pipe1(batch_size=128, num_inference_steps=j, output_type='pil').images
        for img in imgs:
            img.save(f'img/dpm_test_{j}/{cnt}.jpg')
            cnt += 1
