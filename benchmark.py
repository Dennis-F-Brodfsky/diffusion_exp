from diffusers import DDPMPipeline


pipe1 = DDPMPipeline.from_pretrained('models/ddpm-cifar10-32').to('cuda:3')
cnt = 0
for i in range(50):
    imgs = pipe1(batch_size=128, num_inference_steps=200, output_type='pil').images
    for img in imgs:
        img.save(f'img/ddpm_test2/{cnt}.jpg')
        cnt += 1
