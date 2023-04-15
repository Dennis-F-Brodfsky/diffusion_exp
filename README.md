# diffusion_exp
diffusion-RL project which is running on enviroments:
 
Linux Ubuntu 20.04 with GPU NVIDIA GeForce 3090; 

Python version: 3.7

Some main packages version are as fellow:

PyTorch version: 1.12

Torchvision version: 0.13.0

Cuda version: 1.13

Gym version: 0.18.3

And other packages are in requirements.txt 

Thanks for StudioGAN implementation of GAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/tree/master

and Huggingface implementation of DDPM, DPMSolver, DDIM etc: https://github.com/huggingface/diffusers

## QA
Q: How to get PreTrained DDPM?

A: `cd models` and run `git lfs install`、`git clone https://huggingface.co/google/ddpm-cifar10-32`、`GIT_LFS_SKIP_SMUDGE=1` 

Detail Files are here: https://huggingface.co/google/ddpm-cifar10-32/tree/main

Q: How to get PreTrained StudionGAN?(DCGAN)

A: `cd models` then `wget https://huggingface.co/Mingguksky/PyTorch-StudioGAN/blob/main/studiogan_official_ckpt/CIFAR10_tailored/CIFAR10-DCGAN-train-2022_01_11_20_39_29/model%3DD-best-weights-step%3D50000.pth`, and you can get PreTrained Discriminator of DCGAN. Some other preTrained checkpoints are here https://huggingface.co/Mingguksky/PyTorch-StudioGAN/tree/main/studiogan_official_ckpt
