import torchvision


train_set = torchvision.datasets.CIFAR10('data', download=True)
for batch_idx, data in enumerate(train_set):
    real_img, _ = data
    real_img.save(f'img/cifar-10/{batch_idx}.jpg')
