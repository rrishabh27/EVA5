from torchvision import datasets, transforms

def cifar10_transforms():

    train_transforms = transforms.Compose([
        transforms.RandomRotation((-10.0, 10.0), fill=(1,1,1)),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=7, resample=False, fillcolor=0)
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ])

    return train_transforms, test_transforms
