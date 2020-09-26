from torchvision import datasets, transforms

def cifar10_transforms(mean, std):
    # mean = (0.49139968, 0.48215841, 0.44653091)
    # std = (0.24703223, 0.24348513, 0.26158784)

    train_transforms = transforms.Compose([
        transforms.RandomRotation((-10.0, 10.0), fill=(1,1,1)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    return train_transforms, test_transforms
