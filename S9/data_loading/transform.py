from torchvision import datasets, transforms

import albumentations as A
import albumentations.pytorch.transforms as APT


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

def cifar10_albumentations(mean, std):

    train_transforms = A.Compose([
        A.OneOf([
            A.GridDistortion(distort_limit=(-0.05, 0.05), p=0.5),
            A.Rotate(limit=(-10, 10), p=0.5)
        ], p=0.5),
        A.HorizontalFlip(p=0.25),
        A.Normalize(mean=mean, std=std),
        A.Cutout(num_holes=1),
        APT.ToTensor()
#         APT.ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.Normalize(mean=mean, std=std),
        APT.ToTensor()
#         APT.ToTensorV2()
    ])

    return train_transforms, test_transforms

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        '''
        UnNormalizes an image given its mean and standard deviation
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        '''
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
