from torchvision import datasets
import transform

def cifar10_dataset(train_transforms, test_transforms):
    
    # train_transforms, test_transforms = transforms.cifar10_transforms()

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, 
    transform=train_transforms)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
    transform=test_transforms)

    return trainset, testset
