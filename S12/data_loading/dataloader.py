import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .dataset import TinyImageNet

def cifar10_dataloader(trainset, testset, batch_size = 128, num_workers = 4):
    SEED = 1
    
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA available?", cuda)
    
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    dataloader_args = dict(shuffle = True, batch_size = batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle = True, batch_size = 64)

    # trainset, testset = datasets.cifar10_dataset()

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader


class TinyImageNetDataLoader:

    def __init__(self, train_transforms, test_transforms, data_dir, batch_size=128, shuffle=True, num_workers=4, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = TinyImageNet(
            self.data_dir,
            train=True,
            download=True,
            transform=train_transforms
        )

        self.test_set = TinyImageNet(
            self.data_dir,
            train=False,
            download=False, ######
            transform=test_transforms
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)


