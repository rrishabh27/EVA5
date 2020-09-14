import torch
import dataset

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