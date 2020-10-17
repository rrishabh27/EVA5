from __future__ import print_function

import torch
import torch.functional as F

from tqdm import tqdm

# import model
from essentials import loss_functions

def test(net, device, test_loader, test_acc, test_losses):
    '''
    Test function to validate the model
    '''
    # test_acc = []
    # test_losses = []

    net.eval()
    pbar = tqdm(test_loader)
    test_loss = 0
    correct = 0
    processed = 0
    with torch.no_grad(): # since we do not want to compute gradients on the test data, we use torch.no_grad()
        for batch_idx, (data, target) in enumerate(pbar, 0):
            data, target = data.to(device), target.to(device)
            output = net(data)
            criterion = loss_functions.cross_entropy_loss()
            loss = criterion(output, target)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim = True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            processed += len(data)
            pbar.set_description(desc = f'Loss = {loss.item()} Batch_id = {batch_idx} Accuracy = {100. * correct / processed:0.2f}')
        

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))

    # return test_acc, test_losses
