import torch
import torch.functional as F

import model
import loss_functions

def test(net, device, test_loader, test_acc, test_losses):
    '''
    Test function to validate the model
    '''
    # test_acc = []
    # test_losses = []

    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # since we do not want to compute gradients on the test data, we use torch.no_grad()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = loss_functions.cross_entropy_loss()
            test_loss += loss(output, target)
            pred = output.argmax(dim=1, keepdim = True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))

    # return test_acc, test_losses
