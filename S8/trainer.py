from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import model
import loss_functions

def train(net, device, train_loader, optimizer, epoch, train_acc, train_losses):

    net.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar, 0):
        data, target = data.to(device), target.to(device)
        # print(data.shape, target.shape)
        
        optimizer.zero_grad()

        y_pred = net(data)
        # print(y_pred.shape)

        # criterion = nn.CrossEntropyLoss()
        criterion = loss_functions.cross_entropy_loss()
        loss = criterion(y_pred, target)
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss)

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        # accuracy = 100. * correct / processed

        pbar.set_description(desc = f'Loss = {loss.item()} Batch_id = {batch_idx} Accuracy = {100. * correct / processed:0.2f}')
        train_acc.append(100. * correct / processed)

        # return train_acc, train_losses
        
