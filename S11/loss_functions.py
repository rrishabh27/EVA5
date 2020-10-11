import torch
import torch.nn as nn
import torch.optim as optim

def l1_loss(lambda_l1 = 5e-4):
    l1_reg = sum([torch.sum(abs(param)) for param in net.parameters()])
    l1_loss = l1_reg * lambda_l1

    return l1_loss

def cross_entropy_loss():
    return nn.CrossEntropyLoss()

