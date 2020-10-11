from __future__ import print_function

import copy
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import click
import cv2
# pip install torchsummary # !pip install torchsummary on colab
# git clone https://github.com/sksq96/pytorch-summary
from torchsummary import summary
from tqdm import tqdm


from data_loading import transform, dataset, dataloader
from models import resnet
from utils import plot_metrics, misclassifications, classwise_accuracy
import normalisation as norm, loss_functions, trainer, tester, run_model
import run_grad_cam

mean = (0.49139968, 0.48215841, 0.44653091)
std = (0.24703223, 0.24348513, 0.26158784)

train_transforms, test_transforms = transform.cifar10_transforms(mean, std)
trainset, testset = dataset.cifar10_dataset(train_transforms, test_transforms)
train_loader, test_loader = dataloader.cifar10_dataloader(trainset, testset)


# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# classes in the data
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model summary
net = resnet.ResNet18().to(device)
print(summary(net, input_size=(3, 32, 32)))

EPOCHS = 25
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum = 0.9)

train_acc = []
train_losses = []
test_acc = []
test_losses = []

run_model.evaluation( net, train_loader, test_loader, optimizer, EPOCHS, device,
                    train_acc, train_losses, test_acc, test_losses)

train_metric = (train_acc, train_losses)
test_metric = (test_acc, test_losses)

plot_metrics.metrics(train_metric, test_metric)

# grad cam
target_layers = ["layer1", "layer2", "layer3", "layer4"]
run_grad_cam.plot_gradcam(net, device, test_loader, classes, target_layers, mean, std)

# misclassified = misclassifications.get_misclassified(net)
# misclassifications.plot_misclassification(misclassified)











