'''ResNet18 and DavidNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = [128, 512]

        # preparation layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # layer 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, in_val=0) # instead of stride=2 as per resnet

        # middle layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # layer 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                stride=1, padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, in_val=1) # same comment as above
        
        # maxpooling then linear layer to produce 10 outputs
        
        self.pool = nn.MaxPool2d(4,4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, in_val):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes[in_val], planes, stride))
            # self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        prep = self.conv1(x)

        x1 = self.conv2(prep)
        r1 = self.layer2(x1)
        out1 = x1 + r1

        mid = self.conv3(out1)

        x2 = self.conv4(mid)
        r2 = self.layer4(x2)
        out2 = x2 + r2

        out = self.pool(out2)
        out = out.view(out.size(0), -1) 
        # You supply your batch_size as the first number,and then “-1” basically 
        # tells Pytorch, “you figure out this other number for me… please.” 
        # Your tensor will now feed properly into any linear layer. Now we’re talking!
        # Linear layers will always take a 2d input, so we need to convert from 4d to 2d
        # print(out.size())
        out = self.linear(out)
        return out



def DavidNet():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
