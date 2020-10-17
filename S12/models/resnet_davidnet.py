'''Modified ResNet(DavidNet) in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

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
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        out += self.shortcut(x) # passing 'x' inside self.shortcut not 'out'
        out = F.relu(out)
        return out


class ResNet_mod(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_mod, self).__init__()
        # self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1,
                                stride=1, bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2,2)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.pool4 = nn.MaxPool2d(2,2)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1,
                                stride=1,bias=False)


        self.pool5 = nn.MaxPool2d(4,4)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.softmax = nn.Softmax(dim=1)

        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64,  128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256,  512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block,in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # prep layer
        out = F.relu(self.bn1(self.conv1(x))) # in_channels = 3, out_channels = 64

        # layer 1 = conv2, maxpool2d, ResNet layer 2(i.e. block 1)
        out2 = F.relu(self.bn2(self.pool2(self.conv2(out)))) # in = 64, out = 128
        out3 = self.layer2(out2) # in = 64 but should be 128, out = 128
        out3 += (self.bn2(self.conv1_1(out2))) # in = 128, out = 128
        out3 = F.relu(out3)

        # layer 2 = conv3, maxpooling2d, bn, relu
        out4 = F.relu(self.bn3(self.pool3(self.conv3(out3)))) # in = 128, out = 256

        # layer 3 = conv4, maxpool2d, ResNet layer 4(block4)
        out5 = F.relu(self.bn4(self.pool4(self.conv4(out4))))# in = 256, out = 512
        out6 = self.layer4(out5) # in = 512, out = 512
        out6 += (self.bn4(self.conv1_2(out5))) # in  = 512, out = 512
        out6 = F.relu(out6)

        out7 = self.pool5(out6)
        out8 = self.linear(out7) # in = 512, out = 10
        return F.log_softmax(out8)


def ResNet_DavidNet():
    return ResNet_mod(BasicBlock, [2,2,2,2]) # or customise the self.layer2/4 line and pass [2,2]


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
