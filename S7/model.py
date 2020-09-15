import torch.nn as nn
import torch.nn.functional as F

import normalisation as norm

dropout_value = 0.07
num_splits = 2

class Cifar10_Net(nn.Module):
    def __init__(self, norm_type = 'BN'):
        super(Cifar10_Net, self).__init__()
        
        # input channels: 3(RGB channels of size 32x32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU() 
        )# output channels: 32; receptive field = 3x3
        
        # input channels: 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU() 
        ) # output channels: 64; rf = 5x5

        # transition block 
        # input channels: 64
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 32; rf = 6x6
        # end
        
        # input channels: 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 64; rf = 10x10
        
        # input channels: 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 128; rf = 14x14

        
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1), 
        #     nn.BatchNorm2d(20) if norm_type == 'BN' else norm.GBN(num_features=16, num_splits=num_splits),
        #     nn.Dropout(dropout_value),
        #     nn.ReLU()
        # )

        # transition block 
        # input channels: 128
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 32; rf = 16x16
        # end
        
        # DILATED CONVOLUTION
        # input channels: 32
        self.conv5_dilated = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding = 0, dilation = 2),  # preserves resolution
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 64; rf = 32x32(dilated convolutions increase receptive field dramatically)
        
        # input channels: 64
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1), 
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 128; rf = 36x36
        
        # transition block
        # input channels: 128
        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1), 
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output channels: 32; rf = 36x36
        # end
        
        # input channels: 32
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 64; rf = 40x40

        # depthwise separable convolution
        # input channels: 64
        self.conv8_ds = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1, groups = 64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 64; rf = 44x44
        
        # input channels: 64
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 128; rf = 48x48
        
        # input channels: 128
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 0),
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 128; rf = 52x52
        
        # input channels: 128; input size = 6x6
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6)
        )# output channels: 128; output size = 1x1; rf = 72x72(GAP also increases receptive field to a greater extent)

        # final FC layer (read fully convolutional, not fully connected)
        # input channels: 128
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )# output channels: 64
        
        # input channels: 64
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1)
        )# output channels: 10(= number of classes)

        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv_1_1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv_1_2(x)
            x = self.conv5_dilated(x)
            x = self.conv6(x)
            x = self.conv_1_3(x)
            x = self.conv7(x)
            x = self.conv8_ds(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.gap(x)
            x = self.conv11(x)
            x = self.conv12(x)

            x = x.view(-1, 10)
            return x
