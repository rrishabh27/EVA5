import torch.nn as nn
import torch.nn.functional as F

import normalisation as norm

dropout_value = 0.07
num_splits = 2

class Cifar10_Net(nn.Module):
    def __init__(self, norm_type = 'BN'):
        super(Cifar10_Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU() 
        ) # output_size = 26

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU() 
        ) # output_size = 24

        # transition block 
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 12
        # end

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 10

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 8

        
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1), 
        #     nn.BatchNorm2d(20) if norm_type == 'BN' else norm.GBN(num_features=16, num_splits=num_splits),
        #     nn.Dropout(dropout_value),
        #     nn.ReLU()
        # )

        # transition block 
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 12
        # end
        
        # DILATED CONVOLUTION
        self.conv5_dilated = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding = 0, dilation = 2),  # preserves resolution
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1), 
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6

        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1), 
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32) if norm_type == 'BN' else norm.GBN(num_features=32, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 12
        # end

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6( padding = 1)

        # depthwise separable convolution
        self.conv8_ds = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1, groups = 64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6( padding = 1)

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6( padding = 1)

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 0),
            nn.BatchNorm2d(128) if norm_type == 'BN' else norm.GBN(num_features=128, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6( padding = 1)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6)
        ) # output_size = 1

        # final FC layer (read fully convolutional, not fully connected)
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64) if norm_type == 'BN' else norm.GBN(num_features=64, num_splits=num_splits),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1)
        )

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
