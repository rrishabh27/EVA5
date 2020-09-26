
import torch
import torch.nn as nn

class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        # We need to add 1x1 layers in between if we choose to follow different 
        # sized channels in between max-pooling layers.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )

        self.pool_1 = nn.MaxPool2d(2,2) # output size = 16

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) 

        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool_2 = nn.MaxPool2d(2,2) # output size = 8

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 8)
        ) # output_size = 1

        # final FC layer (read fully convolutional, not fully connected)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1)
        )

    
    def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x1 + x2)
            x4 = self.pool_1(x1 + x2 + x3)

            x5 = self.conv4(x4)
            x6 = self.conv5(x5 + x4)
            x7 = self.conv6(x6 + x5 + x4)
            x8 = self.pool_2(x7 + x6 + x5)

            x9 = self.conv7(x8)
            x10 = self.conv8(x9 + x8)
            x11 = self.conv9(x10 + x9 + x8)

            x12 = self.gap(x11)
            x13 = self.fc(x12)

            x13 = x13.view(-1, 10)
            return x
