# Aim

In this exercise, we aim to reach 99.40% accuracy on MNIST dataset. The constraints being:

1. 99.4% accuracy(this must be consistently shown in the last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 10000 Parameters

# Model

dropout_value = 0.10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=0),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU() 
        ) # output_size = 26

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=0),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value),
            nn.ReLU() 
        ) # output_size = 24

        # transition block 
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 12
        # end

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 10

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding = 0),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 8

        #
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1), 
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(dropout_value),
        #     nn.ReLU()
        # )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding = 0), 
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding = 1),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # output_size = 6( padding = 1)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6)
        ) # output_size = 1

        # final FC layer (read fully convolutional, not fully connected)
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=1),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1)
        )

        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv_1_1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            # x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.gap(x)
            x = self.conv8(x)
            x = self.conv9(x)

            x = x.view(-1, 10)
            return F.log_softmax(x)

# Results

* The model has **9,922** parameters. 

* This model uses nine convolution layers and one `MaxPooling` layer after the receptive field of the model becomes **5x5** since in MNIST images, the edges can be detected by zooming in on the images and drawing a 5x5 grid manually.

* We also added one `GAP` layer after the image size became **6x6** and then two fully connected layers to give out 10 outputs.

* We took the model step by step right from getting the inital code setup right and applying changes one by one to know the effects of each change on the model.

* In the final model, we applied two image augmentation techniques(`ColorJitter` and `RandomRotation`) along with `StepLR` learning rate optimizer to reach our goal.


