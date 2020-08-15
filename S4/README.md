In this code, we are trying to achieve the following results while training on MNIST dataset( input image size = 28x28):

1. 99.4% validation accuracy
2. Less than 20k Parameters
3. Less than 20 Epochs
4. No fully connected layer

The model consists of 7 normal convolution layers and 2 transition layers and used a total of 19,042 parameters.

We applied the transition layer( max-pooling followed by 1x1 convolution) twice after the receptive field of model increased by 5x5.

We have used dropout and batch normalization at every convolution layer except just before the last layer.

The batch size was 64 and the data was normalized so that mean = 0 and standard deviation = 1

We got the maximum training accuracy of 99.21% and maximum validation accuracy of 99.42% .
