# SESSION 7

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

![CIFAR10 IMAGES](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)

In this code, we needed to train on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset with these constraints:

* 80% accuracy with less than 1M parameters
* No restriction on number of epochs
* 3 transition layers in the model(maxpooling followed by 1x1 convolutions)
* Receptive field of more than 44
* One of the layers must use Depthwise Separable Convolution
* One of the layers must use Dilated Convolution
* Compulsarily use GAP

---

After training the whole network for 35 epochs, we got:

* 84.70% train accuracy
* 82.61% validation accuracy(highest among all the epochs)
* 453,738 total parameters
* Receptive Field of 72
* The model is underfitting
* One takeaway is that this model can reach better accuracy if trained for some more epochs

**Classwise Accuracy:**
  * Accuracy of plane : 93 %
  * Accuracy of   car : 85 %
  * Accuracy of  bird : 75 %
  * Accuracy of   cat : 67 %
  * Accuracy of  deer : 90 %
  * Accuracy of   dog : 75 %
  * Accuracy of  frog : 97 %
  * Accuracy of horse : 69 %
  * Accuracy of  ship : 93 %
  * Accuracy of truck : 89 %
  
  **Train and Test Metrics:**
  
  ![Metrics](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S7/images/metrics.png)
  
  ## Group Members:
  **RISHABH BHARDWAJ**
  
  **RASHU TYAGI**
  
