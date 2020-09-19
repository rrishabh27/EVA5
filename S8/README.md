# Session 8

In this code, we trained the ResNet-18 model (code: https://github.com/kuangliu/pytorch-cifar) on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

We wanted to achieve >85% accuracy with no restriction on number of epochs.

We trained the model for 25 epochs and achieved 85.30% accuracy in the 11th epoch.
* Highest train accuracy: 97.23%
* Highest test accuracy: 87.36%

Since the ResNet-18 network has a large number of parameters (11 million), we can see that the model is overfitting even after adding some image augmentations.

---

**Parameters and Hyperparameters**

* Loss function : Cross Entropy Loss
* Optimizer : SGD(learning_rate = 0.005, momentum = 0.9)
* Batch Size : 128
* Epochs : 25
---

**Image Augmentation**

* Random Rotation of [-10, 10] degrees
* Random Horizontal Flip(probability = 0.25)
* Random Affine(shear = [-7, 7] degrees)
---

**Classwise Accuracy**

* Accuracy of plane : 92 %
* Accuracy of   car : 86 %
* Accuracy of  bird : 90 %
* Accuracy of   cat : 60 % (least performing)
* Accuracy of  deer : 83 %
* Accuracy of   dog : 87 %
* Accuracy of  frog : 87 %
* Accuracy of horse : 87 %
* Accuracy of  ship : 92 %
* Accuracy of truck : 96 %

---

## Group Members

**Rishabh Bhardwaj**

**Rashu Tyagi**



