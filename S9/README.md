# Session 9

Previously in [S8](https://github.com/rishabh-bhardwaj-64rr/EVA5/tree/master/S8), we trained ResNet18 on CIFAR-10 dataset. Even though we had introduced some transformations, the model exhibited heavy overfitting.

In this code, we have added some other image augmentation techniques also. 

[Albumentations](https://github.com/albumentations-team/albumentations) library has large, diverse set of transformation techniques which has easy integration with PyTorch. 
We used some of the available transforms from albumentations and the model did perform well. Even though there was some overfitting in the later epochs, adding image
augmentation techniques did reduce the gap between train and test accuracy.

On top of that, we used [Grad-CAM](http://gradcam.cloudcv.org/) to highlight the regions in the image which are important for the prediction of the models.
The highlighted region marks the pixels areas which the DNN thinks to be most useful. Unlike CAM, Grad-CAM requires no re-training and is broadly applicable to any 
CNN-based architecture.

---

Our target was to achieve >87% accuracy.
We trained the model for 25 epochs and achieved:
* Highest train accuracy: 92.54%
* Highest test accuracy: 88.68%

This is a significant improvement from the S8 code.

---

**Parameters and Hyperparameters**

* Loss function : Cross Entropy Loss
* Optimizer : SGD(learning_rate = 0.005, momentum = 0.9)
* Batch Size : 128
* Epochs : 25
---

**Classwise Accuracy**

* Accuracy of plane : 96 %
* Accuracy of   car : 86 %
* Accuracy of  bird : 83 %
* Accuracy of   cat : 53 % (least performing)
* Accuracy of  deer : 86 %
* Accuracy of   dog : 93 %
* Accuracy of  frog : 95 %
* Accuracy of horse : 96 %
* Accuracy of  ship : 89 %
* Accuracy of truck : 96 %
---

**Grad-CAM output for the final layer of ResNet18**

![gradcam_output](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S9/images/gradcam_output.png)

---
## Group Members

Rishabh Bhardwaj

Rashu Tyagi

