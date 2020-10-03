# Session 10

Previously in [S9](https://github.com/rishabh-bhardwaj-64rr/EVA5/tree/master/S9), we trained ResNet18 on CIFAR-10 dataset and used albumentations(https://github.com/albumentations-team/albumentations) library. We reached > 87% accuracy under 25 epochs in that code.

In this code, our objective was to reach > 88% validation accuracy by using [lr_finder](https://github.com/davidtvs/pytorch-lr-finder) code for SGD with momentum and also [`ReduceLROnPlateau`] (https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau).


On top of that, we used [Grad-CAM](http://gradcam.cloudcv.org/) on the misclassified images to highlight the regions in the image which are important for the prediction of the models.
The highlighted region marks the pixels areas which the DNN thinks to be most useful. Unlike CAM, Grad-CAM requires no re-training and is broadly applicable to any 
CNN-based architecture.

---

We trained the model for 50 epochs and achieved:
* Highest train accuracy: 96.42%
* Highest test accuracy: 91.65%

---

**lr-finder output**
![lr-finder output](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S10/images/lr_finder%20plot.png)

```
optimizer = optim.SGD(net.parameters(), lr=1e-7,  momentum=0.9)
criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

lr_finder.plot() # loss vs lr curve

lr_finder.reset()

```
* Suggested learning rate: 1.23E-02

---

**Parameters and Hyperparameters**

* Loss function : Cross Entropy Loss
* Optimizer : `SGD(learning_rate = 1.23E-02, momentum = 0.9)`
* Scheduler : `ReduceLROnPlateau(optimizer, patience = 3)`
* Batch Size : 128
* Epochs : 50
---

**Classwise Accuracy**

* Accuracy of plane : 89 %
* Accuracy of   car : 97 %
* Accuracy of  bird : 80 %
* Accuracy of   cat : 88 %
* Accuracy of  deer : 81 %
* Accuracy of   dog : 82 %
* Accuracy of  frog : 100 %
* Accuracy of horse : 91 %
* Accuracy of  ship : 97 %
* Accuracy of truck : 100 %
---

**Grad-CAM on misclassified images for the all the layers of ResNet18**

![gradcam_output](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S10/images/gradcam_misclassifications_5.png)

---
## Group Members

Rishabh Bhardwaj

Rashu Tyagi

