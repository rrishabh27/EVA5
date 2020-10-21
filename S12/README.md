# Session 12

In this code, our objective was to reach > 50% validation accuracy on [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) having 70/30 split using ResNet-18 model. It has 200 classes and is a difficult dataset to train on. One can achieve great accuracy while training but it is very diffcult to get > 60% validation accuracy on this dataset.


Other part of code deals with preliminary task for object detection using YOLO. We built a custom dataset by collecting the images of people wearing hardhat, vest, mask and boots (atleast 50 classes each) and annotating them. We used [this](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) tool to annotate bounding boxes around the hardhat, vest, mask and boots. Further, we calculated the appropriate number of clusters (or number of anchor boxes) for the dataset for training YOLO network using k-means clustering applied on contents of the JSON file we got after annotation. 

---

For the first taks, we trained the model for 50 epochs and achieved:
* Highest train accuracy: 97.32%
* Highest test accuracy: 55.72% (HEAVY OVERFITTING)

We also used `OneCycleLR` with the learning rate being maximum (= 0.05) at 6th epoch. It helped the model quickly reach 42.65% accuracy in 12 epochs.

For the second task, the appropriate number of clusters found were 4 using elbow method.

---

**Parameters and Hyperparameters**

* Loss function : Cross Entropy Loss
* Optimizer : `SGD(net.parameters(), lr = 0.01, momentum = 0.9)`
* Scheduler : `OneCycleLR(optimizer, max_lr=0.05, epochs=50, steps_per_epoch=len(train_loader), pct_start=6/50,anneal_strategy='cos', div_factor=10, final_div_factor=1)`
* Batch Size : 512
* Epochs : 50

---
