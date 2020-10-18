# Session 10

Previously in [S10](https://github.com/rishabh-bhardwaj-64rr/EVA5/tree/master/S10), we trained ResNet18 on CIFAR-10 dataset by using [lr_finder](https://github.com/davidtvs/pytorch-lr-finder) for SGD with momentum and also [`ReduceLROnPlateau`] (https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau) to reach > 88% accuracy

In this code, we are going one step further. Our objective was to reach > 90% validation accuracy by implementing a custom network [(DavidNet)](https://medium.com/fenwicks/tutorial-2-94-accuracy-on-cifar10-in-2-minutes-7b5aaecd9cdd) and using [SuperConvergence](https://arxiv.org/abs/1708.07120) and [OneCycleLR](https://arxiv.org/abs/1803.09820). The author of these papers, Leslie Smith describes the approach to set hyper-parameters (namely learning rate, momentum and weight decay) and batch size. In particular, he suggests One Cycle policy to apply learning rates.

In this method of training, one cycle means that we first go from low learning rate to some high learning rate (typically, highest_lr/lowest_lr >= 5) in just a few epochs(5 - 10 epochs typically). Then we go down from highest learning rate to the lowest learning rate in the same number of epochs. We may then choose to annihilate the learning rate further after this cycle completes to way below the lower learning rate value(1/10 th or 1/100 th).

The motivation behind this is that, during the middle of learning when the learning rate is higher, the learning rate
works as a regularisation method and keep the network from overfitting. This helps the network to avoid steep areas
of loss and land better flatter minima. The accuracy increases dramatically during the first half of cycle when the lr increases.

But does it provide us higher accuracy after training in practice? NO

So why use One Cycle Policy?

* It reduces the time it takes to reach "near" to your accuracy. 

* It allows us to know if we are going right early on. 

* It let us know what kind of accuracies we can target with a given model.

* It reduces the cost of training. 

* It reduces the time to deploy!
 


---

We trained the model for 25 epochs and at the 5th epoch, the learning rate was maximum so the cycle was of 10 epochs. The output metrics were:
* Highest train accuracy: 96.80%
* Highest test accuracy: 90.76%
* Test accuracy after 5th epoch = 71.25%

---

**lr-finder output**

We also used lr-finder to find the maximum learning rate for training

```
optimizer = optim.SGD(net.parameters(), lr=0.01,  momentum=0.9)
criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, start_lr=1e-3, end_lr=0.1, num_iter=400, step_mode='linear')

lr_finder.plot() # loss vs lr curve

best_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
print(best_lr)

lr_finder.reset()

```
* Suggested LR: 5.71E-03 (the learning rate corresponding to maximum gradient in loss vs lr curve)
* best_lr = 0.06600751879699249 (the learning rate corresponding to minimum loss after the range test completes)
---

**Parameters and Hyperparameters**

```
Batch size = 512
EPOCHS = 25
max_lr_epoch = 5
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
pct_start = max_lr_epoch/EPOCHS
scheduler = OneCycleLR(optimizer=optimizer, max_lr=best_lr, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=pct_start,anneal_strategy='linear', div_factor=200, final_div_factor=1)

```

---

**Classwise Accuracy**


* Accuracy of plane : 100 %
* Accuracy of   car : 100 %
* Accuracy of  bird : 80 %
* Accuracy of   cat : 75 %
* Accuracy of  deer : 85 %
* Accuracy of   dog : 83 %
* Accuracy of  frog : 84 %
* Accuracy of horse : 87 %
* Accuracy of  ship : 100 %
* Accuracy of truck : 100 %

---


