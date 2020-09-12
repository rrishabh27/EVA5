In this code, we ran below versions of each of the combinations of the models on the previous session's code for 25 epochs each in one loop over MNIST dataset:

* with L1 regularisation + BN(Batch Normalisation)
* with L2 regularisation + BN
* with L1 and L2 with BN
* with GBN([Ghost Batch Normalisation code](https://github.com/apple/ml-cifar-10-faster/blob/master/utils.py))
* with L1 and L2 with GBN

Due to GPU limitations, training for only 20 epochs each could be done.

* Even though it might not be immediately clear that L1 and L2 regularisations are not much helpful, L1 and L2 are much useful for fully connected layers.
* Ghost Batch Normalisation has power as a regularizer: due to
the stochasticity in normalization statistics caused by the random selection of mini-batches during
training.
* Surprisingly, just using this one simple technique was capable
of improving performance by 5.8% on Caltech-256 and **0.84% on CIFAR-100, which is remarkable
given it has no additional cost during training.**

**Validation Accuracy and Validation Losses:**

![Val Acc](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S6/images/Val_acc.png)

![Val Losses](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S6/images/Val_losses.png)

**25 misclassified images by using GBN**

![25 miss GBN](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S6/images/25%20miss%20GBN.png)
