# CAPSTONE PROJECT

The project was to make a network that can do two tasks simultaneously.

* Predict the depth map of the image
* Predict the boots, vest, hardhat, and mask if there is an image


## YOLOV3

## MiDaS

## Dataset

## Additional Dataset

## Basic Approach

* In Session 13 and Session 14, we used the YOLOV3 network for object detection and the MiDaS network for depth prediction respectively on the same dataset (boots, vest, harhat, and mask dataset).
Both of these use encoder-decoder architectures to output the images.

* YOLOV3 architecture uses Darknet-53 as its encoder and MiDaS uses ResNext-101 as its encoder.

* We know that to get a really good output, we need an encoder-decoder architecture.

* So, the first step was to combine both these networks using one encoder only. I chose the MiDaS encoder and removed the Darknet-53 part of YOLOV3.
  This seemingly easy task took some time to function smoothly.
  
* So now we have one encoder and two decoders. The MiDaS network is as it is and the YOLOV3 has been modified.
 
* Since both Darknet-53 and ResNext-101 have been trained on ImageNet, we can expect decent results overall if we use one for the other.

* One of the important approach this project was to deal with least number of problems. We need not touch the MiDaS network since we might not get better results than what the original authors had got. So we need to deal more with the yolo network to get better results.

* Even though replacing YOLOV3 encoder with the MiDaS encoder might not have made much of a difference, it was necessary to compensate so I added some additional convolutional layers between the encoder and the YOLOV3 heads keeping the other architecture same.

* For the training part, I needed to train on smaller images with highest possible batch size before jumping on to bigger images.

* As much as possible I used the trained weights and the same architecture. I do have 1 decoder untrained branch with trained heads (trained on our dataset from Session 13), but untrained custom layers. So I will focus more on these custom layers.

* I trained these custom layers and the YOLOV3 heads for a good number of epochs and froze the rest of the layers of this mega network.


## Network

visualisation

## Training

* Since I froze the MiDaS part, I only had to train YOLOV3's head and the custom layers before them.

* Instead of training on required image size(s) first, it is better to train on smaller resolution first since training on smaller images is fast and we can leverage what the network learns on smaller resolution when jumping on bigger images. So the overall training is faster.

```
!python train.py  --data data/customdata/custom.data --batch batch_size --resume --depth 0 --yolo 1 --cache --cfg cfg/yolov3-custom.cfg --epochs number_of_epochs --img-size image_size --weights='/content/gdrive/MyDrive/S15_NEW/yolov3/weights/mega_model_weights.pt'
```

* So I trained on image resolution of 64 with batch size of 512 for 300 epochs. The total number of training images were 3186/3590.

* Then I trained on resolution of 128 with batch size of 128 for 100 epochs. Then again I jumped to resolution of 256 with batch size of 32 for 50 epochs.

* Finally, I trained on resolution of 448 and 512 with batch size of 16 and 10 respectively for 10 epochs each.

* The highest mAP I achived was 0.308 when training on 512 sized images. And the lowest loss I got was 3.76 (sum of GIoU loss, objectness loss and classification loss of YOLOV3).

* It took around 12 hours to go through this training on Colab including testing on 386 images after each epoch. (GPU: Tesla T4, total_memory=15079MB).

* The training code file also includes training on depth images. If there is a need to unfreeze the pretrained encoder to see some kind of increase in accuracy in the the overall prediction, we need to cater for the output of the depth network also. So we can train both the YOLOV3 layers and the MiDaS network by putting `--depth 0` when calling the training script.

* I also added more training options in the `train.py` file.

* Refer to (.ipynb file)[link to file] for more details.


### Data Augmentation
>
> - I used the original YOLOV3 data augmentation techniques for train dataset (random_affine(rotation=1.98, translate=0.05, scale=0.05, shear=0.641), flip-lr(p=0.5, hsv_augment(hue gain=0.0138, saturation gain=0.678, value=0.36)).
>
> - This can be found (yolo/utils/dataset)[].

### Loss Functions

>  **BCEWithLogitsLoss**
>
>  - YOLOV3's loss function consists of 5 different loss functions and categorically 3 losses (objectness loss, classification loss, and IoU loss (Intersection over Union)). The objectness and classification losses use BCEWithLogitsLoss. 

> link to yolo loss function explanation, image
>
> - The total loss of YOLOV3 is the summation of obj, cls, and GIoU losses.
>
>  - link to yolo/utils/compute_loss

>  **SSIM Loss**
>
>  - The Structural SIMilarity (SSIM) index is a method for measuring the similarity between two images. The SSIM index can be viewed as a quality measure of one of the images being compared, provided the other image is regarded as of perfect quality.
>
>  - This loss function is used when unfreezing the encoder part to allow the gradients to flow back through the encoder. Since this will affect the prediction of the depth images, it is necessary to keep a check on the prediction of depth images. The ground truth images were the images from the (link to Session 14 depth images).


>  **MSE Loss**

> - The Mean Squared Error (MSE), also called L2 Loss, computes the average of the squared differences between actual values and predicted pixel values. 

>  `total_loss = lambda_yolo * yolo_loss + lambda_depth * (lambda_ssim * ssim_loss + lambda_mse * mse_loss)`
>  These lambdas control the weightage given to each loss so as to ensure some kind of parity amongst all the loss values.


### Optimizers, Scheduler and hyperparameters

YOLOV3 uses SGD optimizer and learning rate scheduler as LamdaLR with cosine annealing.

>  `optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)`

> `lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf`
        
> `scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch = -1)`
>
Hyperparameters: 

> `'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)`
>
>  ` 'lrf': 0.0005,  # final learning rate (with cos scheduler)`
>
> ` 'depth_lr': 0.0005, # learning rate when training the depth network`
>
>   `'momentum': 0.937,  # SGD momentum`
>
> `'weight_decay': 0.000484,  # optimizer weight decay`


## Testing and Results

* link the original outputs from original models

* To test the output of YOLOV3, I just ran the `detect.py` file in the YOLOV3 folder.

* I tested on the same dataset with image size of 512 and got some decent results.

* link insert yolo output results

* To test the output of MiDaS, I ran the `run.py` file in the midas folder and it achieved good results.

* link to midas output images


## Code Structure

* Most of the code structure is same as the individual folders of YOLOV3 and MiDaS.

| Path | Description|
| --- | ---|
| S15_Capstone.ipynb | Notebook for training and testing. |
| mega_model.py | The model used. |
| encoder_utils | This folder contains the utilities for the encoder i.e. ResNext-101.|
| midas/run.py | The file used to run inference directly to get the depth images. |
| yolo/train.py | The file used to train the whole network. |
| yolo/yolov3-custom.cfg | The config file for yolo. Since the network has been created without this cfg file. There is hardly any use of it. |
| yolo/customdata/images | This folder contains the images to train upon. |
| yolo/customdata/labels | This folder contains the .txt files to which contain the positions of hardhat, mask, vest, and boots in the annotated images. |
| yolo/customdata/custom.data | The file containing the number of classes in the dataset (4 classes) and the paths to files containing train and test files. |
| yolo/customdata/custom.names | The file containing the names of classes. |
| yolo/customdata/cusom.shapes | The file containing the shapes of images. |
| yolo/customdata/train.txt | The file containing paths to train images. |
| yolo/customdata/test.txt | The file containing paths to test images. |
| yolo/weights/ | The folder containing the weights (mega_model_weights.pt (the original weights) and last.pt (trained weights) |
| yolo/detect.py | The file used to run inference directly to get the yolo output (bbox depicting hardhat, vest, mask, and boots). |
| yolo/utils | The folder containing the dataloader file (datasets.py) and the loss functions and other bounding box utilities file (utils.py), etc. |
| yolo/output | The folder containing the inference outputs of yolo. |

## Problems Faced 

* The dataset, the model weights, and their outputs take up quite some space. So 15GB gdrive space was not sufficient. I was fortunate enough to have my college's email id due to which I was able to use more gdrive space. But one may get away with gdrive space limitations if space is used very wisely.

* Since there was so much disparity between the code structures of YOLOV3 model and MiDaS model, initially it was quite difficult to link up the two. That took some time.

* I had to patch up the code at so many places so that the whole pipeline atleast becomes trainable. It would have been good if I coded up some files from scratch :sweat_smile: .

## More things to try and What I would do if I get the chance of solving the same problem

* Understand the code pipeline of YOLOV3 at the earliest rather than just jumping into devising some solution for the task at hand.

* At first, I tried to make separate encoder and decoder parts into different files. It worked but ran into some kind of problem one day or the other. I wasted some time on this. Then I quickly switched to write the model in one file only.

* Even though the task seems to be just linking two models and get the results but that is just the tip of the icerberg. But with right strategy, this project could have been done in much lesser time ¯\\_(ツ)_/¯ . I had to experiment a lot about what works and what doesn't.

* I tried to use OneCycleLR policy but could not integrate it with my approach since some of the weights in the network update with different learning rate and the pretrained layers (if trained upon) work with much less learning rate. 

* I could have explored more types of loss functions.

* The GIoU loss from YOLOV3 is a bit high (min 2.26 on image size of 448) as compared to other losses. So if I could have lowered it somehow, the predictions could have been a bit better.


## Key Takeways 

* This project has been an immense learning opportunity to me. Each type of error or the different strategies to solve a sub-problem and probing different QnA forums enriched my knowledge in this domain.

* I am grateful to be a part of EVA5 under an excellent mentor **Rohan Shravan**.











    
 














