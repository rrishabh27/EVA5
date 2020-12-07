# CAPSTONE PROJECT

The project was to make a network that can do two tasks simultaneously.

* Predict the depth map of the image
* Predict the boots, vest, hardhat, and mask if there is an image


## YOLOV3

## MiDaS

## Dataset

* The [dataset](https://drive.google.com/file/d/1EqtOpF7cS74C56EaVQoKNkQmpT6_HFL2/view) I used to train YOLOV3 contains 3591 images of people wearing hardhat, mask, vest, and boots.

* I also collected some 5000 additional images of interiors which I could have used to train the depth network if I did not get good results on the depth images.

* [Full Dataset](https://drive.google.com/drive/folders/1bbYky82pAewASV54030mNTnX3wfVPPge?usp=sharing) (around 8.5k images).


## Basic Approach

* In [Session 13]()https://github.com/rishabh-bhardwaj-64rr/EVA5/tree/master/S13 and [Session 14](https://github.com/rishabh-bhardwaj-64rr/EVA5/tree/master/S14), we used the YOLOV3 network for object detection and the MiDaS network for depth prediction respectively on the same dataset (boots, vest, hardhat, and mask dataset).
Both of them use encoder-decoder architectures to output the images.

* YOLOV3 architecture uses Darknet-53 as its encoder and MiDaS uses ResNext-101 as its encoder.

* We know that to get a really good output, we need an encoder-decoder architecture.

* So, the first step was to combine both these networks using one encoder only. I chose the MiDaS encoder and removed the Darknet-53 part of YOLOV3.
  This seemingly easy task took some time to function smoothly.
  
* So now we have one encoder and two decoders. The MiDaS network is as it is and the YOLOV3 has been modified.
 
* Since both Darknet-53 and ResNext-101 have been pretrained on ImageNet, we can expect decent results overall if we use one for the other.

* One of the important approach for this project was to deal with the least number of problems. We need not touch the MiDaS network since we might not get better results than what the original authors had got. So we need to deal more with the yolo network to get better results. If there is some stagnation, then we can actually train the MiDaS network.

* Even though replacing YOLOV3 encoder with the MiDaS encoder might not have made much of a difference, it was necessary to compensate so I added some additional convolutional layers between the encoder and the YOLOV3 heads keeping the other architecture same.

* For the training part, I needed to train on smaller images with highest possible batch size before jumping on to bigger images.

* As much as possible I used the trained weights and the same architecture. I do have 1 decoder untrained branch with trained heads (trained on our dataset from Session 13), but untrained custom layers. So I will focus more on these custom layers.

* I trained these custom layers and the YOLOV3 heads for a good number of epochs and froze the rest of the layers of this mega network.


## Network


#### Architecture

![model](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/capstone3.png)


#### Summary

```

Layer                                                 Output Shape  \
                                                                   
0_pretrained.layer1.Conv2d_0                        [1, 64, 208, 208]   
1_pretrained.layer1.BatchNorm2d_1                   [1, 64, 208, 208]   
2_pretrained.layer1.ReLU_2                          [1, 64, 208, 208]   
3_pretrained.layer1.MaxPool2d_3                     [1, 64, 104, 104]   
4_pretrained.layer1.4.0.Conv2d_conv1               [1, 256, 104, 104]   
5_pretrained.layer1.4.0.BatchNorm2d_bn1            [1, 256, 104, 104]   
6_pretrained.layer1.4.0.ReLU_relu                  [1, 256, 104, 104]   
7_pretrained.layer1.4.0.Conv2d_conv2               [1, 256, 104, 104]   
8_pretrained.layer1.4.0.BatchNorm2d_bn2            [1, 256, 104, 104]   
9_pretrained.layer1.4.0.ReLU_relu                  [1, 256, 104, 104]   
10_pretrained.layer1.4.0.Conv2d_conv3              [1, 256, 104, 104]   
11_pretrained.layer1.4.0.BatchNorm2d_bn3           [1, 256, 104, 104]   
12_pretrained.layer1.4.0.downsample.Conv2d_0       [1, 256, 104, 104]   
13_pretrained.layer1.4.0.downsample.BatchNorm2d_1  [1, 256, 104, 104]   
14_pretrained.layer1.4.0.ReLU_relu                 [1, 256, 104, 104]   
15_pretrained.layer1.4.1.Conv2d_conv1              [1, 256, 104, 104]   
16_pretrained.layer1.4.1.BatchNorm2d_bn1           [1, 256, 104, 104]   
17_pretrained.layer1.4.1.ReLU_relu                 [1, 256, 104, 104]   
18_pretrained.layer1.4.1.Conv2d_conv2              [1, 256, 104, 104]   
19_pretrained.layer1.4.1.BatchNorm2d_bn2           [1, 256, 104, 104]   
20_pretrained.layer1.4.1.ReLU_relu                 [1, 256, 104, 104]   
21_pretrained.layer1.4.1.Conv2d_conv3              [1, 256, 104, 104]   
22_pretrained.layer1.4.1.BatchNorm2d_bn3           [1, 256, 104, 104]   
23_pretrained.layer1.4.1.ReLU_relu                 [1, 256, 104, 104]   
24_pretrained.layer1.4.2.Conv2d_conv1              [1, 256, 104, 104]   
25_pretrained.layer1.4.2.BatchNorm2d_bn1           [1, 256, 104, 104]   
26_pretrained.layer1.4.2.ReLU_relu                 [1, 256, 104, 104]   
27_pretrained.layer1.4.2.Conv2d_conv2              [1, 256, 104, 104]   
28_pretrained.layer1.4.2.BatchNorm2d_bn2           [1, 256, 104, 104]   
29_pretrained.layer1.4.2.ReLU_relu                 [1, 256, 104, 104]   
30_pretrained.layer1.4.2.Conv2d_conv3              [1, 256, 104, 104]   
31_pretrained.layer1.4.2.BatchNorm2d_bn3           [1, 256, 104, 104]   
32_pretrained.layer1.4.2.ReLU_relu                 [1, 256, 104, 104]   
33_pretrained.layer2.0.Conv2d_conv1                [1, 512, 104, 104]   
34_pretrained.layer2.0.BatchNorm2d_bn1             [1, 512, 104, 104]   
35_pretrained.layer2.0.ReLU_relu                   [1, 512, 104, 104]   
36_pretrained.layer2.0.Conv2d_conv2                  [1, 512, 52, 52]   
37_pretrained.layer2.0.BatchNorm2d_bn2               [1, 512, 52, 52]   
38_pretrained.layer2.0.ReLU_relu                     [1, 512, 52, 52]   
39_pretrained.layer2.0.Conv2d_conv3                  [1, 512, 52, 52]   
40_pretrained.layer2.0.BatchNorm2d_bn3               [1, 512, 52, 52]   
41_pretrained.layer2.0.downsample.Conv2d_0           [1, 512, 52, 52]   
42_pretrained.layer2.0.downsample.BatchNorm2d_1      [1, 512, 52, 52]   
43_pretrained.layer2.0.ReLU_relu                     [1, 512, 52, 52]   
44_pretrained.layer2.1.Conv2d_conv1                  [1, 512, 52, 52]   
45_pretrained.layer2.1.BatchNorm2d_bn1               [1, 512, 52, 52]   
46_pretrained.layer2.1.ReLU_relu                     [1, 512, 52, 52]   
47_pretrained.layer2.1.Conv2d_conv2                  [1, 512, 52, 52]   
48_pretrained.layer2.1.BatchNorm2d_bn2               [1, 512, 52, 52]   
49_pretrained.layer2.1.ReLU_relu                     [1, 512, 52, 52]   
50_pretrained.layer2.1.Conv2d_conv3                  [1, 512, 52, 52]   
51_pretrained.layer2.1.BatchNorm2d_bn3               [1, 512, 52, 52]   
52_pretrained.layer2.1.ReLU_relu                     [1, 512, 52, 52]   
53_pretrained.layer2.2.Conv2d_conv1                  [1, 512, 52, 52]   
54_pretrained.layer2.2.BatchNorm2d_bn1               [1, 512, 52, 52]   
55_pretrained.layer2.2.ReLU_relu                     [1, 512, 52, 52]   
56_pretrained.layer2.2.Conv2d_conv2                  [1, 512, 52, 52]   
57_pretrained.layer2.2.BatchNorm2d_bn2               [1, 512, 52, 52]   
58_pretrained.layer2.2.ReLU_relu                     [1, 512, 52, 52]   
59_pretrained.layer2.2.Conv2d_conv3                  [1, 512, 52, 52]   
60_pretrained.layer2.2.BatchNorm2d_bn3               [1, 512, 52, 52]   
61_pretrained.layer2.2.ReLU_relu                     [1, 512, 52, 52]   
62_pretrained.layer2.3.Conv2d_conv1                  [1, 512, 52, 52]   
63_pretrained.layer2.3.BatchNorm2d_bn1               [1, 512, 52, 52]   
64_pretrained.layer2.3.ReLU_relu                     [1, 512, 52, 52]   
65_pretrained.layer2.3.Conv2d_conv2                  [1, 512, 52, 52]   
66_pretrained.layer2.3.BatchNorm2d_bn2               [1, 512, 52, 52]   
67_pretrained.layer2.3.ReLU_relu                     [1, 512, 52, 52]   
68_pretrained.layer2.3.Conv2d_conv3                  [1, 512, 52, 52]   
69_pretrained.layer2.3.BatchNorm2d_bn3               [1, 512, 52, 52]   
70_pretrained.layer2.3.ReLU_relu                     [1, 512, 52, 52]   
71_pretrained.layer3.0.Conv2d_conv1                 [1, 1024, 52, 52]   
72_pretrained.layer3.0.BatchNorm2d_bn1              [1, 1024, 52, 52]   
73_pretrained.layer3.0.ReLU_relu                    [1, 1024, 52, 52]   
74_pretrained.layer3.0.Conv2d_conv2                 [1, 1024, 26, 26]   
75_pretrained.layer3.0.BatchNorm2d_bn2              [1, 1024, 26, 26]   
76_pretrained.layer3.0.ReLU_relu                    [1, 1024, 26, 26]   
77_pretrained.layer3.0.Conv2d_conv3                 [1, 1024, 26, 26]   
78_pretrained.layer3.0.BatchNorm2d_bn3              [1, 1024, 26, 26]   
79_pretrained.layer3.0.downsample.Conv2d_0          [1, 1024, 26, 26]   
80_pretrained.layer3.0.downsample.BatchNorm2d_1     [1, 1024, 26, 26]   
81_pretrained.layer3.0.ReLU_relu                    [1, 1024, 26, 26]   
82_pretrained.layer3.1.Conv2d_conv1                 [1, 1024, 26, 26]   
83_pretrained.layer3.1.BatchNorm2d_bn1              [1, 1024, 26, 26]   
84_pretrained.layer3.1.ReLU_relu                    [1, 1024, 26, 26]   
85_pretrained.layer3.1.Conv2d_conv2                 [1, 1024, 26, 26]   
86_pretrained.layer3.1.BatchNorm2d_bn2              [1, 1024, 26, 26]   
87_pretrained.layer3.1.ReLU_relu                    [1, 1024, 26, 26]   
88_pretrained.layer3.1.Conv2d_conv3                 [1, 1024, 26, 26]   
89_pretrained.layer3.1.BatchNorm2d_bn3              [1, 1024, 26, 26]   
90_pretrained.layer3.1.ReLU_relu                    [1, 1024, 26, 26]   
91_pretrained.layer3.2.Conv2d_conv1                 [1, 1024, 26, 26]   
92_pretrained.layer3.2.BatchNorm2d_bn1              [1, 1024, 26, 26]   
93_pretrained.layer3.2.ReLU_relu                    [1, 1024, 26, 26]   
94_pretrained.layer3.2.Conv2d_conv2                 [1, 1024, 26, 26]   
95_pretrained.layer3.2.BatchNorm2d_bn2              [1, 1024, 26, 26]   
96_pretrained.layer3.2.ReLU_relu                    [1, 1024, 26, 26]   
97_pretrained.layer3.2.Conv2d_conv3                 [1, 1024, 26, 26]   
98_pretrained.layer3.2.BatchNorm2d_bn3              [1, 1024, 26, 26]   
99_pretrained.layer3.2.ReLU_relu                    [1, 1024, 26, 26]   
100_pretrained.layer3.3.Conv2d_conv1                [1, 1024, 26, 26]   
101_pretrained.layer3.3.BatchNorm2d_bn1             [1, 1024, 26, 26]   
102_pretrained.layer3.3.ReLU_relu                   [1, 1024, 26, 26]   
103_pretrained.layer3.3.Conv2d_conv2                [1, 1024, 26, 26]   
104_pretrained.layer3.3.BatchNorm2d_bn2             [1, 1024, 26, 26]   
105_pretrained.layer3.3.ReLU_relu                   [1, 1024, 26, 26]   
106_pretrained.layer3.3.Conv2d_conv3                [1, 1024, 26, 26]   
107_pretrained.layer3.3.BatchNorm2d_bn3             [1, 1024, 26, 26]   
108_pretrained.layer3.3.ReLU_relu                   [1, 1024, 26, 26]   
109_pretrained.layer3.4.Conv2d_conv1                [1, 1024, 26, 26]   
110_pretrained.layer3.4.BatchNorm2d_bn1             [1, 1024, 26, 26]   
111_pretrained.layer3.4.ReLU_relu                   [1, 1024, 26, 26]   
112_pretrained.layer3.4.Conv2d_conv2                [1, 1024, 26, 26]   
113_pretrained.layer3.4.BatchNorm2d_bn2             [1, 1024, 26, 26]   
114_pretrained.layer3.4.ReLU_relu                   [1, 1024, 26, 26]   
115_pretrained.layer3.4.Conv2d_conv3                [1, 1024, 26, 26]   
116_pretrained.layer3.4.BatchNorm2d_bn3             [1, 1024, 26, 26]   
117_pretrained.layer3.4.ReLU_relu                   [1, 1024, 26, 26]   
118_pretrained.layer3.5.Conv2d_conv1                [1, 1024, 26, 26]   
119_pretrained.layer3.5.BatchNorm2d_bn1             [1, 1024, 26, 26]   
120_pretrained.layer3.5.ReLU_relu                   [1, 1024, 26, 26]   
121_pretrained.layer3.5.Conv2d_conv2                [1, 1024, 26, 26]   
122_pretrained.layer3.5.BatchNorm2d_bn2             [1, 1024, 26, 26]   
123_pretrained.layer3.5.ReLU_relu                   [1, 1024, 26, 26]   
124_pretrained.layer3.5.Conv2d_conv3                [1, 1024, 26, 26]   
125_pretrained.layer3.5.BatchNorm2d_bn3             [1, 1024, 26, 26]   
126_pretrained.layer3.5.ReLU_relu                   [1, 1024, 26, 26]   
127_pretrained.layer3.6.Conv2d_conv1                [1, 1024, 26, 26]   
128_pretrained.layer3.6.BatchNorm2d_bn1             [1, 1024, 26, 26]   
129_pretrained.layer3.6.ReLU_relu                   [1, 1024, 26, 26]   
130_pretrained.layer3.6.Conv2d_conv2                [1, 1024, 26, 26]   
131_pretrained.layer3.6.BatchNorm2d_bn2             [1, 1024, 26, 26]   
132_pretrained.layer3.6.ReLU_relu                   [1, 1024, 26, 26]   
133_pretrained.layer3.6.Conv2d_conv3                [1, 1024, 26, 26]   
134_pretrained.layer3.6.BatchNorm2d_bn3             [1, 1024, 26, 26]   
135_pretrained.layer3.6.ReLU_relu                   [1, 1024, 26, 26]   
136_pretrained.layer3.7.Conv2d_conv1                [1, 1024, 26, 26]   
137_pretrained.layer3.7.BatchNorm2d_bn1             [1, 1024, 26, 26]   
138_pretrained.layer3.7.ReLU_relu                   [1, 1024, 26, 26]   
139_pretrained.layer3.7.Conv2d_conv2                [1, 1024, 26, 26]   
140_pretrained.layer3.7.BatchNorm2d_bn2             [1, 1024, 26, 26]   
141_pretrained.layer3.7.ReLU_relu                   [1, 1024, 26, 26]   
142_pretrained.layer3.7.Conv2d_conv3                [1, 1024, 26, 26]   
143_pretrained.layer3.7.BatchNorm2d_bn3             [1, 1024, 26, 26]   
144_pretrained.layer3.7.ReLU_relu                   [1, 1024, 26, 26]   
145_pretrained.layer3.8.Conv2d_conv1                [1, 1024, 26, 26]   
146_pretrained.layer3.8.BatchNorm2d_bn1             [1, 1024, 26, 26]   
147_pretrained.layer3.8.ReLU_relu                   [1, 1024, 26, 26]   
148_pretrained.layer3.8.Conv2d_conv2                [1, 1024, 26, 26]   
149_pretrained.layer3.8.BatchNorm2d_bn2             [1, 1024, 26, 26]   
150_pretrained.layer3.8.ReLU_relu                   [1, 1024, 26, 26]   
151_pretrained.layer3.8.Conv2d_conv3                [1, 1024, 26, 26]   
152_pretrained.layer3.8.BatchNorm2d_bn3             [1, 1024, 26, 26]   
153_pretrained.layer3.8.ReLU_relu                   [1, 1024, 26, 26]   
154_pretrained.layer3.9.Conv2d_conv1                [1, 1024, 26, 26]   
155_pretrained.layer3.9.BatchNorm2d_bn1             [1, 1024, 26, 26]   
156_pretrained.layer3.9.ReLU_relu                   [1, 1024, 26, 26]   
157_pretrained.layer3.9.Conv2d_conv2                [1, 1024, 26, 26]   
158_pretrained.layer3.9.BatchNorm2d_bn2             [1, 1024, 26, 26]   
159_pretrained.layer3.9.ReLU_relu                   [1, 1024, 26, 26]   
160_pretrained.layer3.9.Conv2d_conv3                [1, 1024, 26, 26]   
161_pretrained.layer3.9.BatchNorm2d_bn3             [1, 1024, 26, 26]   
162_pretrained.layer3.9.ReLU_relu                   [1, 1024, 26, 26]   
163_pretrained.layer3.10.Conv2d_conv1               [1, 1024, 26, 26]   
164_pretrained.layer3.10.BatchNorm2d_bn1            [1, 1024, 26, 26]   
165_pretrained.layer3.10.ReLU_relu                  [1, 1024, 26, 26]   
166_pretrained.layer3.10.Conv2d_conv2               [1, 1024, 26, 26]   
167_pretrained.layer3.10.BatchNorm2d_bn2            [1, 1024, 26, 26]   
168_pretrained.layer3.10.ReLU_relu                  [1, 1024, 26, 26]   
169_pretrained.layer3.10.Conv2d_conv3               [1, 1024, 26, 26]   
170_pretrained.layer3.10.BatchNorm2d_bn3            [1, 1024, 26, 26]   
171_pretrained.layer3.10.ReLU_relu                  [1, 1024, 26, 26]   
172_pretrained.layer3.11.Conv2d_conv1               [1, 1024, 26, 26]   
173_pretrained.layer3.11.BatchNorm2d_bn1            [1, 1024, 26, 26]   
174_pretrained.layer3.11.ReLU_relu                  [1, 1024, 26, 26]   
175_pretrained.layer3.11.Conv2d_conv2               [1, 1024, 26, 26]   
176_pretrained.layer3.11.BatchNorm2d_bn2            [1, 1024, 26, 26]   
177_pretrained.layer3.11.ReLU_relu                  [1, 1024, 26, 26]   
178_pretrained.layer3.11.Conv2d_conv3               [1, 1024, 26, 26]   
179_pretrained.layer3.11.BatchNorm2d_bn3            [1, 1024, 26, 26]   
180_pretrained.layer3.11.ReLU_relu                  [1, 1024, 26, 26]   
181_pretrained.layer3.12.Conv2d_conv1               [1, 1024, 26, 26]   
182_pretrained.layer3.12.BatchNorm2d_bn1            [1, 1024, 26, 26]   
183_pretrained.layer3.12.ReLU_relu                  [1, 1024, 26, 26]   
184_pretrained.layer3.12.Conv2d_conv2               [1, 1024, 26, 26]   
185_pretrained.layer3.12.BatchNorm2d_bn2            [1, 1024, 26, 26]   
186_pretrained.layer3.12.ReLU_relu                  [1, 1024, 26, 26]   
187_pretrained.layer3.12.Conv2d_conv3               [1, 1024, 26, 26]   
188_pretrained.layer3.12.BatchNorm2d_bn3            [1, 1024, 26, 26]   
189_pretrained.layer3.12.ReLU_relu                  [1, 1024, 26, 26]   
190_pretrained.layer3.13.Conv2d_conv1               [1, 1024, 26, 26]   
191_pretrained.layer3.13.BatchNorm2d_bn1            [1, 1024, 26, 26]   
192_pretrained.layer3.13.ReLU_relu                  [1, 1024, 26, 26]   
193_pretrained.layer3.13.Conv2d_conv2               [1, 1024, 26, 26]   
194_pretrained.layer3.13.BatchNorm2d_bn2            [1, 1024, 26, 26]   
195_pretrained.layer3.13.ReLU_relu                  [1, 1024, 26, 26]   
196_pretrained.layer3.13.Conv2d_conv3               [1, 1024, 26, 26]   
197_pretrained.layer3.13.BatchNorm2d_bn3            [1, 1024, 26, 26]   
198_pretrained.layer3.13.ReLU_relu                  [1, 1024, 26, 26]   
199_pretrained.layer3.14.Conv2d_conv1               [1, 1024, 26, 26]   
200_pretrained.layer3.14.BatchNorm2d_bn1            [1, 1024, 26, 26]   
201_pretrained.layer3.14.ReLU_relu                  [1, 1024, 26, 26]   
202_pretrained.layer3.14.Conv2d_conv2               [1, 1024, 26, 26]   
203_pretrained.layer3.14.BatchNorm2d_bn2            [1, 1024, 26, 26]   
204_pretrained.layer3.14.ReLU_relu                  [1, 1024, 26, 26]   
205_pretrained.layer3.14.Conv2d_conv3               [1, 1024, 26, 26]   
206_pretrained.layer3.14.BatchNorm2d_bn3            [1, 1024, 26, 26]   
207_pretrained.layer3.14.ReLU_relu                  [1, 1024, 26, 26]   
208_pretrained.layer3.15.Conv2d_conv1               [1, 1024, 26, 26]   
209_pretrained.layer3.15.BatchNorm2d_bn1            [1, 1024, 26, 26]   
210_pretrained.layer3.15.ReLU_relu                  [1, 1024, 26, 26]   
211_pretrained.layer3.15.Conv2d_conv2               [1, 1024, 26, 26]   
212_pretrained.layer3.15.BatchNorm2d_bn2            [1, 1024, 26, 26]   
213_pretrained.layer3.15.ReLU_relu                  [1, 1024, 26, 26]   
214_pretrained.layer3.15.Conv2d_conv3               [1, 1024, 26, 26]   
215_pretrained.layer3.15.BatchNorm2d_bn3            [1, 1024, 26, 26]   
216_pretrained.layer3.15.ReLU_relu                  [1, 1024, 26, 26]   
217_pretrained.layer3.16.Conv2d_conv1               [1, 1024, 26, 26]   
218_pretrained.layer3.16.BatchNorm2d_bn1            [1, 1024, 26, 26]   
219_pretrained.layer3.16.ReLU_relu                  [1, 1024, 26, 26]   
220_pretrained.layer3.16.Conv2d_conv2               [1, 1024, 26, 26]   
221_pretrained.layer3.16.BatchNorm2d_bn2            [1, 1024, 26, 26]   
222_pretrained.layer3.16.ReLU_relu                  [1, 1024, 26, 26]   
223_pretrained.layer3.16.Conv2d_conv3               [1, 1024, 26, 26]   
224_pretrained.layer3.16.BatchNorm2d_bn3            [1, 1024, 26, 26]   
225_pretrained.layer3.16.ReLU_relu                  [1, 1024, 26, 26]   
226_pretrained.layer3.17.Conv2d_conv1               [1, 1024, 26, 26]   
227_pretrained.layer3.17.BatchNorm2d_bn1            [1, 1024, 26, 26]   
228_pretrained.layer3.17.ReLU_relu                  [1, 1024, 26, 26]   
229_pretrained.layer3.17.Conv2d_conv2               [1, 1024, 26, 26]   
230_pretrained.layer3.17.BatchNorm2d_bn2            [1, 1024, 26, 26]   
231_pretrained.layer3.17.ReLU_relu                  [1, 1024, 26, 26]   
232_pretrained.layer3.17.Conv2d_conv3               [1, 1024, 26, 26]   
233_pretrained.layer3.17.BatchNorm2d_bn3            [1, 1024, 26, 26]   
234_pretrained.layer3.17.ReLU_relu                  [1, 1024, 26, 26]   
235_pretrained.layer3.18.Conv2d_conv1               [1, 1024, 26, 26]   
236_pretrained.layer3.18.BatchNorm2d_bn1            [1, 1024, 26, 26]   
237_pretrained.layer3.18.ReLU_relu                  [1, 1024, 26, 26]   
238_pretrained.layer3.18.Conv2d_conv2               [1, 1024, 26, 26]   
239_pretrained.layer3.18.BatchNorm2d_bn2            [1, 1024, 26, 26]   
240_pretrained.layer3.18.ReLU_relu                  [1, 1024, 26, 26]   
241_pretrained.layer3.18.Conv2d_conv3               [1, 1024, 26, 26]   
242_pretrained.layer3.18.BatchNorm2d_bn3            [1, 1024, 26, 26]   
243_pretrained.layer3.18.ReLU_relu                  [1, 1024, 26, 26]   
244_pretrained.layer3.19.Conv2d_conv1               [1, 1024, 26, 26]   
245_pretrained.layer3.19.BatchNorm2d_bn1            [1, 1024, 26, 26]   
246_pretrained.layer3.19.ReLU_relu                  [1, 1024, 26, 26]   
247_pretrained.layer3.19.Conv2d_conv2               [1, 1024, 26, 26]   
248_pretrained.layer3.19.BatchNorm2d_bn2            [1, 1024, 26, 26]   
249_pretrained.layer3.19.ReLU_relu                  [1, 1024, 26, 26]   
250_pretrained.layer3.19.Conv2d_conv3               [1, 1024, 26, 26]   
251_pretrained.layer3.19.BatchNorm2d_bn3            [1, 1024, 26, 26]   
252_pretrained.layer3.19.ReLU_relu                  [1, 1024, 26, 26]   
253_pretrained.layer3.20.Conv2d_conv1               [1, 1024, 26, 26]   
254_pretrained.layer3.20.BatchNorm2d_bn1            [1, 1024, 26, 26]   
255_pretrained.layer3.20.ReLU_relu                  [1, 1024, 26, 26]   
256_pretrained.layer3.20.Conv2d_conv2               [1, 1024, 26, 26]   
257_pretrained.layer3.20.BatchNorm2d_bn2            [1, 1024, 26, 26]   
258_pretrained.layer3.20.ReLU_relu                  [1, 1024, 26, 26]   
259_pretrained.layer3.20.Conv2d_conv3               [1, 1024, 26, 26]   
260_pretrained.layer3.20.BatchNorm2d_bn3            [1, 1024, 26, 26]   
261_pretrained.layer3.20.ReLU_relu                  [1, 1024, 26, 26]   
262_pretrained.layer3.21.Conv2d_conv1               [1, 1024, 26, 26]   
263_pretrained.layer3.21.BatchNorm2d_bn1            [1, 1024, 26, 26]   
264_pretrained.layer3.21.ReLU_relu                  [1, 1024, 26, 26]   
265_pretrained.layer3.21.Conv2d_conv2               [1, 1024, 26, 26]   
266_pretrained.layer3.21.BatchNorm2d_bn2            [1, 1024, 26, 26]   
267_pretrained.layer3.21.ReLU_relu                  [1, 1024, 26, 26]   
268_pretrained.layer3.21.Conv2d_conv3               [1, 1024, 26, 26]   
269_pretrained.layer3.21.BatchNorm2d_bn3            [1, 1024, 26, 26]   
270_pretrained.layer3.21.ReLU_relu                  [1, 1024, 26, 26]   
271_pretrained.layer3.22.Conv2d_conv1               [1, 1024, 26, 26]   
272_pretrained.layer3.22.BatchNorm2d_bn1            [1, 1024, 26, 26]   
273_pretrained.layer3.22.ReLU_relu                  [1, 1024, 26, 26]   
274_pretrained.layer3.22.Conv2d_conv2               [1, 1024, 26, 26]   
275_pretrained.layer3.22.BatchNorm2d_bn2            [1, 1024, 26, 26]   
276_pretrained.layer3.22.ReLU_relu                  [1, 1024, 26, 26]   
277_pretrained.layer3.22.Conv2d_conv3               [1, 1024, 26, 26]   
278_pretrained.layer3.22.BatchNorm2d_bn3            [1, 1024, 26, 26]   
279_pretrained.layer3.22.ReLU_relu                  [1, 1024, 26, 26]   
280_pretrained.layer4.0.Conv2d_conv1                [1, 2048, 26, 26]   
281_pretrained.layer4.0.BatchNorm2d_bn1             [1, 2048, 26, 26]   
282_pretrained.layer4.0.ReLU_relu                   [1, 2048, 26, 26]   
283_pretrained.layer4.0.Conv2d_conv2                [1, 2048, 13, 13]   
284_pretrained.layer4.0.BatchNorm2d_bn2             [1, 2048, 13, 13]   
285_pretrained.layer4.0.ReLU_relu                   [1, 2048, 13, 13]   
286_pretrained.layer4.0.Conv2d_conv3                [1, 2048, 13, 13]   
287_pretrained.layer4.0.BatchNorm2d_bn3             [1, 2048, 13, 13]   
288_pretrained.layer4.0.downsample.Conv2d_0         [1, 2048, 13, 13]   
289_pretrained.layer4.0.downsample.BatchNorm2d_1    [1, 2048, 13, 13]   
290_pretrained.layer4.0.ReLU_relu                   [1, 2048, 13, 13]   
291_pretrained.layer4.1.Conv2d_conv1                [1, 2048, 13, 13]   
292_pretrained.layer4.1.BatchNorm2d_bn1             [1, 2048, 13, 13]   
293_pretrained.layer4.1.ReLU_relu                   [1, 2048, 13, 13]   
294_pretrained.layer4.1.Conv2d_conv2                [1, 2048, 13, 13]   
295_pretrained.layer4.1.BatchNorm2d_bn2             [1, 2048, 13, 13]   
296_pretrained.layer4.1.ReLU_relu                   [1, 2048, 13, 13]   
297_pretrained.layer4.1.Conv2d_conv3                [1, 2048, 13, 13]   
298_pretrained.layer4.1.BatchNorm2d_bn3             [1, 2048, 13, 13]   
299_pretrained.layer4.1.ReLU_relu                   [1, 2048, 13, 13]   
300_pretrained.layer4.2.Conv2d_conv1                [1, 2048, 13, 13]   
301_pretrained.layer4.2.BatchNorm2d_bn1             [1, 2048, 13, 13]   
302_pretrained.layer4.2.ReLU_relu                   [1, 2048, 13, 13]   
303_pretrained.layer4.2.Conv2d_conv2                [1, 2048, 13, 13]   
304_pretrained.layer4.2.BatchNorm2d_bn2             [1, 2048, 13, 13]   
305_pretrained.layer4.2.ReLU_relu                   [1, 2048, 13, 13]   
306_pretrained.layer4.2.Conv2d_conv3                [1, 2048, 13, 13]   
307_pretrained.layer4.2.BatchNorm2d_bn3             [1, 2048, 13, 13]   
308_pretrained.layer4.2.ReLU_relu                   [1, 2048, 13, 13]   
309_scratch.Conv2d_layer1_rn                       [1, 256, 104, 104]   
310_scratch.Conv2d_layer2_rn                         [1, 256, 52, 52]   
311_scratch.Conv2d_layer3_rn                         [1, 256, 26, 26]   
312_scratch.Conv2d_layer4_rn                         [1, 256, 13, 13]   
313_scratch.refinenet4.resConfUnit2.ReLU_relu        [1, 256, 13, 13]   
314_scratch.refinenet4.resConfUnit2.Conv2d_conv1     [1, 256, 13, 13]   
315_scratch.refinenet4.resConfUnit2.ReLU_relu        [1, 256, 13, 13]   
316_scratch.refinenet4.resConfUnit2.Conv2d_conv2     [1, 256, 13, 13]   
317_scratch.refinenet3.resConfUnit1.ReLU_relu        [1, 256, 26, 26]   
318_scratch.refinenet3.resConfUnit1.Conv2d_conv1     [1, 256, 26, 26]   
319_scratch.refinenet3.resConfUnit1.ReLU_relu        [1, 256, 26, 26]   
320_scratch.refinenet3.resConfUnit1.Conv2d_conv2     [1, 256, 26, 26]   
321_scratch.refinenet3.resConfUnit2.ReLU_relu        [1, 256, 26, 26]   
322_scratch.refinenet3.resConfUnit2.Conv2d_conv1     [1, 256, 26, 26]   
323_scratch.refinenet3.resConfUnit2.ReLU_relu        [1, 256, 26, 26]   
324_scratch.refinenet3.resConfUnit2.Conv2d_conv2     [1, 256, 26, 26]   
325_scratch.refinenet2.resConfUnit1.ReLU_relu        [1, 256, 52, 52]   
326_scratch.refinenet2.resConfUnit1.Conv2d_conv1     [1, 256, 52, 52]   
327_scratch.refinenet2.resConfUnit1.ReLU_relu        [1, 256, 52, 52]   
328_scratch.refinenet2.resConfUnit1.Conv2d_conv2     [1, 256, 52, 52]   
329_scratch.refinenet2.resConfUnit2.ReLU_relu        [1, 256, 52, 52]   
330_scratch.refinenet2.resConfUnit2.Conv2d_conv1     [1, 256, 52, 52]   
331_scratch.refinenet2.resConfUnit2.ReLU_relu        [1, 256, 52, 52]   
332_scratch.refinenet2.resConfUnit2.Conv2d_conv2     [1, 256, 52, 52]   
333_scratch.refinenet1.resConfUnit1.ReLU_relu      [1, 256, 104, 104]   
334_scratch.refinenet1.resConfUnit1.Conv2d_conv1   [1, 256, 104, 104]   
335_scratch.refinenet1.resConfUnit1.ReLU_relu      [1, 256, 104, 104]   
336_scratch.refinenet1.resConfUnit1.Conv2d_conv2   [1, 256, 104, 104]   
337_scratch.refinenet1.resConfUnit2.ReLU_relu      [1, 256, 104, 104]   
338_scratch.refinenet1.resConfUnit2.Conv2d_conv1   [1, 256, 104, 104]   
339_scratch.refinenet1.resConfUnit2.ReLU_relu      [1, 256, 104, 104]   
340_scratch.refinenet1.resConfUnit2.Conv2d_conv2   [1, 256, 104, 104]   
341_scratch.output_conv.Conv2d_0                   [1, 128, 208, 208]   
342_scratch.output_conv.Interpolate_1              [1, 128, 416, 416]   
343_scratch.output_conv.Conv2d_2                    [1, 32, 416, 416]   
344_scratch.output_conv.ReLU_3                      [1, 32, 416, 416]   
345_scratch.output_conv.Conv2d_4                     [1, 1, 416, 416]   
346_scratch.output_conv.ReLU_5                       [1, 1, 416, 416]   
347_conv_a.Conv2d_0                                [1, 512, 208, 208]   
348_conv_a.BatchNorm2d_1                           [1, 512, 208, 208]   
349_conv_a.ReLU_2                                  [1, 512, 208, 208]   
350_conv_b.Conv2d_0                                [1, 512, 104, 104]   
351_conv_b.BatchNorm2d_1                           [1, 512, 104, 104]   
352_conv_b.ReLU_2                                  [1, 512, 104, 104]   
353_conv1.Conv2d_0                                 [1, 512, 104, 104]   
354_conv1.BatchNorm2d_1                            [1, 512, 104, 104]   
355_conv1.ReLU_2                                   [1, 512, 104, 104]   
356_conv_c.Conv2d_0                                  [1, 256, 52, 52]   
357_conv_c.BatchNorm2d_1                             [1, 256, 52, 52]   
358_conv_c.ReLU_2                                    [1, 256, 52, 52]   
359_conv2.Conv2d_0                                   [1, 256, 52, 52]   
360_conv2.BatchNorm2d_1                              [1, 256, 52, 52]   
361_conv2.ReLU_2                                     [1, 256, 52, 52]   
362_conv2.Conv2d_3                                   [1, 256, 52, 52]   
363_conv2.BatchNorm2d_4                              [1, 256, 52, 52]   
364_conv2.ReLU_5                                     [1, 256, 52, 52]   
365_conv_d.Conv2d_0                                  [1, 512, 26, 26]   
366_conv_d.BatchNorm2d_1                             [1, 512, 26, 26]   
367_conv_d.ReLU_2                                    [1, 512, 26, 26]   
368_conv3.Conv2d_0                                   [1, 512, 26, 26]   
369_conv3.BatchNorm2d_1                              [1, 512, 26, 26]   
370_conv3.ReLU_2                                     [1, 512, 26, 26]   
371_conv3.Conv2d_3                                   [1, 512, 26, 26]   
372_conv3.BatchNorm2d_4                              [1, 512, 26, 26]   
373_conv3.ReLU_5                                     [1, 512, 26, 26]   
374_conv_e.Conv2d_0                                 [1, 1024, 13, 13]   
375_conv_e.BatchNorm2d_1                            [1, 1024, 13, 13]   
376_conv_e.ReLU_2                                   [1, 1024, 13, 13]   
377_conv4.Conv2d_0                                  [1, 1024, 13, 13]   
378_conv4.BatchNorm2d_1                             [1, 1024, 13, 13]   
379_conv4.ReLU_2                                    [1, 1024, 13, 13]   
380_conv4.Conv2d_3                                  [1, 1024, 13, 13]   
381_conv4.BatchNorm2d_4                             [1, 1024, 13, 13]   
382_conv4.ReLU_5                                    [1, 1024, 13, 13]   
383_conv_3_1.Conv2d_0                                 [1, 27, 13, 13]   
384_yolo_layer_3                                    [1, 3, 13, 13, 9]   
385_conv_2_1_up.Interpolate_0                       [1, 1024, 26, 26]   
386_conv_2_1_up.Conv2d_1                             [1, 256, 26, 26]   
387_conv_2_1_up.BatchNorm2d_2                        [1, 256, 26, 26]   
388_conv_2_1_up.ReLU_3                               [1, 256, 26, 26]   
389_conv_2_2.Conv2d_0                                [1, 256, 26, 26]   
390_conv_2_2.BatchNorm2d_1                           [1, 256, 26, 26]   
391_conv_2_2.ReLU_2                                  [1, 256, 26, 26]   
392_conv_2_2.Conv2d_3                                [1, 512, 26, 26]   
393_conv_2_2.BatchNorm2d_4                           [1, 512, 26, 26]   
394_conv_2_2.ReLU_5                                  [1, 512, 26, 26]   
395_conv_2_2.Conv2d_6                                [1, 256, 26, 26]   
396_conv_2_2.BatchNorm2d_7                           [1, 256, 26, 26]   
397_conv_2_2.ReLU_8                                  [1, 256, 26, 26]   
398_conv_2_2.Conv2d_9                                [1, 512, 26, 26]   
399_conv_2_2.BatchNorm2d_10                          [1, 512, 26, 26]   
400_conv_2_2.ReLU_11                                 [1, 512, 26, 26]   
401_conv_2_2.Conv2d_12                               [1, 256, 26, 26]   
402_conv_2_2.BatchNorm2d_13                          [1, 256, 26, 26]   
403_conv_2_2.ReLU_14                                 [1, 256, 26, 26]   
404_conv_2_2.Conv2d_15                               [1, 512, 26, 26]   
405_conv_2_2.BatchNorm2d_16                          [1, 512, 26, 26]   
406_conv_2_2.ReLU_17                                 [1, 512, 26, 26]   
407_conv_2_3_out.Conv2d_0                             [1, 27, 26, 26]   
408_yolo_layer_2                                    [1, 3, 26, 26, 9]   
409_conv_1_1_up.Interpolate_0                        [1, 512, 52, 52]   
410_conv_1_1_up.Conv2d_1                             [1, 128, 52, 52]   
411_conv_1_1_up.BatchNorm2d_2                        [1, 128, 52, 52]   
412_conv_1_1_up.ReLU_3                               [1, 128, 52, 52]   
413_conv_1_2.Conv2d_0                                [1, 128, 52, 52]   
414_conv_1_2.BatchNorm2d_1                           [1, 128, 52, 52]   
415_conv_1_2.ReLU_2                                  [1, 128, 52, 52]   
416_conv_1_2.Conv2d_3                                [1, 256, 52, 52]   
417_conv_1_2.BatchNorm2d_4                           [1, 256, 52, 52]   
418_conv_1_2.ReLU_5                                  [1, 256, 52, 52]   
419_conv_1_2.Conv2d_6                                [1, 128, 52, 52]   
420_conv_1_2.BatchNorm2d_7                           [1, 128, 52, 52]   
421_conv_1_2.ReLU_8                                  [1, 128, 52, 52]   
422_conv_1_2.Conv2d_9                                [1, 256, 52, 52]   
423_conv_1_2.BatchNorm2d_10                          [1, 256, 52, 52]   
424_conv_1_2.ReLU_11                                 [1, 256, 52, 52]   
425_conv_1_2.Conv2d_12                               [1, 128, 52, 52]   
426_conv_1_2.BatchNorm2d_13                          [1, 128, 52, 52]   
427_conv_1_2.ReLU_14                                 [1, 128, 52, 52]   
428_conv_1_2.Conv2d_15                               [1, 256, 52, 52]   
429_conv_1_2.BatchNorm2d_16                          [1, 256, 52, 52]   
430_conv_1_2.ReLU_17                                 [1, 256, 52, 52]   
431_conv_1_3_out.Conv2d_0                             [1, 27, 52, 52]   
432_yolo_layer_1                                    [1, 3, 52, 52, 9]   

Total params: 135,353,938

> The layer names starting with the prefix 'conv_a', 'conv_b', 'conv_c', 'conv_d', 'conv_e', 'conv1', 'conv2', 'conv3', 'conv4' are custom layers. 

> The layer names starting with the prefix 'conv_' are the original yolo intermediate layers.

```

## Training

* Since I froze the MiDaS part, I only had to train YOLOV3's head and the custom layers before them.

* We can freeze the layers by setting the `requires_grad=False` for the parameters and sending only those parameters to the optimizer which have `requires_grad=True`.

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

* I tested on the same dataset with image size of 512 and got some decent results. I got **mAP = 0.308** as the highest.

![results](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/yolo/results.png)

* BBOX Outputs

![E92](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/output_yolo/E92.jpg)
![E88](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/output_yolo/E88.jpg)
![GImage_94](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/output_yolo/Gimage_94.jpg)

* To test the output of MiDaS, I ran the `run.py` file in the midas folder and it achieved good results.

* Depth Outputs

![1](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/output_midas/0b59d3bd16%20(1).png)
![2](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/output_midas/2.png)
![3](https://github.com/rishabh-bhardwaj-64rr/EVA5/blob/master/S15%20Capstone/images/output_midas/_112919912_gettyimages-1220030093%20(1).png)


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











    
 














