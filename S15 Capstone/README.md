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


#### Architecture

![model](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/capstone3.png)


#### Summary

```
Layer                                                         Output Shape  \
                                                                   
0_pretrained.layer1.Conv2d_0                        [1, 64, 224, 224]   
1_pretrained.layer1.BatchNorm2d_1                   [1, 64, 224, 224]   
2_pretrained.layer1.ReLU_2                          [1, 64, 224, 224]   
3_pretrained.layer1.MaxPool2d_3                     [1, 64, 112, 112]   
4_pretrained.layer1.4.0.Conv2d_conv1               [1, 256, 112, 112]   
5_pretrained.layer1.4.0.BatchNorm2d_bn1            [1, 256, 112, 112]   
6_pretrained.layer1.4.0.ReLU_relu                  [1, 256, 112, 112]   
7_pretrained.layer1.4.0.Conv2d_conv2               [1, 256, 112, 112]   
8_pretrained.layer1.4.0.BatchNorm2d_bn2            [1, 256, 112, 112]   
9_pretrained.layer1.4.0.ReLU_relu                  [1, 256, 112, 112]   
10_pretrained.layer1.4.0.Conv2d_conv3              [1, 256, 112, 112]   
11_pretrained.layer1.4.0.BatchNorm2d_bn3           [1, 256, 112, 112]   
12_pretrained.layer1.4.0.downsample.Conv2d_0       [1, 256, 112, 112]   
13_pretrained.layer1.4.0.downsample.BatchNorm2d_1  [1, 256, 112, 112]   
14_pretrained.layer1.4.0.ReLU_relu                 [1, 256, 112, 112]   
15_pretrained.layer1.4.1.Conv2d_conv1              [1, 256, 112, 112]   
16_pretrained.layer1.4.1.BatchNorm2d_bn1           [1, 256, 112, 112]   
17_pretrained.layer1.4.1.ReLU_relu                 [1, 256, 112, 112]   
18_pretrained.layer1.4.1.Conv2d_conv2              [1, 256, 112, 112]   
19_pretrained.layer1.4.1.BatchNorm2d_bn2           [1, 256, 112, 112]   
20_pretrained.layer1.4.1.ReLU_relu                 [1, 256, 112, 112]   
21_pretrained.layer1.4.1.Conv2d_conv3              [1, 256, 112, 112]   
22_pretrained.layer1.4.1.BatchNorm2d_bn3           [1, 256, 112, 112]   
23_pretrained.layer1.4.1.ReLU_relu                 [1, 256, 112, 112]   
24_pretrained.layer1.4.2.Conv2d_conv1              [1, 256, 112, 112]   
25_pretrained.layer1.4.2.BatchNorm2d_bn1           [1, 256, 112, 112]   
26_pretrained.layer1.4.2.ReLU_relu                 [1, 256, 112, 112]   
27_pretrained.layer1.4.2.Conv2d_conv2              [1, 256, 112, 112]   
28_pretrained.layer1.4.2.BatchNorm2d_bn2           [1, 256, 112, 112]   
29_pretrained.layer1.4.2.ReLU_relu                 [1, 256, 112, 112]   
30_pretrained.layer1.4.2.Conv2d_conv3              [1, 256, 112, 112]   
31_pretrained.layer1.4.2.BatchNorm2d_bn3           [1, 256, 112, 112]   
32_pretrained.layer1.4.2.ReLU_relu                 [1, 256, 112, 112]   
33_pretrained.layer2.0.Conv2d_conv1                [1, 512, 112, 112]   
34_pretrained.layer2.0.BatchNorm2d_bn1             [1, 512, 112, 112]   
35_pretrained.layer2.0.ReLU_relu                   [1, 512, 112, 112]   
36_pretrained.layer2.0.Conv2d_conv2                  [1, 512, 56, 56]   
37_pretrained.layer2.0.BatchNorm2d_bn2               [1, 512, 56, 56]   
38_pretrained.layer2.0.ReLU_relu                     [1, 512, 56, 56]   
39_pretrained.layer2.0.Conv2d_conv3                  [1, 512, 56, 56]   
40_pretrained.layer2.0.BatchNorm2d_bn3               [1, 512, 56, 56]   
41_pretrained.layer2.0.downsample.Conv2d_0           [1, 512, 56, 56]   
42_pretrained.layer2.0.downsample.BatchNorm2d_1      [1, 512, 56, 56]   
43_pretrained.layer2.0.ReLU_relu                     [1, 512, 56, 56]   
44_pretrained.layer2.1.Conv2d_conv1                  [1, 512, 56, 56]   
45_pretrained.layer2.1.BatchNorm2d_bn1               [1, 512, 56, 56]   
46_pretrained.layer2.1.ReLU_relu                     [1, 512, 56, 56]   
47_pretrained.layer2.1.Conv2d_conv2                  [1, 512, 56, 56]   
48_pretrained.layer2.1.BatchNorm2d_bn2               [1, 512, 56, 56]   
49_pretrained.layer2.1.ReLU_relu                     [1, 512, 56, 56]   
50_pretrained.layer2.1.Conv2d_conv3                  [1, 512, 56, 56]   
51_pretrained.layer2.1.BatchNorm2d_bn3               [1, 512, 56, 56]   
52_pretrained.layer2.1.ReLU_relu                     [1, 512, 56, 56]   
53_pretrained.layer2.2.Conv2d_conv1                  [1, 512, 56, 56]   
54_pretrained.layer2.2.BatchNorm2d_bn1               [1, 512, 56, 56]   
55_pretrained.layer2.2.ReLU_relu                     [1, 512, 56, 56]   
56_pretrained.layer2.2.Conv2d_conv2                  [1, 512, 56, 56]   
57_pretrained.layer2.2.BatchNorm2d_bn2               [1, 512, 56, 56]   
58_pretrained.layer2.2.ReLU_relu                     [1, 512, 56, 56]   
59_pretrained.layer2.2.Conv2d_conv3                  [1, 512, 56, 56]   
60_pretrained.layer2.2.BatchNorm2d_bn3               [1, 512, 56, 56]   
61_pretrained.layer2.2.ReLU_relu                     [1, 512, 56, 56]   
62_pretrained.layer2.3.Conv2d_conv1                  [1, 512, 56, 56]   
63_pretrained.layer2.3.BatchNorm2d_bn1               [1, 512, 56, 56]   
64_pretrained.layer2.3.ReLU_relu                     [1, 512, 56, 56]   
65_pretrained.layer2.3.Conv2d_conv2                  [1, 512, 56, 56]   
66_pretrained.layer2.3.BatchNorm2d_bn2               [1, 512, 56, 56]   
67_pretrained.layer2.3.ReLU_relu                     [1, 512, 56, 56]   
68_pretrained.layer2.3.Conv2d_conv3                  [1, 512, 56, 56]   
69_pretrained.layer2.3.BatchNorm2d_bn3               [1, 512, 56, 56]   
70_pretrained.layer2.3.ReLU_relu                     [1, 512, 56, 56]   
71_pretrained.layer3.0.Conv2d_conv1                 [1, 1024, 56, 56]   
72_pretrained.layer3.0.BatchNorm2d_bn1              [1, 1024, 56, 56]   
73_pretrained.layer3.0.ReLU_relu                    [1, 1024, 56, 56]   
74_pretrained.layer3.0.Conv2d_conv2                 [1, 1024, 28, 28]   
75_pretrained.layer3.0.BatchNorm2d_bn2              [1, 1024, 28, 28]   
76_pretrained.layer3.0.ReLU_relu                    [1, 1024, 28, 28]   
77_pretrained.layer3.0.Conv2d_conv3                 [1, 1024, 28, 28]   
78_pretrained.layer3.0.BatchNorm2d_bn3              [1, 1024, 28, 28]   
79_pretrained.layer3.0.downsample.Conv2d_0          [1, 1024, 28, 28]   
80_pretrained.layer3.0.downsample.BatchNorm2d_1     [1, 1024, 28, 28]   
81_pretrained.layer3.0.ReLU_relu                    [1, 1024, 28, 28]   
82_pretrained.layer3.1.Conv2d_conv1                 [1, 1024, 28, 28]   
83_pretrained.layer3.1.BatchNorm2d_bn1              [1, 1024, 28, 28]   
84_pretrained.layer3.1.ReLU_relu                    [1, 1024, 28, 28]   
85_pretrained.layer3.1.Conv2d_conv2                 [1, 1024, 28, 28]   
86_pretrained.layer3.1.BatchNorm2d_bn2              [1, 1024, 28, 28]   
87_pretrained.layer3.1.ReLU_relu                    [1, 1024, 28, 28]   
88_pretrained.layer3.1.Conv2d_conv3                 [1, 1024, 28, 28]   
89_pretrained.layer3.1.BatchNorm2d_bn3              [1, 1024, 28, 28]   
90_pretrained.layer3.1.ReLU_relu                    [1, 1024, 28, 28]   
91_pretrained.layer3.2.Conv2d_conv1                 [1, 1024, 28, 28]   
92_pretrained.layer3.2.BatchNorm2d_bn1              [1, 1024, 28, 28]   
93_pretrained.layer3.2.ReLU_relu                    [1, 1024, 28, 28]   
94_pretrained.layer3.2.Conv2d_conv2                 [1, 1024, 28, 28]   
95_pretrained.layer3.2.BatchNorm2d_bn2              [1, 1024, 28, 28]   
96_pretrained.layer3.2.ReLU_relu                    [1, 1024, 28, 28]   
97_pretrained.layer3.2.Conv2d_conv3                 [1, 1024, 28, 28]   
98_pretrained.layer3.2.BatchNorm2d_bn3              [1, 1024, 28, 28]   
99_pretrained.layer3.2.ReLU_relu                    [1, 1024, 28, 28]   
100_pretrained.layer3.3.Conv2d_conv1                [1, 1024, 28, 28]   
101_pretrained.layer3.3.BatchNorm2d_bn1             [1, 1024, 28, 28]   
102_pretrained.layer3.3.ReLU_relu                   [1, 1024, 28, 28]   
103_pretrained.layer3.3.Conv2d_conv2                [1, 1024, 28, 28]   
104_pretrained.layer3.3.BatchNorm2d_bn2             [1, 1024, 28, 28]   
105_pretrained.layer3.3.ReLU_relu                   [1, 1024, 28, 28]   
106_pretrained.layer3.3.Conv2d_conv3                [1, 1024, 28, 28]   
107_pretrained.layer3.3.BatchNorm2d_bn3             [1, 1024, 28, 28]   
108_pretrained.layer3.3.ReLU_relu                   [1, 1024, 28, 28]   
109_pretrained.layer3.4.Conv2d_conv1                [1, 1024, 28, 28]   
110_pretrained.layer3.4.BatchNorm2d_bn1             [1, 1024, 28, 28]   
111_pretrained.layer3.4.ReLU_relu                   [1, 1024, 28, 28]   
112_pretrained.layer3.4.Conv2d_conv2                [1, 1024, 28, 28]   
113_pretrained.layer3.4.BatchNorm2d_bn2             [1, 1024, 28, 28]   
114_pretrained.layer3.4.ReLU_relu                   [1, 1024, 28, 28]   
115_pretrained.layer3.4.Conv2d_conv3                [1, 1024, 28, 28]   
116_pretrained.layer3.4.BatchNorm2d_bn3             [1, 1024, 28, 28]   
117_pretrained.layer3.4.ReLU_relu                   [1, 1024, 28, 28]   
118_pretrained.layer3.5.Conv2d_conv1                [1, 1024, 28, 28]   
119_pretrained.layer3.5.BatchNorm2d_bn1             [1, 1024, 28, 28]   
120_pretrained.layer3.5.ReLU_relu                   [1, 1024, 28, 28]   
121_pretrained.layer3.5.Conv2d_conv2                [1, 1024, 28, 28]   
122_pretrained.layer3.5.BatchNorm2d_bn2             [1, 1024, 28, 28]   
123_pretrained.layer3.5.ReLU_relu                   [1, 1024, 28, 28]   
124_pretrained.layer3.5.Conv2d_conv3                [1, 1024, 28, 28]   
125_pretrained.layer3.5.BatchNorm2d_bn3             [1, 1024, 28, 28]   
126_pretrained.layer3.5.ReLU_relu                   [1, 1024, 28, 28]   
127_pretrained.layer3.6.Conv2d_conv1                [1, 1024, 28, 28]   
128_pretrained.layer3.6.BatchNorm2d_bn1             [1, 1024, 28, 28]   
129_pretrained.layer3.6.ReLU_relu                   [1, 1024, 28, 28]   
130_pretrained.layer3.6.Conv2d_conv2                [1, 1024, 28, 28]   
131_pretrained.layer3.6.BatchNorm2d_bn2             [1, 1024, 28, 28]   
132_pretrained.layer3.6.ReLU_relu                   [1, 1024, 28, 28]   
133_pretrained.layer3.6.Conv2d_conv3                [1, 1024, 28, 28]   
134_pretrained.layer3.6.BatchNorm2d_bn3             [1, 1024, 28, 28]   
135_pretrained.layer3.6.ReLU_relu                   [1, 1024, 28, 28]   
136_pretrained.layer3.7.Conv2d_conv1                [1, 1024, 28, 28]   
137_pretrained.layer3.7.BatchNorm2d_bn1             [1, 1024, 28, 28]   
138_pretrained.layer3.7.ReLU_relu                   [1, 1024, 28, 28]   
139_pretrained.layer3.7.Conv2d_conv2                [1, 1024, 28, 28]   
140_pretrained.layer3.7.BatchNorm2d_bn2             [1, 1024, 28, 28]   
141_pretrained.layer3.7.ReLU_relu                   [1, 1024, 28, 28]   
142_pretrained.layer3.7.Conv2d_conv3                [1, 1024, 28, 28]   
143_pretrained.layer3.7.BatchNorm2d_bn3             [1, 1024, 28, 28]   
144_pretrained.layer3.7.ReLU_relu                   [1, 1024, 28, 28]   
145_pretrained.layer3.8.Conv2d_conv1                [1, 1024, 28, 28]   
146_pretrained.layer3.8.BatchNorm2d_bn1             [1, 1024, 28, 28]   
147_pretrained.layer3.8.ReLU_relu                   [1, 1024, 28, 28]   
148_pretrained.layer3.8.Conv2d_conv2                [1, 1024, 28, 28]   
149_pretrained.layer3.8.BatchNorm2d_bn2             [1, 1024, 28, 28]   
150_pretrained.layer3.8.ReLU_relu                   [1, 1024, 28, 28]   
151_pretrained.layer3.8.Conv2d_conv3                [1, 1024, 28, 28]   
152_pretrained.layer3.8.BatchNorm2d_bn3             [1, 1024, 28, 28]   
153_pretrained.layer3.8.ReLU_relu                   [1, 1024, 28, 28]   
154_pretrained.layer3.9.Conv2d_conv1                [1, 1024, 28, 28]   
155_pretrained.layer3.9.BatchNorm2d_bn1             [1, 1024, 28, 28]   
156_pretrained.layer3.9.ReLU_relu                   [1, 1024, 28, 28]   
157_pretrained.layer3.9.Conv2d_conv2                [1, 1024, 28, 28]   
158_pretrained.layer3.9.BatchNorm2d_bn2             [1, 1024, 28, 28]   
159_pretrained.layer3.9.ReLU_relu                   [1, 1024, 28, 28]   
160_pretrained.layer3.9.Conv2d_conv3                [1, 1024, 28, 28]   
161_pretrained.layer3.9.BatchNorm2d_bn3             [1, 1024, 28, 28]   
162_pretrained.layer3.9.ReLU_relu                   [1, 1024, 28, 28]   
163_pretrained.layer3.10.Conv2d_conv1               [1, 1024, 28, 28]   
164_pretrained.layer3.10.BatchNorm2d_bn1            [1, 1024, 28, 28]   
165_pretrained.layer3.10.ReLU_relu                  [1, 1024, 28, 28]   
166_pretrained.layer3.10.Conv2d_conv2               [1, 1024, 28, 28]   
167_pretrained.layer3.10.BatchNorm2d_bn2            [1, 1024, 28, 28]   
168_pretrained.layer3.10.ReLU_relu                  [1, 1024, 28, 28]   
169_pretrained.layer3.10.Conv2d_conv3               [1, 1024, 28, 28]   
170_pretrained.layer3.10.BatchNorm2d_bn3            [1, 1024, 28, 28]   
171_pretrained.layer3.10.ReLU_relu                  [1, 1024, 28, 28]   
172_pretrained.layer3.11.Conv2d_conv1               [1, 1024, 28, 28]   
173_pretrained.layer3.11.BatchNorm2d_bn1            [1, 1024, 28, 28]   
174_pretrained.layer3.11.ReLU_relu                  [1, 1024, 28, 28]   
175_pretrained.layer3.11.Conv2d_conv2               [1, 1024, 28, 28]   
176_pretrained.layer3.11.BatchNorm2d_bn2            [1, 1024, 28, 28]   
177_pretrained.layer3.11.ReLU_relu                  [1, 1024, 28, 28]   
178_pretrained.layer3.11.Conv2d_conv3               [1, 1024, 28, 28]   
179_pretrained.layer3.11.BatchNorm2d_bn3            [1, 1024, 28, 28]   
180_pretrained.layer3.11.ReLU_relu                  [1, 1024, 28, 28]   
181_pretrained.layer3.12.Conv2d_conv1               [1, 1024, 28, 28]   
182_pretrained.layer3.12.BatchNorm2d_bn1            [1, 1024, 28, 28]   
183_pretrained.layer3.12.ReLU_relu                  [1, 1024, 28, 28]   
184_pretrained.layer3.12.Conv2d_conv2               [1, 1024, 28, 28]   
185_pretrained.layer3.12.BatchNorm2d_bn2            [1, 1024, 28, 28]   
186_pretrained.layer3.12.ReLU_relu                  [1, 1024, 28, 28]   
187_pretrained.layer3.12.Conv2d_conv3               [1, 1024, 28, 28]   
188_pretrained.layer3.12.BatchNorm2d_bn3            [1, 1024, 28, 28]   
189_pretrained.layer3.12.ReLU_relu                  [1, 1024, 28, 28]   
190_pretrained.layer3.13.Conv2d_conv1               [1, 1024, 28, 28]   
191_pretrained.layer3.13.BatchNorm2d_bn1            [1, 1024, 28, 28]   
192_pretrained.layer3.13.ReLU_relu                  [1, 1024, 28, 28]   
193_pretrained.layer3.13.Conv2d_conv2               [1, 1024, 28, 28]   
194_pretrained.layer3.13.BatchNorm2d_bn2            [1, 1024, 28, 28]   
195_pretrained.layer3.13.ReLU_relu                  [1, 1024, 28, 28]   
196_pretrained.layer3.13.Conv2d_conv3               [1, 1024, 28, 28]   
197_pretrained.layer3.13.BatchNorm2d_bn3            [1, 1024, 28, 28]   
198_pretrained.layer3.13.ReLU_relu                  [1, 1024, 28, 28]   
199_pretrained.layer3.14.Conv2d_conv1               [1, 1024, 28, 28]   
200_pretrained.layer3.14.BatchNorm2d_bn1            [1, 1024, 28, 28]   
201_pretrained.layer3.14.ReLU_relu                  [1, 1024, 28, 28]   
202_pretrained.layer3.14.Conv2d_conv2               [1, 1024, 28, 28]   
203_pretrained.layer3.14.BatchNorm2d_bn2            [1, 1024, 28, 28]   
204_pretrained.layer3.14.ReLU_relu                  [1, 1024, 28, 28]   
205_pretrained.layer3.14.Conv2d_conv3               [1, 1024, 28, 28]   
206_pretrained.layer3.14.BatchNorm2d_bn3            [1, 1024, 28, 28]   
207_pretrained.layer3.14.ReLU_relu                  [1, 1024, 28, 28]   
208_pretrained.layer3.15.Conv2d_conv1               [1, 1024, 28, 28]   
209_pretrained.layer3.15.BatchNorm2d_bn1            [1, 1024, 28, 28]   
210_pretrained.layer3.15.ReLU_relu                  [1, 1024, 28, 28]   
211_pretrained.layer3.15.Conv2d_conv2               [1, 1024, 28, 28]   
212_pretrained.layer3.15.BatchNorm2d_bn2            [1, 1024, 28, 28]   
213_pretrained.layer3.15.ReLU_relu                  [1, 1024, 28, 28]   
214_pretrained.layer3.15.Conv2d_conv3               [1, 1024, 28, 28]   
215_pretrained.layer3.15.BatchNorm2d_bn3            [1, 1024, 28, 28]   
216_pretrained.layer3.15.ReLU_relu                  [1, 1024, 28, 28]   
217_pretrained.layer3.16.Conv2d_conv1               [1, 1024, 28, 28]   
218_pretrained.layer3.16.BatchNorm2d_bn1            [1, 1024, 28, 28]   
219_pretrained.layer3.16.ReLU_relu                  [1, 1024, 28, 28]   
220_pretrained.layer3.16.Conv2d_conv2               [1, 1024, 28, 28]   
221_pretrained.layer3.16.BatchNorm2d_bn2            [1, 1024, 28, 28]   
222_pretrained.layer3.16.ReLU_relu                  [1, 1024, 28, 28]   
223_pretrained.layer3.16.Conv2d_conv3               [1, 1024, 28, 28]   
224_pretrained.layer3.16.BatchNorm2d_bn3            [1, 1024, 28, 28]   
225_pretrained.layer3.16.ReLU_relu                  [1, 1024, 28, 28]   
226_pretrained.layer3.17.Conv2d_conv1               [1, 1024, 28, 28]   
227_pretrained.layer3.17.BatchNorm2d_bn1            [1, 1024, 28, 28]   
228_pretrained.layer3.17.ReLU_relu                  [1, 1024, 28, 28]   
229_pretrained.layer3.17.Conv2d_conv2               [1, 1024, 28, 28]   
230_pretrained.layer3.17.BatchNorm2d_bn2            [1, 1024, 28, 28]   
231_pretrained.layer3.17.ReLU_relu                  [1, 1024, 28, 28]   
232_pretrained.layer3.17.Conv2d_conv3               [1, 1024, 28, 28]   
233_pretrained.layer3.17.BatchNorm2d_bn3            [1, 1024, 28, 28]   
234_pretrained.layer3.17.ReLU_relu                  [1, 1024, 28, 28]   
235_pretrained.layer3.18.Conv2d_conv1               [1, 1024, 28, 28]   
236_pretrained.layer3.18.BatchNorm2d_bn1            [1, 1024, 28, 28]   
237_pretrained.layer3.18.ReLU_relu                  [1, 1024, 28, 28]   
238_pretrained.layer3.18.Conv2d_conv2               [1, 1024, 28, 28]   
239_pretrained.layer3.18.BatchNorm2d_bn2            [1, 1024, 28, 28]   
240_pretrained.layer3.18.ReLU_relu                  [1, 1024, 28, 28]   
241_pretrained.layer3.18.Conv2d_conv3               [1, 1024, 28, 28]   
242_pretrained.layer3.18.BatchNorm2d_bn3            [1, 1024, 28, 28]   
243_pretrained.layer3.18.ReLU_relu                  [1, 1024, 28, 28]   
244_pretrained.layer3.19.Conv2d_conv1               [1, 1024, 28, 28]   
245_pretrained.layer3.19.BatchNorm2d_bn1            [1, 1024, 28, 28]   
246_pretrained.layer3.19.ReLU_relu                  [1, 1024, 28, 28]   
247_pretrained.layer3.19.Conv2d_conv2               [1, 1024, 28, 28]   
248_pretrained.layer3.19.BatchNorm2d_bn2            [1, 1024, 28, 28]   
249_pretrained.layer3.19.ReLU_relu                  [1, 1024, 28, 28]   
250_pretrained.layer3.19.Conv2d_conv3               [1, 1024, 28, 28]   
251_pretrained.layer3.19.BatchNorm2d_bn3            [1, 1024, 28, 28]   
252_pretrained.layer3.19.ReLU_relu                  [1, 1024, 28, 28]   
253_pretrained.layer3.20.Conv2d_conv1               [1, 1024, 28, 28]   
254_pretrained.layer3.20.BatchNorm2d_bn1            [1, 1024, 28, 28]   
255_pretrained.layer3.20.ReLU_relu                  [1, 1024, 28, 28]   
256_pretrained.layer3.20.Conv2d_conv2               [1, 1024, 28, 28]   
257_pretrained.layer3.20.BatchNorm2d_bn2            [1, 1024, 28, 28]   
258_pretrained.layer3.20.ReLU_relu                  [1, 1024, 28, 28]   
259_pretrained.layer3.20.Conv2d_conv3               [1, 1024, 28, 28]   
260_pretrained.layer3.20.BatchNorm2d_bn3            [1, 1024, 28, 28]   
261_pretrained.layer3.20.ReLU_relu                  [1, 1024, 28, 28]   
262_pretrained.layer3.21.Conv2d_conv1               [1, 1024, 28, 28]   
263_pretrained.layer3.21.BatchNorm2d_bn1            [1, 1024, 28, 28]   
264_pretrained.layer3.21.ReLU_relu                  [1, 1024, 28, 28]   
265_pretrained.layer3.21.Conv2d_conv2               [1, 1024, 28, 28]   
266_pretrained.layer3.21.BatchNorm2d_bn2            [1, 1024, 28, 28]   
267_pretrained.layer3.21.ReLU_relu                  [1, 1024, 28, 28]   
268_pretrained.layer3.21.Conv2d_conv3               [1, 1024, 28, 28]   
269_pretrained.layer3.21.BatchNorm2d_bn3            [1, 1024, 28, 28]   
270_pretrained.layer3.21.ReLU_relu                  [1, 1024, 28, 28]   
271_pretrained.layer3.22.Conv2d_conv1               [1, 1024, 28, 28]   
272_pretrained.layer3.22.BatchNorm2d_bn1            [1, 1024, 28, 28]   
273_pretrained.layer3.22.ReLU_relu                  [1, 1024, 28, 28]   
274_pretrained.layer3.22.Conv2d_conv2               [1, 1024, 28, 28]   
275_pretrained.layer3.22.BatchNorm2d_bn2            [1, 1024, 28, 28]   
276_pretrained.layer3.22.ReLU_relu                  [1, 1024, 28, 28]   
277_pretrained.layer3.22.Conv2d_conv3               [1, 1024, 28, 28]   
278_pretrained.layer3.22.BatchNorm2d_bn3            [1, 1024, 28, 28]   
279_pretrained.layer3.22.ReLU_relu                  [1, 1024, 28, 28]   
280_pretrained.layer4.0.Conv2d_conv1                [1, 2048, 28, 28]   
281_pretrained.layer4.0.BatchNorm2d_bn1             [1, 2048, 28, 28]   
282_pretrained.layer4.0.ReLU_relu                   [1, 2048, 28, 28]   
283_pretrained.layer4.0.Conv2d_conv2                [1, 2048, 14, 14]   
284_pretrained.layer4.0.BatchNorm2d_bn2             [1, 2048, 14, 14]   
285_pretrained.layer4.0.ReLU_relu                   [1, 2048, 14, 14]   
286_pretrained.layer4.0.Conv2d_conv3                [1, 2048, 14, 14]   
287_pretrained.layer4.0.BatchNorm2d_bn3             [1, 2048, 14, 14]   
288_pretrained.layer4.0.downsample.Conv2d_0         [1, 2048, 14, 14]   
289_pretrained.layer4.0.downsample.BatchNorm2d_1    [1, 2048, 14, 14]   
290_pretrained.layer4.0.ReLU_relu                   [1, 2048, 14, 14]   
291_pretrained.layer4.1.Conv2d_conv1                [1, 2048, 14, 14]   
292_pretrained.layer4.1.BatchNorm2d_bn1             [1, 2048, 14, 14]   
293_pretrained.layer4.1.ReLU_relu                   [1, 2048, 14, 14]   
294_pretrained.layer4.1.Conv2d_conv2                [1, 2048, 14, 14]   
295_pretrained.layer4.1.BatchNorm2d_bn2             [1, 2048, 14, 14]   
296_pretrained.layer4.1.ReLU_relu                   [1, 2048, 14, 14]   
297_pretrained.layer4.1.Conv2d_conv3                [1, 2048, 14, 14]   
298_pretrained.layer4.1.BatchNorm2d_bn3             [1, 2048, 14, 14]   
299_pretrained.layer4.1.ReLU_relu                   [1, 2048, 14, 14]   
300_pretrained.layer4.2.Conv2d_conv1                [1, 2048, 14, 14]   
301_pretrained.layer4.2.BatchNorm2d_bn1             [1, 2048, 14, 14]   
302_pretrained.layer4.2.ReLU_relu                   [1, 2048, 14, 14]   
303_pretrained.layer4.2.Conv2d_conv2                [1, 2048, 14, 14]   
304_pretrained.layer4.2.BatchNorm2d_bn2             [1, 2048, 14, 14]   
305_pretrained.layer4.2.ReLU_relu                   [1, 2048, 14, 14]   
306_pretrained.layer4.2.Conv2d_conv3                [1, 2048, 14, 14]   
307_pretrained.layer4.2.BatchNorm2d_bn3             [1, 2048, 14, 14]   
308_pretrained.layer4.2.ReLU_relu                   [1, 2048, 14, 14]   
309_scratch.Conv2d_layer1_rn                       [1, 256, 112, 112]   
310_scratch.Conv2d_layer2_rn                         [1, 256, 56, 56]   
311_scratch.Conv2d_layer3_rn                         [1, 256, 28, 28]   
312_scratch.Conv2d_layer4_rn                         [1, 256, 14, 14]   
313_scratch.refinenet4.resConfUnit2.ReLU_relu        [1, 256, 14, 14]   
314_scratch.refinenet4.resConfUnit2.Conv2d_conv1     [1, 256, 14, 14]   
315_scratch.refinenet4.resConfUnit2.ReLU_relu        [1, 256, 14, 14]   
316_scratch.refinenet4.resConfUnit2.Conv2d_conv2     [1, 256, 14, 14]   
317_scratch.refinenet3.resConfUnit1.ReLU_relu        [1, 256, 28, 28]   
318_scratch.refinenet3.resConfUnit1.Conv2d_conv1     [1, 256, 28, 28]   
319_scratch.refinenet3.resConfUnit1.ReLU_relu        [1, 256, 28, 28]   
320_scratch.refinenet3.resConfUnit1.Conv2d_conv2     [1, 256, 28, 28]   
321_scratch.refinenet3.resConfUnit2.ReLU_relu        [1, 256, 28, 28]   
322_scratch.refinenet3.resConfUnit2.Conv2d_conv1     [1, 256, 28, 28]   
323_scratch.refinenet3.resConfUnit2.ReLU_relu        [1, 256, 28, 28]   
324_scratch.refinenet3.resConfUnit2.Conv2d_conv2     [1, 256, 28, 28]   
325_scratch.refinenet2.resConfUnit1.ReLU_relu        [1, 256, 56, 56]   
326_scratch.refinenet2.resConfUnit1.Conv2d_conv1     [1, 256, 56, 56]   
327_scratch.refinenet2.resConfUnit1.ReLU_relu        [1, 256, 56, 56]   
328_scratch.refinenet2.resConfUnit1.Conv2d_conv2     [1, 256, 56, 56]   
329_scratch.refinenet2.resConfUnit2.ReLU_relu        [1, 256, 56, 56]   
330_scratch.refinenet2.resConfUnit2.Conv2d_conv1     [1, 256, 56, 56]   
331_scratch.refinenet2.resConfUnit2.ReLU_relu        [1, 256, 56, 56]   
332_scratch.refinenet2.resConfUnit2.Conv2d_conv2     [1, 256, 56, 56]   
333_scratch.refinenet1.resConfUnit1.ReLU_relu      [1, 256, 112, 112]   
334_scratch.refinenet1.resConfUnit1.Conv2d_conv1   [1, 256, 112, 112]   
335_scratch.refinenet1.resConfUnit1.ReLU_relu      [1, 256, 112, 112]   
336_scratch.refinenet1.resConfUnit1.Conv2d_conv2   [1, 256, 112, 112]   
337_scratch.refinenet1.resConfUnit2.ReLU_relu      [1, 256, 112, 112]   
338_scratch.refinenet1.resConfUnit2.Conv2d_conv1   [1, 256, 112, 112]   
339_scratch.refinenet1.resConfUnit2.ReLU_relu      [1, 256, 112, 112]   
340_scratch.refinenet1.resConfUnit2.Conv2d_conv2   [1, 256, 112, 112]   
341_scratch.output_conv.Conv2d_0                   [1, 128, 224, 224]   
342_scratch.output_conv.Interpolate_1              [1, 128, 448, 448]   
343_scratch.output_conv.Conv2d_2                    [1, 32, 448, 448]   
344_scratch.output_conv.ReLU_3                      [1, 32, 448, 448]   
345_scratch.output_conv.Conv2d_4                     [1, 1, 448, 448]   
346_scratch.output_conv.ReLU_5                       [1, 1, 448, 448]   
347_conv_a.Conv2d_0                                [1, 512, 224, 224]   
348_conv_a.BatchNorm2d_1                           [1, 512, 224, 224]   
349_conv_a.ReLU_2                                  [1, 512, 224, 224]   
350_conv_b.Conv2d_0                                [1, 512, 112, 112]   
351_conv_b.BatchNorm2d_1                           [1, 512, 112, 112]   
352_conv_b.ReLU_2                                  [1, 512, 112, 112]   
353_conv1.Conv2d_0                                 [1, 512, 112, 112]   
354_conv1.BatchNorm2d_1                            [1, 512, 112, 112]   
355_conv1.ReLU_2                                   [1, 512, 112, 112]   
356_conv_c.Conv2d_0                                  [1, 256, 56, 56]   
357_conv_c.BatchNorm2d_1                             [1, 256, 56, 56]   
358_conv_c.ReLU_2                                    [1, 256, 56, 56]   
359_conv2.Conv2d_0                                   [1, 256, 56, 56]   
360_conv2.BatchNorm2d_1                              [1, 256, 56, 56]   
361_conv2.ReLU_2                                     [1, 256, 56, 56]   
362_conv3.Conv2d_0                                   [1, 512, 28, 28]   
363_conv3.BatchNorm2d_1                              [1, 512, 28, 28]   
364_conv3.ReLU_2                                     [1, 512, 28, 28]   
365_conv4.Conv2d_0                                  [1, 1024, 14, 14]   
366_conv4.BatchNorm2d_1                             [1, 1024, 14, 14]   
367_conv4.ReLU_2                                    [1, 1024, 14, 14]   
368_conv_3_1.Conv2d_0                                 [1, 27, 14, 14]   
369_yolo_layer_3                                    [1, 3, 14, 14, 9]   
370_conv_2_1_up.Interpolate_0                       [1, 1024, 28, 28]   
371_conv_2_1_up.Conv2d_1                             [1, 256, 28, 28]   
372_conv_2_1_up.BatchNorm2d_2                        [1, 256, 28, 28]   
373_conv_2_1_up.ReLU_3                               [1, 256, 28, 28]   
374_conv_2_2.Conv2d_0                                [1, 256, 28, 28]   
375_conv_2_2.BatchNorm2d_1                           [1, 256, 28, 28]   
376_conv_2_2.ReLU_2                                  [1, 256, 28, 28]   
377_conv_2_2.Conv2d_3                                [1, 512, 28, 28]   
378_conv_2_2.BatchNorm2d_4                           [1, 512, 28, 28]   
379_conv_2_2.ReLU_5                                  [1, 512, 28, 28]   
380_conv_2_2.Conv2d_6                                [1, 256, 28, 28]   
381_conv_2_2.BatchNorm2d_7                           [1, 256, 28, 28]   
382_conv_2_2.ReLU_8                                  [1, 256, 28, 28]   
383_conv_2_2.Conv2d_9                                [1, 512, 28, 28]   
384_conv_2_2.BatchNorm2d_10                          [1, 512, 28, 28]   
385_conv_2_2.ReLU_11                                 [1, 512, 28, 28]   
386_conv_2_2.Conv2d_12                               [1, 256, 28, 28]   
387_conv_2_2.BatchNorm2d_13                          [1, 256, 28, 28]   
388_conv_2_2.ReLU_14                                 [1, 256, 28, 28]   
389_conv_2_2.Conv2d_15                               [1, 512, 28, 28]   
390_conv_2_2.BatchNorm2d_16                          [1, 512, 28, 28]   
391_conv_2_2.ReLU_17                                 [1, 512, 28, 28]   
392_conv_2_3_out.Conv2d_0                             [1, 27, 28, 28]   
393_yolo_layer_2                                    [1, 3, 28, 28, 9]   
394_conv_1_1_up.Interpolate_0                        [1, 512, 56, 56]   
395_conv_1_1_up.Conv2d_1                             [1, 128, 56, 56]   
396_conv_1_1_up.BatchNorm2d_2                        [1, 128, 56, 56]   
397_conv_1_1_up.ReLU_3                               [1, 128, 56, 56]   
398_conv_1_2.Conv2d_0                                [1, 128, 56, 56]   
399_conv_1_2.BatchNorm2d_1                           [1, 128, 56, 56]   
400_conv_1_2.ReLU_2                                  [1, 128, 56, 56]   
401_conv_1_2.Conv2d_3                                [1, 256, 56, 56]   
402_conv_1_2.BatchNorm2d_4                           [1, 256, 56, 56]   
403_conv_1_2.ReLU_5                                  [1, 256, 56, 56]   
404_conv_1_2.Conv2d_6                                [1, 128, 56, 56]   
405_conv_1_2.BatchNorm2d_7                           [1, 128, 56, 56]   
406_conv_1_2.ReLU_8                                  [1, 128, 56, 56]   
407_conv_1_2.Conv2d_9                                [1, 256, 56, 56]   
408_conv_1_2.BatchNorm2d_10                          [1, 256, 56, 56]   
409_conv_1_2.ReLU_11                                 [1, 256, 56, 56]   
410_conv_1_2.Conv2d_12                               [1, 128, 56, 56]   
411_conv_1_2.BatchNorm2d_13                          [1, 128, 56, 56]   
412_conv_1_2.ReLU_14                                 [1, 128, 56, 56]   
413_conv_1_2.Conv2d_15                               [1, 256, 56, 56]   
414_conv_1_2.BatchNorm2d_16                          [1, 256, 56, 56]   
415_conv_1_2.ReLU_17                                 [1, 256, 56, 56]   
416_conv_1_3_out.Conv2d_0                             [1, 27, 56, 56]   
417_yolo_layer_1                                    [1, 3, 56, 56, 9]   

Total params: 139,079,506


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

![results](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/yolo/results.png)

* BBOX Outputs

![E92](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/output_yolo/E92.jpg)
![E88](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/output_yolo/E88.jpg)
![GImage_94](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/output_yolo/Gimage_94.jpg)

* To test the output of MiDaS, I ran the `run.py` file in the midas folder and it achieved good results.

* Depth Outputs

![1](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/output_midas/0b59d3bd16%20(1).png)
![2](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/output_midas/2.png)
![3](https://github.com/rishabh-bhardwaj-64rr/experiments_EVA5_Phase1/blob/master/Session%2015%20Capstone/images/output_midas/_112919912_gettyimages-1220030093%20(1).png)


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











    
 














