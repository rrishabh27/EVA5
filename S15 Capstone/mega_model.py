"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

# from midas.midas.base_model import BaseModel
# from midas.midas.blocks import FeatureFusionBlock, Interpolate, _make_encoder

from encoder_utils.base_model import BaseModel
from encoder_utils.blocks import FeatureFusionBlock, Interpolate, _make_encoder


from yolov3.models import YOLOLayer
from yolov3.utils import torch_utils
from yolov3.utils.google_utils import *
from yolov3.utils.layers import *
from yolov3.utils.parse_config import *
from pathlib import Path


class Mynet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True, inference=False, classes=4, img_size = (416, 416)):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)
        
        ##################################### MIDAS ########################################
        
        super(Mynet, self).__init__()

        use_pretrained = False if path is None else True
        self.inference = inference

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)
            
            
        ########################################### YOLO ##########################################
        
        # self.aux = auxiliary()
        
                        ######### CUSTOM YOLO LAYERS #########
                        
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) # for a skip conn, output size = 208

        
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) # output size = 104

        
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2), # keep padding = 0 if kernel_size=1 while using stride 2
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # 1x1 with stride 2; output size = 52
        
        
        self.conv_d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) # output size = 26
        
        
        self.conv_e = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        ) # output size = 13
        
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        
        
                        ########### NORMAL YOLO ###########

        self.mask = [0,1,2,3,4,5,6,7,8]
        self.anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
        self.classes=classes
        self.num=9
        self.jitter=.3
        self.ignore_thresh = .7
        self.truth_thresh = 1
        self.random=1
        self.img_size = img_size

        # YOLO head
        conv_output_size = (self.classes + 5) * int((len(self.anchors) / 3))
        
        # self.yolo2_learner_s1 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),# Custom
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     ) # yolo2_learner_s1

        self.conv_1_1_up = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),## Missing in yolo weights
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ) # yolo2_learner_s2

        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.106.Conv2d.weight torch.Size([128, 384, 1, 1])
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.107.Conv2d.weight torch.Size([256, 128, 3, 3])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.103.Conv2d.weight torch.Size([128, 256, 1, 1])
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.109.Conv2d.weight torch.Size([256, 128, 3, 3])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.108.Conv2d.weight torch.Size([128, 256, 1, 1])
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.111.Conv2d.weight torch.Size([256, 128, 3, 3])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ) # yolo2_learner_s3

        self.conv_1_3_out = nn.Sequential(
            nn.Conv2d(256, conv_output_size, kernel_size=1, stride=1, padding=0) ## module_list.112.Conv2d.weight torch.Size([27, 256, 1, 1])
        )

        self.yolo_layer_1 = YOLOLayer(anchors=self.anchors[:3],  # anchor list
                                nc=self.classes,  # number of classes
                                img_size=self.img_size,  # (416, 416)
                                yolo_index=0,  # 0, 1, 2...
                                layers=[],  # output layers
                                stride=32)


        
        
        self.conv_2_1_up = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False), ## missing in yolo weights
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ) # yolo3_learner_s1

        # self.conv_2_2 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False), ## Custom
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     )# yolo3_learner_s2

        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.94.Conv2d.weight torch.Size([256, 768, 1, 1])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.95.Conv2d.weight torch.Size([512, 256, 3, 3])
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.91.Conv2d.weight torch.Size([256, 512, 1, 1])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.97.Conv2d.weight torch.Size([512, 256, 3, 3])
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.96.Conv2d.weight torch.Size([256, 512, 1, 1])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.99.Conv2d.weight torch.Size([512, 256, 3, 3])
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ) # yolo3_learner_s3
        
        
        
        self.conv_2_3_out = nn.Sequential(
            nn.Conv2d(512, conv_output_size, kernel_size=1, stride=1, padding=0) ## module_list.100.Conv2d.weight torch.Size([27, 512, 1, 1])
        )


        self.yolo_layer_2 = YOLOLayer(anchors=self.anchors[3:6],  # anchor list
                                nc=self.classes,  # number of classes
                                img_size=self.img_size,  # (416, 416)
                                yolo_index=1,  # 0, 1, 2...
                                layers=[],  # output layers
                                stride=16)



        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=conv_output_size, kernel_size=1, padding=0, stride=1)
        ) ## module_list.88.Conv2d.weight torch.Size([27, 1024, 1, 1])

        self.yolo_layer_3 = YOLOLayer(anchors=self.anchors[6:],  # anchor list
                                nc=self.classes,  # number of classes
                                img_size=self.img_size,  # (416, 416)
                                yolo_index=2,  # 0, 1, 2...
                                layers=[],  # output layers
                                stride=8)



    def forward(self, x, augment=False):
        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[1][0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None
            

    def forward_once(self, x, augment=False):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth, yolo_outputs
        """
        img_size = x.shape[-2:]  # height, width

        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)
        
        ########################## MIDAS ########################   
        
                        ###### ENCODER LAYERS START (RESNET-101) #######
                ###### ALL THE THREE MODELS WILL USE THIS ENCODER ONLY ######
        layer_1 = self.pretrained.layer1(x) # out channels = 256
        layer_2 = self.pretrained.layer2(layer_1) # out channels = 512
        layer_3 = self.pretrained.layer3(layer_2) # out channels = 1024
        layer_4 = self.pretrained.layer4(layer_3) # out channels = 2048
                            ###### END ######

        layer_1_rn = self.scratch.layer1_rn(layer_1) # out channels = 256
        layer_2_rn = self.scratch.layer2_rn(layer_2) # out channels = 256
        layer_3_rn = self.scratch.layer3_rn(layer_3) # out channels = 256
        layer_4_rn = self.scratch.layer4_rn(layer_4) # out channels = 256

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out_midas = self.scratch.output_conv(path_1)
        out_midas = torch.squeeze(out_midas, dim=1)
        # return torch.squeeze(out_midas, dim=1)
        
        
        ########################## YOLOV3 ########################
        
        out = []

        # CUSTOM NETWORK
        # aux_out = self.aux(x) # output of custom layers added; [out2, out3, out4]
        
        out0 = self.conv_a(x)
        out1 = self.conv_b(out0) + self.conv1(layer_1) # output size 104
        out2 = self.conv_c(out1) + self.conv2(layer_2) # 52
        out3 = self.conv_d(out2) + self.conv3(layer_3) # 26
        out4 = self.conv_e(out3) + self.conv4(layer_4) # 13
        # return (out2, out3, out4)
        

        # yolo 13x13 cells output
        if self.inference:
            yolo_out_3_view, yolo_out_3 = self.yolo_layer_3(self.conv_3_1(out4), out)
        else:
            yolo_out_3 = self.yolo_layer_3(self.conv_3_1(out4), out)

        # yolo 26x26 cells output
        yolo_out_2_up = self.conv_2_1_up(out4) # upsampling from 13x13 to 26x26
        yolo_out_2_1 = torch.cat((yolo_out_2_up, out3), 1)
        yolo_out_2_2 = self.conv_2_2(yolo_out_2_1)
        if self.inference:
            yolo_out_2_view, yolo_out_2 = self.yolo_layer_2(self.conv_2_3_out(yolo_out_2_2), out)
        else:
            yolo_out_2 = self.yolo_layer_2(self.conv_2_3_out(yolo_out_2_2), out)

        # yolo 52x52 cells output
        yolo_out_1_up = self.conv_1_1_up(yolo_out_2_2)
        yolo_out_1_1 = torch.cat((yolo_out_1_up, out2), 1)
        yolo_out_1_2 = self.conv_1_2(yolo_out_1_1)
        if self.inference:
            yolo_out_1_view, yolo_out_1 = self.yolo_layer_1(self.conv_1_3_out(yolo_out_1_2), out)
        else:
            yolo_out_1 = self.yolo_layer_1(self.conv_1_3_out(yolo_out_1_2), out)
        

        if not self.inference:
            yolo_out_1_temp = [torch.unsqueeze(item, dim=0) for item in yolo_out_1]
            yolo_out_1_final = torch.stack(yolo_out_1_temp, dim=1)
            yolo_out_1_final = torch.squeeze(yolo_out_1_final, dim=0)
            
            yolo_out_2_temp = [torch.unsqueeze(item, dim=0) for item in yolo_out_2]
            yolo_out_2_final = torch.stack(yolo_out_2_temp, dim=1)
            yolo_out_2_final = torch.squeeze(yolo_out_2_final, dim=0)

            yolo_out_3_temp = [torch.unsqueeze(item, dim=0) for item in yolo_out_3]
            yolo_out_3_final = torch.stack(yolo_out_3_temp, dim=1)
            yolo_out_3_final = torch.squeeze(yolo_out_3_final, dim=0)
            
            yolo_layers = [yolo_out_3_final, yolo_out_2_final, yolo_out_1_final]
            return out_midas, yolo_layers # depth output, training output

        else:

            img_size = x.shape[-2:] # height, width
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = [yolo_out_3_view, yolo_out_2_view, yolo_out_1_view]
            x = torch.cat(x, 1)  # cat yolo outputs

            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            
            yolo_layers_x = [x,(yolo_out_3, yolo_out_2, yolo_out_1)] 
        
            return out_midas, yolo_layers_x # depth output, (inference output, training output)


