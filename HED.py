# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:01:09 2023

@author: omar
"""

# HED
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils import model_zoo
from collections import OrderedDict

class SideLayer(nn.Module):
    # Side branch layer used in hed model.
    # Args:
    # in_ch(int): Input channel size from main stream.
    # scale(int): Scale factor used in upscaling. This value should be doubled after maxpool2D.
    def __init__(self, in_ch, scale):
        super(SideLayer, self).__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class FuseLayer(nn.Module):
    # Concatinate layer used in hed model.
    def __init__(self):
        super(FuseLayer, self).__init__()
        self.weight_sum = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, a1, a2, a3, a4, a5):
        x = torch.cat((a1, a2, a3, a4, a5), dim=1) # N,5,H,W
        x = self.weight_sum(x)
        x = F.sigmoid(x)
        return x

    
class HED(nn.Module):
    # Main implementation of HED model
    def __init__(self):
        super(HED, self).__init__()
        self.vgg_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side1 = SideLayer(64, 1)
        self.vgg_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.vgg_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side2 = SideLayer(128, 2)
        self.vgg_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.vgg_conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side3 = SideLayer(256, 4)
        self.vgg_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.vgg_conv8 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side4 = SideLayer(512, 8)
        self.vgg_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.vgg_conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side5 = SideLayer(512, 16)
        
        self.fuse = FuseLayer()
    
    def forward(self, x):
        x = F.relu(self.vgg_conv1(x))
        x = F.relu(self.vgg_conv2(x))
        a1 = self.side1(x)
        s1 = F.sigmoid(a1)
        x = self.vgg_pool1(x)
        x = F.relu(self.vgg_conv3(x))
        x = F.relu(self.vgg_conv4(x))
        a2 = self.side2(x)
        s2 = F.sigmoid(a2)
        x = self.vgg_pool2(x)
        x = F.relu(self.vgg_conv5(x))
        x = F.relu(self.vgg_conv6(x))
        x = F.relu(self.vgg_conv7(x))
        a3 = self.side3(x)
        s3 = F.sigmoid(a3)
        x = self.vgg_pool3(x)
        x = F.relu(self.vgg_conv8(x))
        x = F.relu(self.vgg_conv9(x))
        x = F.relu(self.vgg_conv10(x))
        a4 = self.side4(x)
        s4 = F.sigmoid(a4)
        x = self.vgg_pool4(x)
        x = F.relu(self.vgg_conv11(x))
        x = F.relu(self.vgg_conv12(x))
        x = F.relu(self.vgg_conv13(x))
        a5 = self.side5(x)
        s5 = F.sigmoid(a5)
        fuse = self.fuse(a1, a2, a3, a4, a5)
        return fuse, s1, s2, s3, s4, s5
    
    def loss(self, fuse, s1, s2, s3, s4, s5, y):
        l = BalancedCrossEntropy()
        dist = nn.BCELoss()
        L_side = l(s1, y) + l(s2, y) + l(s3, y) + l(s4, y) + l(s5, y)
        L_fuse = dist(fuse, y)
        loss = L_side + L_fuse
        return loss
      
class BalancedCrossEntropy(nn.Module):
    # Binary cross entropy 2D with weights. 
    # Args:
    # input(Tensor): Input 2D image. Each pixel value must be float in [0, 1]
    # target(Tensor): Target 2D label. Each pixel value must be 0 or 1.
    def __init__(self):
        super(BalancedCrossEntropy, self).__init__()
        self.eps = 0.0000000000000001
    
    def forward(self, input, target):
        y_minus = torch.sum(1-target) # edge=0
        y_plus = torch.sum(target) # non-edge=1
        beta = y_minus / (y_minus + y_plus)
        loss = torch.mean(-beta*target*torch.log(self.eps + input) - (1-beta)*(1-target)*torch.log(1-input+self.eps) )
        return loss
    
def get_vgg_weights():
    # Download the pre-trained VGG wegiths and load in HED model.
    # Names of weights are modified from the original.
    vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
    vgg_state_dict = model_zoo.load_url(vgg16_url, model_dir="./")
    idx = 0
    partial_vgg = OrderedDict()
    for k, v in vgg_state_dict.items():
        if k == "classifier.0.weight":
            break
        new_key = "vgg_conv" + str(int(idx/2)+1) + k[k.rfind("."):]
        partial_vgg[new_key] = v
        idx+=1
    return partial_vgg