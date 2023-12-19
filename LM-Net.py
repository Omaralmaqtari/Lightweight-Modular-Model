# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

## LM-Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from FRCM import FRCM

def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n,sigma)
    gaussiankernel = torch.from_numpy(g)
    
    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    
    return laplacekernel.float()


class SEM(nn.Module):
    def __init__(self, ch_out, reduction=16):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)

    
class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, groups, reduction):
        super(EEM, self).__init__()
        
        self.groups = groups
        self.gk = gaussiankernel(ch_in, int(ch_in/groups), kernel, kernel-2, 0.9).to('cuda')
        self.lk = laplaceiankernel(ch_in, int(ch_in/groups), kernel, 0.9).to('cuda')
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
            )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(int(ch_out/2), ch_out, kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out)
            )
            
        self.sem1 = SEM(ch_out, reduction=reduction)
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)
      
    def forward(self, x):
        DoG = F.conv2d(x, self.gk, padding='same',groups=self.groups)
        LoG = F.conv2d(DoG, self.lk, padding='same',groups=self.groups)
        DoG = self.conv1(DoG-x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG*LoG)
        
        tot1 = self.sem1(tot)
        x1 = self.sem2(x)
        
        return self.prelu(x+x1+tot+tot1)
    

class PFM(nn.Module):
    def __init__(self, ch_in, ch_out, ch_out_3x3e, pool_ch_out, EEM_ch_out, reduction, shortcut=False):
        super(PFM, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.03),
            nn.GroupNorm(4, ch_out)
            )
        ch_in1 = ch_out
        
        # 3x3 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out, kernel_size=3,padding=1,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, ch_out)
            )
        # 3x3 conv extended branch
        self.b2 = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out_3x3e, kernel_size=3,padding=1,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out_3x3e, init=0.),
            nn.GroupNorm(4, ch_out_3x3e)
            )
        # 3x3 pool branch
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_in1, pool_ch_out, kernel_size=1,padding=0,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, pool_ch_out)
            )
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out+ch_out_3x3e, kernel_size=1,padding=0,groups=4,bias=False),
                nn.GroupNorm(4, ch_out+ch_out_3x3e)
                )
        
        self.EEM = EEM(ch_in1,EEM_ch_out,kernel=3,groups=ch_in1,reduction=reduction[0]).to("cuda")
        self.sem1 = SEM(ch_out+ch_out_3x3e,reduction=reduction[1])
        self.sem2 = SEM(ch_out+ch_out_3x3e,reduction=reduction[1])
        self.prelu = nn.PReLU(num_parameters=ch_out+ch_out_3x3e, init=0.03)

    def forward(self, x, shortcut=False): 
        x1 = self.reducer(x)
        
        b1 = self.b1(x1) 
        b2 = self.b2(x1+b1)
        b3 = self.b3(x1)
        eem = self.EEM(x1)
        
        y1 = torch.cat([x1+b1+b3+eem,b2], 1)
        y2 = self.sem1(y1)
        
        if shortcut:
            x = self.shortcut(x)
        y3 = self.sem2(x)
        
        return self.prelu(x+y1+y2+y3)


class PDAM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction, dropout):
        super(PDAM, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(ch_in[0],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv1b = nn.Sequential(
            nn.Conv2d(ch_in[0],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv2a = nn.Sequential(
            nn.Conv2d(ch_in[1],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv2b = nn.Sequential(
            nn.Conv2d(ch_in[1],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=2,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv3a = nn.Sequential(
            nn.Conv2d(ch_in[2],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv3b = nn.Sequential(
            nn.Conv2d(ch_in[2],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=4,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_out,ch_out,kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out)
            )
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout2d(dropout)
        self.sem = SEM(ch_in[0], reduction=reduction)
        
    def forward(self,x,x1,x2):
        x0 = self.sem(x)
        x0 = torch.cat([self.conv1a(x+x0),self.conv1b(x+x0)],1)
        x1 = torch.cat([self.conv2a(x1),self.conv2b(x1)],1)
        x2 = torch.cat([self.conv3a(x2),self.conv3b(x2)],1)
        
        x3 = self.dropout(self.softmax(x1*x2))
        
        return self.conv4(x0+x1+x2+x3)
    

class LEE_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(LEE_Net, self).__init__()
        self.prelayer = nn.Conv2d(img_ch,96,kernel_size=3,padding=1,bias=False)
        self.sem1 = SEM(96, reduction=24)
        
        self.PFM1 = PFM(96, 32, 64, 32, 32, [8,24])
        self.PFM2 = PFM(96, 32, 64, 32, 32, [8,24])
        
        self.PFM3 = PFM(96, 32, 96, 32, 32, [8,32], True)
        self.PFM4 = PFM(128, 32, 96, 32, 32, [8,32])
        self.PFM5 = PFM(128, 32, 96, 32, 32, [8,32])
        
        self.PFM6 = PFM(128, 64, 128, 64, 64, [16,48], True)
        self.PFM7 = PFM(192, 64, 128, 64, 64, [16,48])
        self.PFM8 = PFM(192, 64, 128, 64, 64, [16,48])
        self.PFM9 = PFM(192, 64, 128, 64, 64, [16,48])
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3,padding=1,dilation=1,groups=4,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 128)
            )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3,padding=2,dilation=2,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 128)
            )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3,padding=4,dilation=4,groups=4,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 64)
            )
        self.sem2 = SEM(320, reduction=80)
        
        self.PDAM4 = PDAM([320,192,192],128,64,0.0125)
        self.PDAM3 = PDAM([128,192,192],128,32,0.0125)
        self.PDAM2 = PDAM([128,128,128],96,32,0.025)
        self.PDAM1 = PDAM([96,96,96],64,12,0.05)
        
        self.FRCM = FRCM(ch_ins=[96,96,128,128,128,192,192,192,192,64,320],ch_out=2)
        
        self.sem3 = SEM(64+24, reduction=11)
        self.out = nn.Sequential(
            nn.Conv2d(64+24, 64, kernel_size=3,padding=1,groups=4,bias=False),
            nn.PReLU(num_parameters=64, init=-0.01),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 64, kernel_size=1,padding=0,bias=False),
            nn.PReLU(num_parameters=64, init=0.),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, output_ch,kernel_size=1,bias=False),
            nn.PReLU(num_parameters=output_ch, init=0.)
            )
        
        self.Max_Pooling = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.1)

    def forward(self,x):
        img_shape = x.shape[2:]
        x1 = self.prelayer(x)
        x2 = self.sem1(x1)
        x = x1+x2
        
        # encoding path
        i1 = self.PFM1(x)
        i2 = self.PFM2(i1)
        x = self.dropout1(self.Max_Pooling(i2))
        
        i3 = self.PFM3(x, True)
        i4 = self.PFM4(i3)
        i5 = self.PFM5(i4)
        x = self.dropout2(self.Max_Pooling(i5))

        i6 = self.PFM6(x, True)
        i7 = self.PFM7(i6)
        i8 = self.PFM8(i7)
        i9 = self.PFM9(i8)
        x = self.dropout3(self.Max_Pooling(i9))
        
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x1 = torch.cat([x1,x2,x3],1)
        x2 = self.sem2(x1)
        
        # decoding path         
        x = F.interpolate(x1+x2, scale_factor=(2), mode='bilinear', align_corners=False)
        x = self.PDAM4(x,i9,i8)
        x = self.PDAM3(x,i7,i6)
        x = self.dropout3(x)
        
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
        x = self.PDAM2(x,i5,i4)
        x = self.dropout2(x)
        
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
        x = self.PDAM1(x,i2,i1)
        x = self.dropout1(x)
        
        sides = self.FRCM(img_shape,[i1,i2,i3,i4,i5,i6,i7,i8,i9,x,x1])
        x = torch.cat([x,sides[0],sides[1],sides[2],sides[3],sides[4],sides[5],
                       sides[6],sides[7],sides[8],sides[9],sides[10],sides[11]],1)
        
        x1 = self.sem3(x)

        return self.out(x+x1)

