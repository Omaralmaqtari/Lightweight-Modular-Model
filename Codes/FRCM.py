# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import torch
from torch import nn
import torch.nn.functional as F

class FRCM(nn.Module):
    def __init__(self,ch_ins,ch_out,n_sides=11):
        super(FRCM,self).__init__()

        self.reducers = nn.ModuleList([
            nn.Conv2d(ch_ins[0],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[1],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[2],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[3],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[4],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[5],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[6],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[7],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[8],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[9],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[10],ch_out,kernel_size=1)
            ])

        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        self.gn = nn.GroupNorm(1, ch_out)
        
        self.fused = nn.Conv2d(ch_out*n_sides, ch_out, kernel_size=1)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        
        for m in self.reducers:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fused.weight, std=0.01)
        nn.init.constant_(self.fused.bias, 0)

    def get_weight(self):
        return [self.fused.weight]
    
    def get_bias(self):
        return [self.fused.bias]

    def forward_sides(self, sides, img_shape):
        # pass through base_model and store intermediate activations (sides)
        late_sides = []
        for x, conv in zip(sides, self.reducers):
            x = F.interpolate(conv(x), size=img_shape, mode='bilinear', align_corners=True)
            x = self.gn(self.prelu(x))
            late_sides.append(x)

        return late_sides

    def forward(self, img_shape, sides):
        late_sides = self.forward_sides(sides, img_shape)
        
        late_sides1 = torch.cat([late_sides[0],late_sides[1],late_sides[2],late_sides[3],late_sides[4],
                                 late_sides[5],late_sides[6],late_sides[7],late_sides[8],late_sides[9],
                                 late_sides[10]],1)

        fused = self.prelu(self.fused(late_sides1))
        late_sides.append(fused)

        return late_sides
