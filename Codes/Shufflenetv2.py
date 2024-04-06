# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:11:47 2023

@author: omar
"""

## Shufflenetv2

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


def channel_shuffle(x, groups):
    n, c, h, w = x.size()

    channels_per_group = c // groups
    x = x.view(n, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(n, -1, h, w)

    return x
    

class ShuffleNetV2Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ShuffleNetV2Unit, self).__init__()
        assert stride in [1, 2, 3]
        self.stride = stride
        self.dilation = dilation

        inter_channels = out_channels // 2

        if (stride > 1) or (dilation > 1):
            self.branch1 = nn.Sequential(
                DWConv(in_channels, in_channels, 3, stride, dilation, dilation),
                norm_layer(in_channels),
                ConvBNReLU(in_channels, inter_channels, 1, norm_layer=norm_layer))
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels if (stride > 1) else inter_channels, inter_channels, 1, norm_layer=norm_layer),
            DWConv(inter_channels, inter_channels, 3, stride, dilation, dilation),
            norm_layer(inter_channels),
            ConvBNReLU(inter_channels, inter_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        if (self.stride == 1) and (self.dilation == 1):
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, dilated=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ShuffleNetV2, self).__init__()

        self.conv1 = ConvBNReLU(3, 24, 3, 2, 1, norm_layer=norm_layer)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.in_channels = 24

        self.stage2 = self.make_stage(stages_out_channels[0], stages_repeats[0], norm_layer=norm_layer)

        if dilated:
            self.stage3 = self.make_stage(stages_out_channels[1], stages_repeats[1], 2, norm_layer)
            self.stage4 = self.make_stage(stages_out_channels[2], stages_repeats[2], 2, norm_layer)
        else:
            self.stage3 = self.make_stage(stages_out_channels[1], stages_repeats[1], norm_layer=norm_layer)
            self.stage4 = self.make_stage(stages_out_channels[2], stages_repeats[2], norm_layer=norm_layer)


    def make_stage(self, out_channels, repeats, dilation=1, norm_layer=nn.BatchNorm2d):
        stride = 2 if (dilation == 1) else 1
        layers = [ShuffleNetV2Unit(self.in_channels, out_channels, stride, dilation, norm_layer)]
        self.in_channels = out_channels
        for i in range(repeats - 1):
            layers.append(ShuffleNetV2Unit(self.in_channels, out_channels, 1, 1, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        c1 = self.maxpool(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        return c1, c2, c3, c4 
    
def get_shufflenet_v2(stages_repeats, stages_out_channels, **kwargs):
    return ShuffleNetV2(stages_repeats, stages_out_channels, **kwargs)


def shufflenet_v2_0_5(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [48, 96, 192, 1024], **kwargs)

def shufflenet_v2_1_0(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [116, 232, 464, 1024], **kwargs)

def shufflenet_v2_1_5(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [176, 352, 704, 1024], **kwargs)

def shufflenet_v2_2_0(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [244, 488, 976, 2048], **kwargs)

    
class ShuffleNetV2Seg(nn.Module):
    def __init__(self, shufflenetv2, c1, c2, nclass, **kwargs):
        super(ShuffleNetV2Seg, self).__init__()
        self.base_model = shufflenetv2(**kwargs)
        
        self.head = FCNHead(c2, nclass, **kwargs)
        self.auxlayer = FCNHead(c1, nclass, **kwargs)
        self.out = nn.Conv2d(2, nclass, 1, padding=0, bias=False)

    def forward(self, x):
        size = x.size()[2:]

        _, _, c3, c4 = self.base_model(x)
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        auxout = self.auxlayer(c3)
        auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
        out = torch.cat([x,auxout],1)
        
        return self.out(out) 


shufflenets = [shufflenet_v2_0_5, shufflenet_v2_1_0, shufflenet_v2_1_5, shufflenet_v2_2_0]

#if __name__ == '__main__':
    #model = ShuffleNetV2Seg(shufflenets[1], 232, 464, 1)