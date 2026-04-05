import torch
import numpy as np
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import math
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from .AssNet_encoder_v2 import *#Encoder
#from hswin import DASwin_96_tiny_224
from logger_ import get_root_logger
from mmcv.cnn import build_norm_layer
from .swinv2 import swinv2_small_window8_256
import torch.nn.functional as F




import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernel_size,padding, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size,padding):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,kernel_size,padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AssNet(nn.Module):
    def __init__(self, n_channels, ch, bilinear=False):
        super(AssNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.ch = ch
        self.inc = nn.Sequential(torch.nn.Conv2d(3,self.ch*2,kernel_size=4,stride=4),
                    nn.BatchNorm2d(self.ch*2),
                    nn.ReLU(inplace=True))                

        self.down1 = (Down(64, 128,7,3))
        self.down2 = (Down(128, 256,5,2))
        self.down3 = (Down(256, 512,3,1))


    def forward(self, x):

        x0 = self.inc(x)
        x1,x1_=x0[:,:self.ch,:,:],x0[:,self.ch:,:,:]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return x1_,x1,x2,x3,x4