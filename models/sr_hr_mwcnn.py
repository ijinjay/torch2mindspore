# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F


logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class Padding1(torch.nn.Module):
    def __init__(self, input_channel):
        super(Padding1, self).__init__()
        self.requires_grad = False
        self.conv = torch.nn.ConvTranspose2d(input_channel, input_channel, 1,stride=2, padding=0, groups=input_channel, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.nn.functional.pad(x1, (0, 1, 0, 1))
        return y

class Padding2(torch.nn.Module):
    def __init__(self, input_channel):
        super(Padding2, self).__init__()
        self.requires_grad = False
        self.conv = torch.nn.ConvTranspose2d(input_channel, input_channel, 1,stride=2, padding=0, groups=input_channel, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.nn.functional.pad(x1, (1, 0, 0, 1))
        return y

class Padding3(torch.nn.Module):
    def __init__(self, input_channel):
        super(Padding3, self).__init__()
        self.requires_grad = False
        self.conv = torch.nn.ConvTranspose2d(input_channel, input_channel, 1,stride=2, padding=0, groups=input_channel, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.nn.functional.pad(x1, (0, 1, 1, 0))
        return y

class Padding4(torch.nn.Module):
    def __init__(self, input_channel):
        super(Padding4, self).__init__()
        self.requires_grad = False
        self.conv = torch.nn.ConvTranspose2d(input_channel, input_channel, 1,stride=2, padding=0, groups=input_channel, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        return y

class IWT(torch.nn.Module):
    def __init__(self, input_channel=1):
        super(IWT, self).__init__()
        self.requires_grad = False
        self.padding1 = Padding1(int(input_channel / 4))
        self.padding2 = Padding2(int(input_channel / 4))
        self.padding3 = Padding3(int(input_channel / 4))
        self.padding4 = Padding4(int(input_channel / 4))

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()

        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        y1 = x1 - x2 - x3 + x4
        y2 = x1 - x2 + x3 - x4
        y3 = x1 + x2 - x3 - x4
        y4 = x1 + x2 + x3 + x4

        t_h1 = self.padding1(y1)
        t_h2 = self.padding2(y3)
        t_h3 = self.padding3(y2)
        t_h4 = self.padding4(y4)

        r= t_h1 + t_h2 + t_h3 + t_h4
        return  r


class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class HighResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=False)

        # downsample2
        self.DWT = DWT()
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv21 = nn.Conv2d(32 * 4, 32 * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22 = nn.Conv2d(32 * 4, 32 * 4, kernel_size=3, stride=1, padding=1, bias=False)

        # merge
        self.IWT1 = IWT(input_channel=32 * 4)

        # downsample2
        self.conv15 = nn.Conv2d(32 * 2, 32*2, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv16 = nn.Conv2d(32 * 2, 32*2, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv23 = nn.Conv2d(32 * 8, 32 * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv24 = nn.Conv2d(32 * 8, 32 * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv31 = nn.Conv2d(32 * 8 * 4, 32 * 8 * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv32 = nn.Conv2d(32 * 8 * 4, 32 * 8 * 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.IWT2 = IWT(input_channel=32*8)
        self.IWT3 = IWT(input_channel=32*8*4)
        self.IWT4 = IWT(input_channel=32*8)

        # output layer
        self.conv_last2 = nn.Conv2d(32 * 2 * 3, 32 * 2 * 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_last1 = nn.Conv2d(32 * 2 * 3, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_last0 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, _x):
        x = self.conv1(_x)
        x = self.relu(x)
        x = self.conv12(x)

        # downsample 1
        x1 = self.conv13(x)
        x1 = self.relu(x1)
        x1 = self.conv14(x1)

        x2 = self.DWT(x)
        x2 = self.conv21(x2)
        x2 = self.relu(x2)
        x2 = self.conv22(x2)

        # merge
        x12 = self.DWT(x1)
        x21 = self.IWT1(x2)
        x1 = torch.cat([x1, x21], 1) # features * 2
        x2 = torch.cat([x2, x12], 1)

        # downsample 2

        x1 = self.conv15(x1)
        x1 = self.relu(x1)
        x1 = self.conv16(x1)

        x2 = self.conv23(x2)
        x2 = self.relu(x2)
        x2 = self.conv24(x2)

        x3 = self.DWT(x2)
        x3 = self.conv31(x3)
        x3 = self.relu(x3)
        x3 = self.conv32(x3)

        x21 = self.IWT2(x2)
        x32 = self.IWT3(x3)
        x31 = self.IWT4(x32)

        x1 = torch.cat([x1, x21, x31], 1)

        # output
        x1 = self.conv_last2(x1)
        x1 = self.relu(x1)
        x1 = self.conv_last1(x1)
        x1 = self.relu(x1)
        x1 = self.conv_last0(x1)
        x1 = x1 + _x

        return x1


def get_sr_hrnet_model(**kwargs):
    model = HighResolutionNet(**kwargs)

    return model

