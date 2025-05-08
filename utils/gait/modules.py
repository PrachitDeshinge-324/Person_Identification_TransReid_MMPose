# OpenGait core modules for inference
# Copied from OpenGait/opengait/modeling/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import math


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SetBlock(nn.Module):
    def __init__(self, block):
        super(SetBlock, self).__init__()
        self.block = block

    def forward(self, x):
        x_size = x.size()
        x = x.view((-1,) + x_size[2:])
        x = self.block(x)
        x = x.view(x_size[:2] + x.size()[1:])
        return x


class SetBlockWrapper(nn.Module):
    def __init__(self, block):
        super(SetBlockWrapper, self).__init__()
        self.block = block

    def forward(self, x):
        x_size = x.size()
        x = x.view((-1,) + x_size[2:])
        x = self.block(x)
        x = x.view(x_size[:2] + x.size()[1:])
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, pool=None):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.pool = pool

    def forward(self, x):
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GaitSet(nn.Module):
    def __init__(self, backbone, pool=None):
        super(GaitSet, self).__init__()
        self.backbone = backbone
        self.pool = pool

    def forward(self, x):
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GaitPart(nn.Module):
    def __init__(self, backbone, pool=None):
        super(GaitPart, self).__init__()
        self.backbone = backbone
        self.pool = pool

    def forward(self, x):
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GaitGL(nn.Module):
    def __init__(self, backbone, pool=None):
        super(GaitGL, self).__init__()
        self.backbone = backbone
        self.pool = pool

    def forward(self, x):
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        return x