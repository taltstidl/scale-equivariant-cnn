# pylint: disable=no-member
"""
This module contains the models presented in the paper.
"""
from torch import nn

from siconvnet.layers import SiConv2d, ScalePool, GlobalMaxPool


class StandardModel(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        channels, layers = 3, []
        for i in range(2):
            layers.append(nn.Conv2d(channels, 32, (7, 7)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(32))
            channels = 32
        for i in range(4):
            layers.append(nn.Conv2d(channels, 64, (7, 7)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(64))
            channels = 64
        layers.append(nn.Conv2d(channels, 128, (7, 7)))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(128))
        layers.append(GlobalMaxPool())
        layers.append(nn.Linear(128, 256))
        layers.append(nn.Linear(256, 70))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ScaleEquivModel(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        channels, scales, layers = 3, 29, []
        for i in range(2):
            layers.append(SiConv2d(channels, 32, scales, 7, interp_mode='bicubic'))
            layers.append(ScalePool(mode='slice'))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(32))
            channels = 32
            scales -= 3
        for i in range(4):
            layers.append(SiConv2d(channels, 64, scales, 7, interp_mode='bicubic'))
            layers.append(ScalePool(mode='slice'))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(64))
            channels = 64
            scales -= 3
        layers.append(SiConv2d(channels, 128, scales, 7, interp_mode='bicubic'))
        layers.append(ScalePool(mode='slice'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(128))
        layers.append(GlobalMaxPool())
        layers.append(nn.Linear(128, 256))
        layers.append(nn.Linear(256, 70))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# TODO: Spatial Transformer and Ensemble
