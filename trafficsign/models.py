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


ScaleEquivariantModel = nn.Sequential(
    SiConv2d(3, 64, 29, 7, interp_mode='bicubic'),
    ScalePool(mode='slice'),
    nn.ReLU(),
    SiConv2d(64, 128, 26, 7, interp_mode='bicubic'),
    ScalePool(mode='slice'),
    nn.ReLU(),
    SiConv2d(128, 128, 23, 7, interp_mode='bicubic'),
    ScalePool(mode='slice'),
    nn.ReLU(),
    SiConv2d(128, 128, 20, 7, interp_mode='bicubic'),
    ScalePool(mode='slice'),
    nn.ReLU(),
    GlobalMaxPool(),
    nn.Linear(128, 96),
    nn.Linear(96, 70),
)

# TODO: Spatial Transformer and Ensemble
