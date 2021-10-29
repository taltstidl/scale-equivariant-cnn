# pylint: disable=no-member
"""
This module contains the models presented in the paper.
"""
from torch import nn

from siconvnet.layers import SiConv2d, ScalePool, GlobalMaxPool

StandardModel = nn.Sequential(
    nn.Conv2d(3, 64, (7, 7)),
    nn.ReLU(),
    nn.Conv2d(64, 128, (7, 7)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (7, 7)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (7, 7)),
    nn.ReLU(),
    GlobalMaxPool(),
    nn.Linear(128, 96),
    nn.Linear(96, 70),
)

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
