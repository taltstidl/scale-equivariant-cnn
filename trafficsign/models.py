# pylint: disable=no-member
"""
This module contains the models presented in the paper.
"""
import torch
import torch.nn.functional as F

from kornia.geometry.transform import build_pyramid
from torch import nn

from siconvnet.layers import SiConv2d, ScalePool, GlobalMaxPool


class StandardModel(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        layers = []
        channels = [3, 32, 64, 64]
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, (7, 7)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(GlobalMaxPool())
        layers.append(nn.Linear(64, 36))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ScaleEquivModel(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        layers = []
        channels, scales = [3, 32, 64, 64], [29, 26, 23]
        for in_c, out_c, s in zip(channels[:-1], channels[1:], scales):
            layers.append(SiConv2d(in_c, out_c, s, 7, interp_mode='bicubic'))
            layers.append(ScalePool(mode='slice'))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(GlobalMaxPool())
        layers.append(nn.Linear(64, 36))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpatialTransformerModel(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        # Build base model
        layers = []
        channels = [3, 32, 64, 64]
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, (7, 7)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(GlobalMaxPool())
        layers.append(nn.Linear(64, 36))
        self.base_model = nn.Sequential(*layers)
        # Build transformer model
        self.transform_model = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )
        # Init with identity transform
        self.transform_model[-1].weight.data.zero_()
        self.transform_model[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def transform(self, x):
        theta = self.transform_model(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x = self.transform(x)
        return self.base_model(x)


class EnsembleModel(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        layers = []
        channels = [3, 32, 64, 64]
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, (7, 7), padding=(3, 3)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(GlobalMaxPool())
        layers.append(nn.Linear(64, 36))
        self.base_model = nn.Sequential(*layers)

    def forward(self, x):
        pyramid = build_pyramid(x, max_level=3)
        energies = [self.base_model(p) for p in pyramid]
        energies = torch.stack(energies)
        return torch.mean(energies, dim=0)
