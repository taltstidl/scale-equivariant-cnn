# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.gaussian import get_gaussian_kernel2d


class LocalContrastNorm(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        # Compute Gaussian kernel
        self.kernel = get_gaussian_kernel2d((kernel_size, kernel_size), (1, 1))
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0).repeat(1, in_channels, 1, 1)
        self.kernel /= self.kernel.sum()

    def forward(self, x):
        padding = {'pad': 4 * [self.kernel_size // 2], 'mode': 'replicate'}
        conv1 = F.conv2d(F.pad(x, **padding), self.kernel)
        v = x - conv1  # Subtractive normalization
        conv2 = F.conv2d(F.pad(v * v, **padding), self.kernel)
        sigma = conv2.sqrt()
        c = torch.max(torch.tensor(1e-4), sigma.mean(dim=(2, 3)))
        c = c.unsqueeze(-1).unsqueeze(-1)
        y = v / torch.max(c, sigma)  # Divisive normalization
        return y
