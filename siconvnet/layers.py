# pylint: disable=no-member
"""
This module contains the layers used by the scale-invariant models. Currently there are three:

* `Interpolate`: Implements the interpolation of the kernels.
* `SiConv2d`: Provides a replacement for [`Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) that
  uses scale-invariant convolutions. Note that only the input interface is compatible, as the output contains an
  additional dimension for the scales.
* `ScalePool`: Provides a method to remove the additional dimension using maximum pooling.

These layers make it easy to create your own models employing scale-invariant convolutions.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class Interpolate(nn.Module):
    """ Kernel interpolation layer.

    To implement scale-invariant convolutions, a reference kernel that stores the shared weights needs to be scaled to
    larger (or possibly smaller) sizes. As the sizes do not match, interpolation needs to be applied.

    Parameters
    ----------
    kernel_size: int
        Input kernel size of the given reference kernel.
    target_size: int
        Targeted kernel size for the scaled kernel.
    mode: Literal['nearest', 'bilinear', 'bicubic', 'area']
        Method used for interpolation. Must be either nearest, bilinear or bicubic. Internally used the functional
        [`interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate) method provided by PyTorch.
    """

    def __init__(self, kernel_size: int, target_size: int, mode: str = 'bilinear'):
        """"""
        super().__init__()
        if mode not in ['nearest', 'bilinear', 'bicubic', 'area']:
            raise ValueError(
                'Interpolation mode must be either `nearest`, `bilinear`, `bicubic` or `slice`, got {}'.format(mode))
        self.kernel_size = kernel_size
        self.target_size = target_size
        self.mode = mode
        if self.mode == 'area':
            interpolation_matrix = self._generate_interpolation_matrix(kernel_size, target_size)
            self.register_buffer('interpolation_matrix', interpolation_matrix, persistent=False)

    def forward(self, x):
        """
        Interpolates the kernels using the given interpolation method. In addition, it normalizes the rescaled kernel
        such that the outputs remain comparable across scales, i.e. by multiplying with \\(k_{in}^2 / k_{out}^2\\)
        where \\(k_{in}\\) is the input kernel size and \\(k_{out}\\) is the targeted kernel size.

        Parameters
        ----------
        x: torch.Tensor
            Kernels that should be interpolated, of shape (out_channels, in_channels, kernel_size, kernel_size)

        Returns
        -------
        torch.Tensor
            Rescaled kernels, of shape (out_channels, in_channels, target_size, target_size)
        """
        if self.mode == 'area':
            out_channels, in_channels, kernel_h, kernel_w = x.shape
            x = x.view(out_channels, in_channels, kernel_h * kernel_w, 1)
            x = torch.matmul(self.interpolation_matrix, x)
            x = x.view(out_channels, in_channels, self.target_size, self.target_size)
        else:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode=self.mode)
        x *= (self.kernel_size * self.kernel_size) / (self.target_size * self.target_size)
        return x

    @staticmethod
    def _generate_interpolation_matrix(kernel_size, target_size):
        """ Generates an interpolation matrix for use with area-based interpolation.

        Parameters
        ----------
        kernel_size: int
            Input kernel size of the given reference kernel.
        target_size: int
            Targeted kernel size for the scaled kernel.

        Returns
        -------
        torch.Tensor
            Interpolation matrix, of shape (target_size**2, kernel_size**2)
        """
        # Generate interpolation for up-scaling to size m
        matrix = torch.zeros(target_size * target_size, kernel_size * kernel_size)
        # i and j are indices into the up-scaled kernel
        for i in range(target_size):
            ix = (i * kernel_size) // target_size
            id = max(0, (i + 1) * kernel_size - (ix + 1) * target_size)
            for j in range(target_size):
                jx = (j * kernel_size) // target_size
                jd = max(0, (j + 1) * kernel_size - (jx + 1) * target_size)
                if id == 0 and jd == 0:
                    matrix[i * target_size + j, ix * kernel_size + jx] = 1.0
                elif id == 0:
                    jfrac = jd / kernel_size
                    matrix[i * target_size + j, ix * kernel_size + jx] = 1.0 - jfrac
                    matrix[i * target_size + j, ix * kernel_size + (jx + 1)] = jfrac
                elif jd == 0:
                    ifrac = id / kernel_size
                    matrix[i * target_size + j, ix * kernel_size + jx] = 1.0 - ifrac
                    matrix[i * target_size + j, (ix + 1) * kernel_size + jx] = ifrac
                else:
                    ifrac, jfrac = id / kernel_size, jd / kernel_size
                    matrix[i * target_size + j, ix * kernel_size + jx] = (1.0 - ifrac) * (1.0 - jfrac)
                    matrix[i * target_size + j, ix * kernel_size + (jx + 1)] = jfrac * (1.0 - ifrac)
                    matrix[i * target_size + j, (ix + 1) * kernel_size + jx] = ifrac * (1.0 - jfrac)
                    matrix[i * target_size + j, (ix + 1) * kernel_size + (jx + 1)] = ifrac * jfrac
        return matrix


class SiConv2d(nn.Module):
    """ Scale-invariant convolution layer.

    Applies a scale-invariant 2D convolution over an input signal composed of several
    input planes. This operation is comparable to a standard convolution, but generates an
    additional output dimension representing scale.

    Parameters
    ----------
    in_channels: int
        Number of channels in the input image.
    out_channels: int
        Number of channels produced by the convolution.
    num_scales: int
        Number of scales the kernels are applied on. Starts with kernel size and then incrementally
        increases by 2 on each side. Usually chosen so largest kernel size covers complete input.
    kernel_size: int
        Size of the convolution kernel. Only kernels with the same height and width are supported.
    padding: bool
        Whether zero-padding should be applied to the input. If enabled, significantly more operations
        are performed which might degrade performance.
    interp_mode: Literal['nearest', 'bilinear', 'bicubic', 'area']
        Method used for kernel interpolation. See `Interpolate` for details.
    """

    def __init__(self, in_channels: int, out_channels: int, num_scales: int, kernel_size: int, stride: int = 1,
                 padding: bool = False, interp_mode: str = 'nearest'):
        """"""
        super().__init__()
        # Save all arguments as instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.interp_mode = interp_mode
        # Generate weights and bias
        weight = torch.empty((out_channels, in_channels, kernel_size, kernel_size))
        self.weights = nn.Parameter(weight)
        bias = torch.empty((out_channels,))
        self.bias = nn.Parameter(bias)
        # Initialize weights and bias
        stdv = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # Initialize interpolation layers
        interpolate = list()
        for i in range(num_scales):
            interpolate.append(Interpolate(kernel_size, kernel_size + 2 * i, mode=interp_mode))
        self.interpolate = nn.ModuleList(interpolate)

    def forward(self, x):
        """ Performs the scale-invariant convolution on the input.
        """
        # Assert shape matches expectations
        _, in_channels, height, width = x.shape
        assert in_channels == self.in_channels, 'Expected {} input channels, got {}' \
            .format(self.in_channels, in_channels)
        num_scales = (min(height, width) - self.kernel_size) // 2 + 1
        assert num_scales == self.num_scales, 'Expected {} scales, got {}' \
            .format(self.num_scales, num_scales)
        # Apply scale-invariant convolution
        # 1. Generate scaled kernel for each scale
        kernels = list()
        for interpolate in self.interpolate:
            kernels.append(interpolate(self.weights))
        # 2. Compute convolution for each scale
        outputs = list()
        for kernel in kernels:
            if self.padding:
                _padding = kernel.shape[-1] // 2
                output = F.conv2d(x, kernel, self.bias, padding=_padding)
            else:
                output = F.conv2d(x, kernel, self.bias)
                _padding = (kernel.shape[-1] // 2) - (kernels[0].shape[-1] // 2)
                output = F.pad(output, 4 * (_padding,))
            outputs.append(output)
        return torch.stack(outputs, 2)


class ScalePool(nn.Module):
    """ Pooling layer to collapse the additional scale dimension.

    One of the problems after applying the scale-invariant convolution is the additional dimension representing the
    scale. This makes it incompatible with many standard architectures and increases the overall complexity. One way
    to handle this is to remove the dimension via pooling. Two pooling strategies are proposed:

    * Pixel-wise Pooling (`pixel`): This is considered the standard pooling operation, whereby each pixel is pooled
    independently of each other over the scale domain. However, this can lead to uncoordinated activations as the
    scales for each pixel don't necessarily match or correlate.
    * Slice-wise Pooling (`slice`): This is an alternative to the standard pooling operation, where the slice
    containing the largest activation is used. Since the scales for each pixel are thus equal, the activations are
    better coordinated.
    * Energy-wise Pooling (`energy`): This is an alternative to the standard pooling operation, where the slice
    containing the largest sum of activations is used. Since the scales for each pixel are thus equal, the activations
    are better coordinated. In addition, it may be more stable than pure slice-wise pooling.

    Parameters
    ----------
    mode: Literal['pixel', 'slice', 'energy']
        Method used for pooling. Must be either pixel, slice or energy. For the former, the maximum for each individual
        pixel is returned, for the latter the slice with the largest activation or sum of activations is returned.
    """

    def __init__(self, mode: str = 'pixel'):
        """"""
        super().__init__()
        if mode not in ['pixel', 'slice', 'energy']:
            raise ValueError('Pooling mode must be either `pixel`, `slice` or `energy`, got {}'.format(mode))
        self.mode = mode
        self.indices = None  # Book-keeping for activation indices

    def forward(self, x):
        """ Performs the pooling on the input.

        Parameters
        ----------
        x: torch.Tensor
            Feature maps that should be pooled, of shape (mini_batch, in_channels, num_scales, height, width)

        Returns
        -------
        torch.Tensor
            Pooled feature maps, of shape (mini_batch, in_channels, height, width)
        """
        num_scales, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
        if self.mode == 'pixel':
            activations, indices = F.max_pool3d(x, kernel_size=(num_scales, 1, 1), return_indices=True)
            indices = indices.squeeze(dim=-3)  # remove scale dim
            self.indices = torch.div(indices, height * width, rounding_mode='trunc')
            return activations.squeeze(dim=-3)
        if self.mode == 'slice':
            _, indices = F.max_pool3d(x, kernel_size=(num_scales, height, width), return_indices=True)
            indices = indices.squeeze(dim=-3).squeeze(dim=-2).squeeze(dim=-1)  # remove scale, height and width dims
            self.indices = torch.div(indices, height * width, rounding_mode='trunc')
            indices = self.indices[:, :, None, None, None].repeat(1, 1, 1, height, width)
            return x.gather(dim=-3, index=indices).squeeze(dim=-3)
        if self.mode == 'energy':
            self.indices = torch.sum(x, dim=(-2, -1)).argmax(dim=-1)
            indices = self.indices[:, :, None, None, None].repeat(1, 1, 1, height, width)
            return x.gather(dim=-3, index=indices).squeeze(dim=-3)


class GlobalMaxPool(nn.Module):
    """ Global maximum pooling layer.

    Unlike a standard maximum pooling layer, the pooling is applied globally such that the kernel size equals the input
    size. This leads to a single maximum value per channel. Subsequent dimensions are automatically squeezed.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Performs global maximum pooling on the input.

        Parameters
        ----------
        x: torch.Tensor
            Feature maps that should be pooled, of shape (mini_batch, in_channels, *dims)

        Returns
        -------
        torch.Tensor
            Pooled maximum value per channel, of shape (mini_batch, in_channels)
        """
        dims = x.shape[2:]  # Shape of remaining dimensions
        num_dims = len(dims)  # Number of remaining dimensions
        x = F.max_pool2d(x, kernel_size=dims) if num_dims == 2 else F.max_pool3d(x, kernel_size=dims)
        for _ in range(num_dims):
            x = x.squeeze(dim=-1)
        return x
