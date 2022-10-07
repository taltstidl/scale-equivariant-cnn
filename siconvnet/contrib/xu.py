# pylint: disable=no-member
import torch
import torch.nn.functional as F
from torch import nn


class XuConv2d(nn.Conv2d):
    """ Scale-Invariant Convolutional Neural Networks.

    Paper: Xu et al., https://arxiv.org/pdf/1411.6369.pdf
    Implementation: https://github.com/ISosnovik/sesn/blob/master/models/impl/scale_modules.py

    Instead of using multiple columns as in the original paper, we utilize the `groups` argument in PyTorch Conv2d as in
    "Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups" (https://arxiv.org/pdf/1605.06489.pdf)

    Parameters
    ----------
    in_channels: int
        Number of channels in the input image.
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int
        Size of the convolution kernel. Only kernels with the same height and width are supported.
    scales: list[int]
        List of scales to which images will be rescaled and convolution will be performed
    initial: bool
        Whether the layer is the initial layer within the network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, kernel_sizes=None, initial=False):
        """"""
        if kernel_sizes is None:
            kernel_sizes = [kernel_size]
        self.num_groups = 1 if initial else len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.max_kernel_size = max(kernel_sizes)
        super().__init__(in_channels, out_channels, kernel_size, padding=self.max_kernel_size // 2, bias=False)

    def forward(self, x):
        kernel = self._get_transformed_kernel()
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding, groups=self.num_groups)

    def _get_transformed_kernel(self):
        """ Returns stack of scaled and interpolated kernels usable for convolution groups. """
        pad = (self.max_kernel_size - self.kernel_size[0]) // 2
        kernel = F.pad(self.weight, (pad, pad, pad, pad))
        kernel_norm = torch.norm(kernel, p=1, dim=(2, 3), keepdim=True)
        scaled_kernels = []
        for size in self.kernel_sizes:
            scaled_kernel = self._rescale4d(kernel, size)
            scaled_kernel_norm = torch.norm(scaled_kernel, p=1, dim=(2, 3), keepdim=True)
            scaled_kernel = scaled_kernel * kernel_norm / scaled_kernel_norm
            scaled_kernels.append(scaled_kernel)
        return torch.cat(scaled_kernels)

    def _rescale4d(self, x: torch.Tensor, size):
        """ Rescales a 4D tensor while preserving its shape. """
        if size == self.kernel_size[0]:
            return x
        # Interpolate input image or feature map
        if size > self.kernel_size[0]:  # `nearest` upscaling, method with smallest l2 norm paired with bilinear
            rescaled_x = F.interpolate(x, size=(size, size), mode='nearest')
        else:  # `bilinear` downscaling
            rescaled_x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)
        # Apply appropriate zero padding (amount may be negative if output is larger)
        _, _, H, W = x.shape
        _, _, h, w = rescaled_x.shape
        pad_l, pad_t = (W - w) // 2, (H - h) // 2
        pad_r, pad_b = W - w - pad_l, H - h - pad_t
        return F.pad(rescaled_x, (pad_l, pad_r, pad_t, pad_b))
