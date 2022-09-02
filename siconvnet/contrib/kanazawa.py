# pylint: disable=no-member
import torch
import torch.nn.functional as F
from torch import nn


class KanazawaConv2d(nn.Conv2d):
    """ Locally Scale-Invariant Convolutional Neural Networks.

    Paper: Kanazawa et al., https://arxiv.org/pdf/1412.5104.pdf
    Implementation: https://github.com/ISosnovik/sesn/blob/master/models/impl/scale_modules.py

    Parameters
    ----------
    in_channels: int
        Number of channels in the input image.
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int
        Size of the convolution kernel. Only kernels with the same height and width are supported.
    scales: list[int]
        List of scales to which images will be rescaled and convolution will be performed.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scales=None):
        """"""
        super().__init__(in_channels, out_channels, kernel_size)
        if scales is None:
            scales = [1]
        self.scales = scales

    def forward(self, x):
        # Rescale inputs to different scales
        stack = []
        for scale in self.scales:
            x_rescaled = KanazawaConv2d._rescale4d(x, scale)
            stack.append(x_rescaled)
        xs = torch.stack(stack, 0)
        # Apply standard convolution on stack
        xs = KanazawaConv2d._batchify(xs)
        xs = super().forward(xs)
        xs = KanazawaConv2d._unbatchify(xs, len(self.scales)).unbind(0)
        # Normalize outputs to consistent scale
        output_stack = []
        for x, scale in zip(xs, self.scales):
            x_rescaled = KanazawaConv2d._rescale4d(x, 1 / scale)
            output_stack.append(x_rescaled)
        y = torch.stack(output_stack, 0)
        # Perform maximum pooling across scale domain
        return y.max(0)[0]

    @staticmethod
    def _batchify(x: torch.Tensor):
        # s: scales, n: batch, c: channels, h: height, w: width
        scales, n, c, h, w = x.shape
        return x.view((scales * n, c, h, w))

    @staticmethod
    def _unbatchify(x: torch.Tensor, scales):
        # s: scales, n: batch, c: channels, h: height, w: width
        _, c, h, w = x.shape
        return x.view((scales, -1, c, h, w))

    @staticmethod
    def _rescale4d(x: torch.Tensor, scale):
        """ Rescales a 4D tensor while preserving its shape. """
        if scale == 1:
            return x
        # Interpolate input image or feature map
        rescaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
        # Apply appropriate zero padding (amount may be negative if output is larger)
        _, _, H, W = x.shape
        _, _, h, w = rescaled_x.shape
        pad_l, pad_t = (W - w) // 2, (H - h) // 2
        pad_r, pad_b = W - w - pad_l, H - h - pad_t
        return F.pad(rescaled_x, (pad_l, pad_r, pad_t, pad_b))
