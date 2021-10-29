# pylint: disable=no-member
"""
This module contains the models presented in the paper.
"""
import numpy as np
from torch import nn

from siconvnet.layers import SiConv2d, ScalePool, GlobalMaxPool


class BaseModel(nn.Module):
    """ Base class for all models.

    This model establishes a common baseline for all experiments. More specifically, it applies the cross entropy loss,
    records the training and validation accuracy and configures an Adam optimizer.
    """

    def __init__(self, kernel_size, interpolation):
        """"""
        super().__init__()
        self.kernel_size = kernel_size
        self.interpolation = interpolation
        self.tracing = False
        self.tracing_cache = {}

    def compute_params(self):
        im_size = 64  # Size of input image, here fixed to 64
        first_scales = (im_size - self.kernel_size) // 2 + 1
        im_size -= self.kernel_size - 1
        second_scales = (im_size - self.kernel_size) // 2 + 1
        return self.kernel_size, (first_scales, second_scales)

    def enable_tracing(self):
        self.tracing = True

    def save_trace(self, name, tensor):
        if not self.tracing:
            return
        if name not in self.tracing_cache:
            self.tracing_cache[name] = []
        tensor = tensor.cpu().numpy()
        self.tracing_cache[name].append(tensor)

    def get_traces(self):
        for name in self.tracing_cache:
            arrays = self.tracing_cache[name]
            concat = np.concatenate(arrays)
            self.tracing_cache[name] = concat
        return self.tracing_cache

    def forward(self, x):
        """"""
        pass


class StandardModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = nn.Conv2d(1, 16, (k, k))
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, (k, k))
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.act1(x)
        self.save_trace('stage1', x)
        x = self.conv2(x)
        x = self.act2(x)
        self.save_trace('stage2', x)
        x = self.global_pool(x)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('predictions', x)
        return x


class PixelPoolModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = SiConv2d(1, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.pool1 = ScalePool(mode='pixel')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, num_scales[1], k, interp_mode=self.interpolation)
        self.pool2 = ScalePool(mode='pixel')
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act1(x)
        self.save_trace('stage1', x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)
        self.save_trace('stage2', x)
        x = self.global_pool(x)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('prediction', x)
        return x


class SlicePoolModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = SiConv2d(1, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.pool1 = ScalePool(mode='slice')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, num_scales[1], k, interp_mode=self.interpolation)
        self.pool2 = ScalePool(mode='slice')
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act1(x)
        self.save_trace('stage1', x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)
        self.save_trace('stage2', x)
        x = self.global_pool(x)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('predictions', x)
        return x


class Conv3dModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = SiConv2d(1, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(16, 32, (k, k, k))
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('predictions', x)
        return x
