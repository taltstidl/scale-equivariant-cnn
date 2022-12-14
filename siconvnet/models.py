# pylint: disable=no-member
"""
This module contains the models presented in the paper.
"""
import torch
import torch.nn.functional as F
from kornia.geometry import build_pyramid
from torch import nn

from siconvnet.layers import SiConv2d, ScalePool, GlobalMaxPool
from siconvnet.contrib.xu import XuConv2d
from siconvnet.contrib.kanazawa import KanazawaConv2d
from siconvnet.contrib.steerable import SESConv_Z2_H, SESConv_H_H


class BaseModel(nn.Module):
    """ Base class for all models.

    This model establishes a common baseline for all experiments. More specifically, it applies the cross entropy loss,
    records the training and validation accuracy and configures an Adam optimizer.
    """

    def __init__(self, kernel_size, interpolation, num_channels, num_classes):
        """"""
        super().__init__()
        self.kernel_size = kernel_size
        self.interpolation = interpolation
        self.num_channels = num_channels
        self.num_classes = num_classes
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

    def disable_tracing(self):
        del self.tracing_cache
        self.tracing = False
        self.tracing_cache = {}

    def save_trace(self, name, tensor):
        if not self.tracing:
            return
        # if name not in self.tracing_cache:
        #     self.tracing_cache[name] = []
        # tensor = tensor.cpu().numpy()
        # self.tracing_cache[name].append(tensor)
        self.tracing_cache[name] = tensor.detach().cpu()

    def get_traces(self):
        # for name in self.tracing_cache:
        #     arrays = self.tracing_cache[name]
        #     concat = np.concatenate(arrays)
        #     self.tracing_cache[name] = concat
        return self.tracing_cache

    def forward(self, x):
        """"""
        pass


class StandardModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = nn.Conv2d(self.num_channels, 16, (k, k))
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, (k, k))
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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
        self.conv1 = SiConv2d(self.num_channels, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.pool1 = ScalePool(mode='pixel')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, num_scales[1], k, interp_mode=self.interpolation)
        self.pool2 = ScalePool(mode='pixel')
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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


class SlicePoolModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = SiConv2d(self.num_channels, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.pool1 = ScalePool(mode='slice')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, num_scales[1], k, interp_mode=self.interpolation)
        self.pool2 = ScalePool(mode='slice')
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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


class EnergyPoolModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.conv1 = SiConv2d(self.num_channels, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.pool1 = ScalePool(mode='energy')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, num_scales[1], k, interp_mode=self.interpolation)
        self.pool2 = ScalePool(mode='energy')
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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
        self.conv1 = SiConv2d(self.num_channels, 16, num_scales[0], k, interp_mode=self.interpolation)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(16, 32, (k, k, k))
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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


class EnsembleModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        p = (k - 1) // 2
        conv1 = nn.Conv2d(self.num_channels, 16, (k, k), padding=(p, p))
        act1 = nn.ReLU()
        conv2 = nn.Conv2d(16, 32, (k, k), padding=(p, p))
        act2 = nn.ReLU()
        global_pool = GlobalMaxPool()
        lin = nn.Linear(32, self.num_classes)
        self.base_model = nn.Sequential(conv1, act1, conv2, act2, global_pool, lin)

    def forward(self, x):
        """"""
        pyramid = build_pyramid(x, max_level=3)
        energies = [self.base_model(p) for p in pyramid]
        energies = torch.stack(energies)
        return torch.mean(energies, dim=0)


class SpatialTransformModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        conv1 = nn.Conv2d(self.num_channels, 16, (k, k))
        act1 = nn.ReLU()
        conv2 = nn.Conv2d(16, 32, (k, k))
        act2 = nn.ReLU()
        global_pool = GlobalMaxPool()
        lin = nn.Linear(32, self.num_classes)
        self.base_model = nn.Sequential(conv1, act1, conv2, act2, global_pool, lin)
        # Build transformer model
        self.transform_model = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, (3, 3), padding=(1, 1)),
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
        """"""
        theta = self.transform_model(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        """"""
        x = self.transform(x)
        return self.base_model(x)


class XuModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.kernel_sizes = [3, 5, 7, 9, 11]
        self.conv1 = XuConv2d(self.num_channels, 16, k, self.kernel_sizes, initial=True)
        self.act1 = nn.ReLU()
        self.conv2 = XuConv2d(16, 32, k, self.kernel_sizes)
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

    def column_pool(self, x):
        """"""
        b, c, h, w = x.shape
        x = x.view(b, len(self.kernel_sizes), c // len(self.kernel_sizes), h, w)
        x = x.max(dim=1)[0]
        return x

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.column_pool(x)
        x = self.global_pool(x)
        x = self.lin(x)
        return x


class KanazawaModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.scales = [1.26**e for e in range(-2, 4)]
        self.conv1 = KanazawaConv2d(self.num_channels, 16, k, self.scales)
        self.act1 = nn.ReLU()
        self.conv2 = KanazawaConv2d(16, 32, k, self.scales)
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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


class HermiteModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.scales = [2.0, 2.52, 3.17, 4.0]
        basis_kwargs = {
            'basis_type': 'hermite_b',
            'basis_mult': 1.5,
            'basis_max_order': 4,
        }
        self.conv1 = SESConv_Z2_H(self.num_channels, 16, 15, k, bias=True, scales=self.scales, **basis_kwargs)
        self.act1 = nn.ReLU()
        self.conv2 = SESConv_H_H(16, 32, 1, 15, k, bias=True, scales=self.scales, **basis_kwargs)
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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


class DiscoModel(BaseModel):
    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        k, num_scales = self.compute_params()
        self.scales = [1.0, 1.26, 1.59, 2.0]
        basis_kwargs = {
            'basis_type': 'disco_b',
            'basis_mult': 1.9,
            'basis_max_order': 4,
            'basis_min_scale': 1.9,
            'basis_save_dir': '../siconvnet/contrib',
        }
        self.conv1 = SESConv_Z2_H(self.num_channels, 16, 15, k, bias=True, scales=self.scales, **basis_kwargs)
        self.act1 = nn.ReLU()
        self.conv2 = SESConv_H_H(16, 32, 1, 15, k, bias=True, scales=self.scales, **basis_kwargs)
        self.act2 = nn.ReLU()
        self.global_pool = GlobalMaxPool()
        self.lin = nn.Linear(32, self.num_classes)

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
