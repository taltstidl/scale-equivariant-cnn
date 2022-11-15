# Model Directory

To properly evaluate the ability of CNN-based models to generalize towards previously unseen scales, we implement a set of existing and proposed models.

* `contrib/` contains third-party layer implementations. See individual files for further references.
* `data.py` contains a PyTorch data module compatible with the STIR benchmark. It also implements the evaluation scenario splits.
* `layers.py` contains the implementation of the scaled convolutional layer and the scale pooling methods. This is the file you will need when using it in your own work.
* `metrics.py` contains functions for computing evaluation metrics. This is primarily used to compute result files in `scripts/eval.py`.
* `models.py` contains the implementation of all existing and proposed models. An overview of these is given below.

## Overview of Implemented Models

| Name | Implementation Details | Reference |
| ---- | ---------------------- | --------- |
| Standard | Based on standard `torch.nn.Conv2d`. | None |
| SpatialTrans | Transforms input using `torch.nn.functional.affine_grid`. | [arXiv preprint](https://arxiv.org/abs/1506.02025) |
| Ensemble | Multi-column architecture based on Gaussian pyramid by [Kornia](https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.build_pyramid). | None | 
| Xu | Multi-column architecture with interpolated kernels. | [arXiv preprint](https://arxiv.org/abs/1411.6369) |
| Kanazawa | Applies convolutions on interpolated inputs followed by maximum pooling. | [arXiv preprint](https://arxiv.org/abs/1412.5104) |
| Hermite | Scale-equivariant steerable layer based on official [code](https://github.com/ISosnovik/disco) (see `contrib/steerable.py`). | [arXiv preprint](https://arxiv.org/abs/2106.02733) |
| Disco | Scale-equivariant steerable layer based on official [code](https://github.com/ISosnovik/disco) (see `contrib/steerable.py`). | [arXiv Preprint](https://arxiv.org/abs/1910.11093) |
| PixelPool | Uses `SiConv2d` and `ScalePool` with pooling mode `pixel`. | (Ours) |
| SlicePool | Uses `SiConv2d` and `ScalePool` with pooling mode `slice`. | (Ours) |
| EnergyPool | Uses `SiConv2d` and `ScalePool` with pooling mode `energy`. | (Ours) |
| Conv3d | Uses `SiConv2d` followed by standard `torch.nn.Conv3d`. | (Ours) |

## Usage of Proposed Layer

If you are interested in using our proposed layer in your own work, the easiest way to do so is to copy `layers.py`. It primarily contains two layers of interest:

* `SiConv2d` implements the scaled convolutional layer described in Sec. 3.1 of the paper. See docstring for further details on its parameters.
* `ScalePool` implements the pooling required to collapse the additional dimension. See docstring for further details on its parameters.

You will typically want to chain both of these as the pooling ensures that additional layers can be applied afterwards. An example of its usage is given below.

```python
from torch import nn
from siconvnet.layers import SiConv2d, ScalePool

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ...
        # 7x7 base kernel rescaled to 29 different scales
        self.conv = SiConv2d(3, 16, 29, 7, interp_mode='bicubic')
        self.pool = ScalePool(mode='pixel')
        # ...

    def forward(self, x):
        # ...
        x = self.conv(x)
        x = self.pool(x)
        # ...
```

Compared to standard convolutions, you will also need to specify the total number of (enlarged and interpolated) kernels that should be computed. This corresponds to the length of the scale output dimension. For interpolation, we find `bicubic` to work best, though others are supported as well. Scale pooling implements three different modes, which are described in detail in the docs.