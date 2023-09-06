# Just a Matter of Scale? Scale Equivariance in CNNs

> [!NOTE]
> A detailed analysis of scale generalization for various models is given in our preprint
> 
> [Just a Matter of Scale? Reevaluating Scale Equivariance in Convolutional Neural Networks](https://arxiv.org/abs/2211.10288)  
> **Thomas Altstidl, An Nguyen, Leo Schwinn, Franz Köferl, Christopher Mutschler, Björn Eskofier, Dario Zanca**

This repository contains the official source code accompanying our preprint. If you are reading this, it is likely that you fall into one or more of the following groups. Click on those that are applicable for you to get started.

<details>
<summary><strong>I am interested in using the Scaled and Translated Image Recognition (STIR) dataset.</strong></summary>

* Download one or more data files from [Zenodo](https://zenodo.org/record/6578038).
* Grab a copy of [dataset.py](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/data/dataset.py).
* Example usage that loads training data from `emoji.npz` for scales 17 through 64.
```python
from dataset import STIRDataset

dataset = STIRDataset('data/emoji.npz')
# Obtain images and labels for training
images, labels = dataset.to_torch(split='train', scales=range(17, 65), shuffle=True)
# Obtain known scales and positions for above
scales, positions = dataset.get_latents(split='train', scales=range(17, 65), shuffle=True)
# Get metadata and label descriptions
metadata = dataset.metadata
label_descriptions = dataset.labeldata
```
</details>

<details>
<summary><strong>I am interested in reviewing your results.</strong></summary>

We provide a subset of our results for review. Others are available upon request as they are larger in size.
* [clean.csv](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/scripts/clean.csv) contains testing accuracy and time (columns `metrics.test_acc` and `metrics.train_time`)
* [generalization.csv](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/plots/generalization.csv) contains accuracies per scale (columns `s17` through `s64`)
</details>

<details>
<summary><strong>I am interested in using the proposed layer in my own work.</strong></summary>

* Grab a copy of [layers.py](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/siconvnet/layers.py).
* Example usage that applies one 7x7 scaled convolutional layer followed by pixel-wise pooling.
```python
from torch import nn
from layers import SiConv2d, ScalePool

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 7x7 base kernel rescaled to 29 different scales
        self.conv = SiConv2d(3, 16, 29, 7, interp_mode='bicubic')
        self.pool = ScalePool(mode='pixel')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
```
</details>

The remainer of this document will focus on reproducing the results given in our preprint.

## Reproducing Results

> [!WARNING]
> While we have taken great care to document everything, the scope of this project makes it likely that minor details may still be missing. If you have trouble recreating our experiments on your own machines, please create a new [issue](https://github.com/taltstidl/scale-equivariant-cnn/issues/new/choose) and we'd be more than happy to assist.

### Setting up environment

The provided code should work in most environments and has been tested to work at least in Windows 10/11 (local environment) and Linux (cluster node environment). Python 3.8 was used, although newer versions should also work. We recommend creating a new virtual environment and installing all requirements there:

```bash
cd /path/to/provided/code
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```

### Training models

Before training a model, you will need to either create or download the respective data files you intend to use. These can be downloaded from [Zenodo](https://zenodo.org/record/6578038). Then, execute the following script with your selected parameters to train a single model.

```bash
python scripts/train.py [...]
```

* **--model** _{standard, pixel_pool, slice_pool, energy_pool, conv3d, ensemble, spatial_transform, xu, kanazawa, hermite, disco}_
Name of (scale-equivariant) model that should be trained. Implementations are given in `siconvnet/models.py`.
* **--dataset** _{emoji, mnist, trafficsign, aerial}_
Name of dataset on which the model should be trained. The respective `[d].npz` file needs to be in the current working directory. See paper Fig. 3.
* **--evaluation** _{1, 2, 3, 4}_
Evaluation scenario on which the model should be trained. Defines scales for training and evaluation. See paper Fig. 3.
* **--kernel-size** _{3, 7, 11, 15}_
Kernel size of all convolutions. Defines size $k \times k$ of trainable kernel weights. Fixed to 7 in paper.
* **--interpolation** _{nearest, bilinear, bicubic, area}_
Interpolation method used to generate larger kernels. Only applies to our models. Fixed to bicubic in paper.
* **--lr** _{1e-2, 1e-3}_
Learning rate of Adam optimizer used to train model.
* **--seed** _number_
Seed used to initialize random number generators for reproducibility. Seeds used in paper are 1 through 50.

### Evaluating models

The training script writes results to MLflow. Before proceeding with the evaluation, you need to export all runs. Unless you changed the tracking destination, this is done using the following command. We provide our own filtered export in [clean.csv](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/scripts/clean.csv).

```bash
mlflow experiments csv -x 0 -o clean.csv
```

Then, execute the following script with your selected parameters to evaluate all models.

```bash
python scripts/eval.py [...]
```

* **--runs** _path/to/clean.csv_ Path to the exported runs from MLflow. Should point to file exported using above command.
* **--models** _path/to/models_ Path to the run artifacts saved by MLflow. Should be `mlruns/0` when run locally.
* **--data** _{emoji, mnist, trafficsign, aerial}_ Name of dataset for which models should be evaluated.
* **--generalization** Flag for scale generalization. If enabled, will write `generalization_*.csv` files.
* **--equivariance** Flag for scale equivariance. If enabled, will write `eval/*/errors.npz` files.
* **--index-correlation** Flag for pooling scale correlation. If enabled, will write `eval/*/indices.npz` files.

### Generating plots

To recreate the plots given in the paper and in the supplementary document you may use the scripts provided in the `plots/` directory. We provide [clean.csv](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/scripts/clean.csv) and [generalization.csv](https://github.com/taltstidl/scale-equivariant-cnn/blob/main/plots/generalization.csv) here. Others are available upon request as they are larger in size.

* **`equivariance.py`** was used for Fig. 6 & Suppl. Fig. 3 and requires both `scripts/clean.csv` and `plots/eval/*/errors.npz`
* **`generalization.py`** was used for Fig. 5 & Suppl. Fig. 2 and requires both `scripts/clean.csv` and `plots/generalization_*.csv`
* **`hyperparam.py`** was used for Fig. 4 & Suppl. Fig. 1 and requires only `scripts/clean.csv`
* **`indices.py`** was used for Fig. 7 & Suppl. Fig. 4 and requires both `scripts/clean.csv`and `plots/eval/*/indices.npz`
* **`time.py`** was used for Tab. 2 and requires only `scripts/clean.csv`
