# Just a Matter of Scale? Scale Equivariance in CNNs

This repository contains the source code for the paper "Just a Matter of Scale? Reevaluating Scale Equivariance in Convolutional Neural Networks". It is split into multiple modules.

* `data/` contains the code for generating the Scaled and Translated Image Recognition (STIR) benchmark
* `plots/` contains the code for creating the plots given in the paper and supplementary material
* `scripts/` contains the code for training and evaluating all models
* `siconvnet/` contains the actual implementation of all layers and models

## Setting up environment

The provided code should work in most environments and has been tested to work at least in Windows 10/11 (local environment) and Linux (cluster node environment). Python 3.8 was used, although newer versions should also work. We recommend creating a new virtual environment and installing all requirements there:

```bash
cd /path/to/provided/code
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```

## Training models

Before training a model, you will need to either create or download the respective dataset you intend to use. See the `data/` folder for additional instructions. Then, execute the following script with your selected parameters to train a single model.

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

## Generating plots

To recreate the plots given in the paper and in the supplementary document you may use the scripts provided in the `plots/` directory. All of them require result files, which are provided with the code.

* **`equivariance.py`** was used for Fig. 6 & Suppl. Fig. 3 and requires both `scripts/clean.csv` and `plots/eval/*/errors.npz`
* **`generalization.py`** was used for Fig. 5 & Suppl. Fig. 2 and requires both `scripts/clean.csv` and `plots/generalization_*.csv`
* **`hyperparam.py`** was used for Fig. 4 & Suppl. Fig. 1 requires only `scripts/clean.csv`
* **`indices.py`** was used for Fig. 7 & Suppl. Fig. 4 requires both `scripts/clean.csv`and `plots/eval/*/indices.npz`
* **`time.py`** was used for Tab. 2 and requires only `scripts/clean.csv`

The `scripts/clean.csv` contains the experiment runs that were exported from MLflow. The remaining result files were generated using `scripts/eval.py`.