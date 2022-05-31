# Scaled and Translated Image Recognition

While convolutions are known to be invariant to (discrete) translations, scaling continues to be a challenge and most image recognition networks are not invariant to them. To explore these effects, we have created the Scaled and Translated Image Recognition (STIR) dataset. This dataset contains objects of size $s \in [17,64]$, each randomly placed in a $64 \times 64$ pixel image.

## Using the dataset

Depending on which data you are planning to use, download one or more of the following files. Data is stored in compressed `.npz` format and can be loaded as documented [here](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

| File | Description | Download |
| ---- | ----------- | -------- |
| `emoji.npz` | Emoji vector icons rendered as white icon on black background | [on Zenodo]() |
| `mnist.npz` | Classic MNIST handwritten digits rescaled to varying sizes | [on Zenodo]() |
| `trafficsign.npz` | Traffic signs from street imagery downscaled to varying sizes | [on Zenodo]() |

Each file contains multiple arrays that can be accessed in a dictionary-like fashion. The keys are documented below, where `n` is the number of classes for a given file. Both `emoji.npz` (36 classes) and `mnist.npz` (10 classes) are in black & white while `trafficsign.npz` (40 classes) is in color.

| Key | Shape | Description |
| --- | ----- | ----------- |
| `imgs` | `(3, 48, n, 64, 64)` black & white, `(3, 48, n, 64, 64, 3)` color | Images grouped into 3 sets (training, validation, testing) and 48 different scales. Values will be in range `0` to `255`. |
| `lbls` | `(3, 48, n)` | Indices referencing ground truth labels. Values will be in range `0` to `n - 1`. |
| `scls` | `(3, 48, n)` | Known scales as given by bounding box size. Values will be in range `17` to `64`. |
| `psts` | `(3, 48, n, 2)` | Known position of bounding box. First value is distance to left edge, second value distance to top edge. |
| `metadata` | `(6, 2)` | Metadata on title, description, author, license, version and date. |
| `lbldata` | `(n,)` | Descriptive names for each ground truth labels. |

For use in Python a dataset class is provided that implements the basic functionality for loading a certain split and scale selection, as illustrated in the code below. It ensures shuffling is done in a consistent manner such that ground truth scales and positions can be retrieved. Metadata and label descriptions can be retrieved via `metadata` and `labeldata`, respectively.

```python
from data.dataset import STIRDataset

dataset = STIRDataset('data/emoji.npz')
# Obtain images and labels for training
images, labels = dataset.to_torch(split='train', scales=[32, 64], shuffle=True)
# Obtain known scales and positions for above
scales, positions = dataset.get_latents(split='train', scales=[32, 64], shuffle=True)
# Get metadata and label descriptions
metadata = dataset.metadata
label_descriptions = dataset.labeldata
```

## Generating the dataset

The scripts for generating the datasets are provided for transparency and reproducibility. They are prefixed by `gen_*` and should run without arguments. Downloaded files expected within the working directory are given below.

* `fontawesome/` (from [Font Awesome](https://fontawesome.com/v5/download) 5.15.3 "Free for Desktop")
  * `svgs/` unzipped from archive
* `mapillary/` (from [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign))
  * `mtsd_v2_fully_annotated` unzipped from archive
  * `train.0.zip` **not** unzipped
  * `train.1.zip` **not** unzipped
  * `train.2.zip` **not** unzipped
  * `val.zip` **not** unzipped
* `mnist/` (from [Yann LeCun](http://yann.lecun.com/exdb/mnist/) website)
  * `t10k-images-idx3-ubyte.gz`
  * `t10k-labels-idx1-ubyte.gz`
  * `train-images-idx3-ubyte.gz`
  * `train-labels-idx1-ubyte.gz`