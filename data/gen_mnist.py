import gzip
import os
import warnings
import struct

import numpy as np
from skimage.transform import resize
from skimage.filters import gaussian


def read_idx(file_path, dims):
    """ Reads idx file format used for MNIST images and labels. """
    with gzip.open(file_path, 'r') as file:
        buf = file.read(4 + dims * 4)  # Header
        magic, *shape = struct.unpack('>' + (1 + dims) * 'I', buf)
        if magic != 2049 and magic != 2051:
            warnings.warn('Magic number was {}, but expected 2049 or 2051.'.format(magic))
        buf = file.read()  # Data
        data = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        return data


def rescaled_mnist(images, labels, digit, i, scale):
    """ Gets rescaled MNIST image of digit at specified scale. """
    # See also: https://arxiv.org/abs/2004.01536
    digits = images[labels == digit]
    # Shuffle images of selected digit in consistent manner
    rng = np.random.RandomState(42)
    rng.shuffle(digits)
    # Select i-th image of selected digit
    image = digits[i]
    # Rescale image to target size
    image = resize(image, (scale, scale), order=3, preserve_range=True)  # bicubic interpolation
    # Apply random padding to complete 64x64 image
    pad_x, pad_y = np.random.randint(0, 64 - scale + 1, 2)
    left, right, top, bottom = pad_x, 64 - scale - pad_x, pad_y, 64 - scale - pad_y
    image = np.pad(image, ((top, bottom), (left, right)))
    # Apply Gaussian smoothing
    image = gaussian(image, sigma=7. / 8. * scale / 28.)
    # Apply non-linear thresholding
    image = 2 / np.pi * np.arctan(0.02 * (image - 128))
    # Normalize to range [0, 255]
    image = (255. * (image - image.min()) / (image.max() - image.min()))
    return image.astype(np.uint8), (pad_x, pad_y)


def generate():
    # Set seed for reproducibility
    np.random.seed(42)
    # Load images and labels
    train_images = read_idx(os.path.join('mnist', 'train-images-idx3-ubyte.gz'), 3)
    train_labels = read_idx(os.path.join('mnist', 'train-labels-idx1-ubyte.gz'), 1)
    test_images = read_idx(os.path.join('mnist', 't10k-images-idx3-ubyte.gz'), 3)
    test_labels = read_idx(os.path.join('mnist', 't10k-labels-idx1-ubyte.gz'), 1)
    # Create empty package contents
    images = [[[] for _ in range(48)] for _ in range(3)]
    labels = [[[] for _ in range(48)] for _ in range(3)]
    scales = [[[] for _ in range(48)] for _ in range(3)]
    translations = [[[] for _ in range(48)] for _ in range(3)]
    # Iterate over all combinations to generate images
    for index in range(10):  # 10 different digits
        for i, scale in enumerate(range(64, 16, -1)):  # 32 different scales
            for j in range(3):  # 3 sets (training, validation, testing)
                # Get correct origin set
                src_images = test_images if i == 2 else train_images
                src_labels = test_labels if i == 2 else train_labels
                # Add a rescaled MNIST image to the dataset
                image, translation = rescaled_mnist(src_images, src_labels, index, i + j * 32, scale)
                images[j][i].append(image)
                labels[j][i].append(index)
                scales[j][i].append(scale)
                translations[j][i].append(translation)
    # Collect metadata, two-dimensional numpy array to avoid pickling
    metadata = np.array([
        ['title', 'Scaled and Translated Image Recognition (STIR) MNIST'],
        ['description',
         'Testing data for scale invariance. 10 digits rescaled to sizes between 17x17 and 64x64 pixels with random position, constrained by image bounds. White digit on black background.'],
        ['author', 'Thomas R. Altstidl (thomas.r.altstidl@fau.de)'],
        ['license',
         'CC BY 4.0 modified from MNIST by Y. LeCun, C. Cortes and C. Burges - http://yann.lecun.com/exdb/mnist/'],
        ['version', '1.0.0'],
        ['date', '24 May 2022']
    ])
    lbldata = np.array([str(i) for i in range(10)])
    # Save data file
    imgs, lbls, scls, psts = np.array(images), np.array(labels), np.array(scales), np.array(translations)
    np.savez_compressed('mnist.npz', imgs=imgs, lbls=lbls, scls=scls, psts=psts,
                        metadata=metadata, lbldata=lbldata)


if __name__ == '__main__':
    generate()
