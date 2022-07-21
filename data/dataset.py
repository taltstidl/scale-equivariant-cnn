import os
import warnings

import numpy as np


class STIRDataset:
    def __init__(self, path='emoji.npz'):
        if not os.path.exists(path):
            warnings.warn('Downloading Scaled and Translated Icon Recognition Dataset')
            return
        data = np.load(path)
        assert data['imgs'].shape[0] == data['lbls'].shape[0], \
            'Image count and label count don\'t match'
        self.images = data['imgs']
        self.labels = data['lbls']
        self.scales = data['scls']
        self.positions = data['psts']
        self.metadata = {m[0]: m[1] for m in data['metadata']}
        self.labeldata = data['lbldata']
        # Retrieve number of classes and channels
        self.num_classes = self.images.shape[2]
        self.num_channels = self.images.shape[6] if len(self.images.shape) == 7 else 1

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int), \
            'Dataset was indexed using {}, should be int'.format(type(index))
        return (self.images[index], self.labels[index])

    def __iter__(self):
        count = self.images.shape[0]
        for i in range(count):
            yield (self.images[i], self.labels[i])

    @staticmethod
    def encode_labels(labels):
        """ One-hot encode the input labels. """
        length = labels.shape[0]
        categorical = np.zeros((length, 36), dtype=int)
        categorical[np.arange(length), labels] = 1
        return categorical

    @staticmethod
    def decode_labels(labels):
        """ One-hot decode the input labels. """
        indices = np.argmax(labels)
        return indices

    @staticmethod
    def _check_arguments(split, scales):
        """ Checks whether the arguments are within the expected bounds. """
        if split not in ['train', 'valid', 'test']:
            raise ValueError('Split must be either `train`, `valid`, or `split`, got `{}`'.format(split))
        if any([s < 17 or s > 64 for s in scales]):
            raise ValueError('Scales must be between 17 and 64 (both inclusive), got `{}`'.format(scales))

    @staticmethod
    def _shuffle_arrays(*arrays):
        """ Randomly shuffle multiple arrays in unison with given seed. """
        for array in arrays:
            random = np.random.RandomState(seed=42)
            random.shuffle(array)

    def to_numpy(self, split='train', scales=range(64, 16, -1), shuffle=True):
        """ Convert data into a ndarray compatible with Keras. """
        self._check_arguments(split, scales)
        num_classes = self.labeldata.shape[0]
        # Compute start and stop indices for split and scale range
        split_i = {'train': 0, 'valid': 1, 'test': 2}[split] * self.images.shape[0] // 3
        scale_i = [64 - s for s in scales]
        # Retrieve appropriate images and labels
        images, labels = self.images[split_i, scale_i], self.labels[split_i, scale_i]
        # Reshape arrays
        if len(images.shape) == 5:
            images = images.reshape((-1, 64, 64, 1))
        elif len(images.shape) == 6:
            images = images.reshape((-1, 64, 64, 3))
        images = images.transpose((0, 3, 1, 2))
        labels = labels.reshape((-1,))
        # Convert from [0, 255] range to [0.0, 1.0] range
        images = images.astype(np.float32) / 255.0
        if shuffle:
            self._shuffle_arrays(images, labels)
        return images, labels

    def to_torch(self, split='train', scales=range(64, 16, -1), shuffle=True):
        """ Convert data into a TensorDataset compatible with PyTorch. """
        import torch
        from torch.utils.data import TensorDataset
        images, labels = self.to_numpy(split, scales, shuffle)
        images = torch.as_tensor(images, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return TensorDataset(images, labels)

    def get_latents(self, split='train', scales=range(64, 16, -1), shuffle=True):
        """ Retrieve the latent scales and positions. """
        self._check_arguments(split, scales)
        num_icons = self.labeldata.shape[0]
        # Compute start and stop indices for split and scale range
        split_i = {'train': 0, 'valid': 1, 'test': 2}[split] * self.images.shape[0] // 3
        scale_i = [64 - s for s in scales]
        # Retrieve appropriate scales and translations
        scales, positions = self.scales[split_i, scale_i], self.positions[split_i, scale_i]
        # Reshape arrays
        scales = scales.reshape((-1,))
        positions = positions.reshape((-1, 2))
        if shuffle:
            self._shuffle_arrays(scales, positions)
        return scales, positions
