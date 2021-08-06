import os
import warnings

import numpy as np


class EmojiDataset:
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
        self.icondata = data['icondata']

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
    def _check_arguments(split, scale_start, scale_stop):
        """ Checks whether the arguments are within the expected bounds. """
        if split not in ['train', 'valid', 'test']:
            raise ValueError('Split must be either `train`, `valid`, or `split`, got `{}`'.format(split))
        if scale_start < 33 or scale_start > 64:
            raise ValueError('Scale start must be between 33 and 64 (both inclusive), got `{}`'.format(scale_start))
        if scale_stop < 32 or scale_stop > 63:
            raise ValueError('Scale stop must be between 32 and 63 (both inclusive), got `{}`'.format(scale_stop))
        if scale_start <= scale_stop:
            raise ValueError('Scale start must be larger than stop, got `{}` and `{}`'.format(scale_start, scale_stop))

    @staticmethod
    def _shuffle_arrays(*arrays):
        """ Randomly shuffle multiple arrays in unison with given seed. """
        for array in arrays:
            random = np.random.RandomState(seed=42)
            random.shuffle(array)

    def to_numpy(self, split='train', scale_start=64, scale_stop=32, shuffle=True):
        """ Convert data into a ndarray compatible with Keras. """
        self._check_arguments(split, scale_start, scale_stop)
        num_icons = self.icondata.shape[0]
        # Compute start and stop indices for split and scale range
        offset = {'train': 0, 'valid': 1, 'test': 2}[split] * self.images.shape[0] // 3
        start = offset + (64 - scale_start) * num_icons
        stop = offset + (64 - scale_stop) * num_icons
        # Retrieve appropriate images and labels
        images, labels = self.images[start:stop], self.labels[start:stop]
        images = np.expand_dims(images, axis=1).astype(np.float32) / 255.0
        if shuffle:
            self._shuffle_arrays(images, labels)
        return images, labels

    def to_torch(self, split='train', scale_start=64, scale_stop=32, shuffle=True):
        """ Convert data into a TensorDataset compatible with PyTorch. """
        import torch
        from torch.utils.data import TensorDataset
        images, labels = self.to_numpy(split, scale_start, scale_stop, shuffle)
        images = torch.as_tensor(images, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return TensorDataset(images, labels)

    def get_latents(self, split='train', scale_start=64, scale_stop=32, shuffle=True):
        """ Retrieve the latent scales and positions. """
        self._check_arguments(split, scale_start, scale_stop)
        num_icons = self.icondata.shape[0]
        # Compute start and stop indices for split and scale range
        offset = {'train': 0, 'valid': 1, 'test': 2}[split] * self.images.shape[0] // 3
        start = offset + (64 - scale_start) * num_icons
        stop = offset + (64 - scale_stop) * num_icons
        # Retrieve appropriate scales and translations
        scales, positions = self.scales[start:stop], self.positions[start:stop]
        if shuffle:
            self._shuffle_arrays(scales, positions)
        return scales, positions
