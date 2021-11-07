import numpy as np

from torch import DoubleTensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


class TrafficSignDataModule:
    """ Traffic sign data module.

    Parameters
    ----------
    batch_size: int
        Number of training samples per batch.
    """
    def __init__(self, batch_size: int = 16, evaluation: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.evaluation = evaluation
        transform = ToTensor()
        self.signs_train = ImageFolder('trafficsign/signs/evaluation{}/train'.format(evaluation), transform=transform)
        self.signs_valid = ImageFolder('trafficsign/signs/evaluation{}/val'.format(evaluation), transform=transform)
        self.signs_test = ImageFolder('trafficsign/signs/evaluation{}/test'.format(evaluation), transform=transform)

    def _create_sampler(self, dataset):
        """ Create custom sampler to ensure uniform and balanced class sampling. """
        targets = np.array(dataset.targets)
        labels, counts = np.unique(targets, return_counts=True)
        weights = 1. / counts
        samples = np.zeros_like(targets, dtype=np.float)
        for label, weight in zip(labels, weights):
            samples[targets == label] = weight
        samples = DoubleTensor(samples)
        sampler = WeightedRandomSampler(samples, targets.shape[0], replacement=False)
        return sampler

    def train_loader(self):
        # sampler = self._create_sampler(self.signs_train)
        return DataLoader(self.signs_train, batch_size=self.batch_size, shuffle=True)

    def valid_loader(self):
        return DataLoader(self.signs_valid, batch_size=self.batch_size)

    def test_loader(self):
        return DataLoader(self.signs_test, batch_size=self.batch_size)
