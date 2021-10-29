from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


class TrafficSignDataModule:
    """ Traffic sign data module.

    Parameters
    ----------
    batch_size: int
        Number of training samples per batch.
    """
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.batch_size = batch_size
        transform = ToTensor()
        self.signs_train = ImageFolder('signs/train', transform=transform)
        self.signs_valid = ImageFolder('signs/val', transform=transform)
        self.signs_test = ImageFolder('signs/test', transform=transform)

    def train_loader(self):
        return DataLoader(self.signs_train, batch_size=self.batch_size)

    def valid_loader(self):
        return DataLoader(self.signs_valid, batch_size=self.batch_size)

    def test_loader(self):
        return DataLoader(self.signs_test, batch_size=self.batch_size)
