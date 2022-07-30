from torch.utils.data import DataLoader

from data.dataset import STIRDataset


class STIRDataModule:
    """ Scaled and translated image recognition data module.

    Parameters
    ----------
    data: str
        Key of dataset used for scaled and translated images. Corresponds to file name within online dataset.
    batch_size: int
        Number of training samples per batch.
    evaluation: int
        Evaluation scheme used, as defined in the paper. The evaluation scheme determines the selection of scales that
        participate in training and in testing.
    """
    def __init__(self, data: str = 'emoji', batch_size: int = 32, evaluation: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.evaluation = evaluation
        self.dataset = STIRDataset(path='{}.npz'.format(data))
        self.num_classes = self.dataset.num_classes
        self.num_channels = self.dataset.num_channels
        self.split_dataset()

    def split_dataset(self):
        if self.evaluation == 1:  # Train on first third, evaluate on other scales
            train_val_scales, test_scales = list(range(17, 33)), list(range(33, 65))
            self.train = self.dataset.to_torch(split='train', scales=train_val_scales)
            self.valid = self.dataset.to_torch(split='valid', scales=train_val_scales)
            self.test = self.dataset.to_torch(split='test', scales=test_scales)
        if self.evaluation == 2:  # Train on middle third, evaluate on other scales
            train_val_scales, test_scales = list(range(33, 49)), list(range(17, 33)) + list(range(49, 65))
            self.train = self.dataset.to_torch(split='train', scales=train_val_scales)
            self.valid = self.dataset.to_torch(split='valid', scales=train_val_scales)
            self.test = self.dataset.to_torch(split='test', scales=test_scales)
        if self.evaluation == 3:  # Train on last third, evaluate on other scales
            train_val_scales, test_scales = list(range(49, 65)), list(range(17, 49))
            self.train = self.dataset.to_torch(split='train', scales=train_val_scales)
            self.valid = self.dataset.to_torch(split='valid', scales=train_val_scales)
            self.test = self.dataset.to_torch(split='test', scales=test_scales)
        if self.evaluation == 4:  # Train on all scales, evaluate on all scales
            scales = range(17, 65)
            self.train = self.dataset.to_torch(split='train', scales=scales)
            self.valid = self.dataset.to_torch(split='valid', scales=scales)
            self.test = self.dataset.to_torch(split='test', scales=scales)

    def train_loader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def valid_loader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_loader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def to(self, device):
        self.train.tensors = [t.to(device) for t in self.train.tensors]
        self.valid.tensors = [t.to(device) for t in self.valid.tensors]
        self.test.tensors = [t.to(device) for t in self.test.tensors]
