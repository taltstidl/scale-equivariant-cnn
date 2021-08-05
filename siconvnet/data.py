from typing import Optional

from torch.utils.data import DataLoader
from dataset import EmojiDataset


class EmojiDataModule:
    """ Emoji data module.

    Parameters
    ----------
    batch_size: int
        Number of training samples per batch.
    evaluation: int
        Evaluation scheme used, as defined in the paper. The evaluation scheme determines the selection of scales that
        participate in training and in testing.
    """
    def __init__(self, batch_size: int = 32, evaluation: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.evaluation = evaluation
        self.dataset = EmojiDataset()
        self.split_dataset()

    def split_dataset(self):
        if self.evaluation == 1:  # Train on all scales, evaluate on all scales
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=64, scale_stop=32)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=64, scale_stop=32)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=64, scale_stop=32)
        if self.evaluation == 2:  # Train on half the scales, evaluate on other half
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=64, scale_stop=48)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=64, scale_stop=48)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=48, scale_stop=32)
        if self.evaluation == 3:  # Train on half the scales, evaluate on other half
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=48, scale_stop=32)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=48, scale_stop=32)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=64, scale_stop=48)
        if self.evaluation == 4:  # Train with single scale, evaluate with all other scales
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=64, scale_stop=63)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=64, scale_stop=63)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=63, scale_stop=32)
        if self.evaluation == 5:  # Train with single scale, evaluate with all other scales
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=33, scale_stop=32)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=33, scale_stop=32)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=64, scale_stop=33)

    def train_loader(self):
        return DataLoader(self.emoji_train, batch_size=self.batch_size)

    def valid_loader(self):
        return DataLoader(self.emoji_valid, batch_size=self.batch_size)

    def test_loader(self):
        return DataLoader(self.emoji_test, batch_size=self.batch_size)

    def to(self, device):
        self.emoji_train.tensors = [t.to(device) for t in self.emoji_train.tensors]
        self.emoji_valid.tensors = [t.to(device) for t in self.emoji_valid.tensors]
        self.emoji_test.tensors = [t.to(device) for t in self.emoji_test.tensors]
