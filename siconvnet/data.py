from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import EmojiDataset


class EmojiDataModule(pl.LightningDataModule):
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

    def prepare_data(self):
        pass  # No data preparation required

    def setup(self, stage: Optional[str] = None):
        self.dataset = EmojiDataset()
        if self.evaluation == 1:  # Train on all scales, evaluate on all scales
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=64, scale_stop=32)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=64, scale_stop=32)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=64, scale_stop=32)
        if self.evaluation == 2:  # Train on half the scales, evaluate on other half
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=64, scale_stop=48)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=64, scale_stop=48)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=48, scale_stop=32)
        if self.evaluation == 3:  # Train with single scale, evaluate with all other scales
            self.emoji_train = self.dataset.to_torch(split='train', scale_start=64, scale_stop=63)
            self.emoji_valid = self.dataset.to_torch(split='valid', scale_start=64, scale_stop=63)
            self.emoji_test = self.dataset.to_torch(split='test', scale_start=63, scale_stop=32)

    def train_dataloader(self):
        return DataLoader(self.emoji_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.emoji_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.emoji_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.test_dataloader()
