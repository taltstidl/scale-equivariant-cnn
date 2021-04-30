# pylint: disable=no-member
"""
This module contains the models presented in the paper.
"""
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from siconvnet.layers import SiConv2d, ScalePool


class BaseModel(pl.LightningModule):
    """ Base class for all models.

    This model establishes a common baseline for all experiments. More specifically, it applies the cross entropy loss,
    records the training and validation accuracy and configures an Adam optimizer.
    """

    def __init__(self):
        """"""
        super().__init__()
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.learning_rate = 1e-2
        self.tracing = False
        self.tracing_cache = {}

    def enable_tracing(self):
        self.tracing = True

    def save_trace(self, name, tensor):
        if not self.tracing:
            return
        if name not in self.tracing_cache:
            self.tracing_cache[name] = []
        tensor = tensor.cpu().numpy()
        self.tracing_cache[name].append(tensor)

    def get_traces(self):
        for name in self.tracing_cache:
            arrays = self.tracing_cache[name]
            concat = np.concatenate(arrays)
            self.tracing_cache[name] = concat
        return self.tracing_cache

    def training_step(self, batch, batch_idx):
        """"""
        images, labels = batch
        output = self.forward(images)
        loss = F.cross_entropy(output, labels)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc_step', self.train_acc(output.argmax(dim=1), labels))
        return loss

    def training_epoch_end(self, outs):
        """"""
        self.log('train_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        """"""
        images, labels = batch
        output = self.forward(images)
        loss = F.cross_entropy(output, labels)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc_step', self.val_acc(output.argmax(dim=1), labels))

    def validation_epoch_end(self, outs):
        """"""
        self.log('val_acc_epoch', self.val_acc.compute())

    def configure_optimizers(self):
        """"""
        return Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        """"""
        pass


class StandardModel(BaseModel):
    def __init__(self):
        """"""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (7, 7))
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, (7, 7))
        self.act2 = nn.ReLU()
        self.global_pool = nn.MaxPool2d((52, 52))
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.act1(x)
        self.save_trace('stage1', x)
        x = self.conv2(x)
        x = self.act2(x)
        self.save_trace('stage2', x)
        x = self.global_pool(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('predictions', x)
        return x


class PixelPoolModel(BaseModel):
    def __init__(self):
        """"""
        super().__init__()
        self.conv1 = SiConv2d(1, 16, 29, 7)
        self.pool1 = ScalePool(mode='pixel')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, 26, 7)
        self.pool2 = ScalePool(mode='pixel')
        self.act2 = nn.ReLU()
        self.global_pool = nn.MaxPool2d((52, 52))
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act1(x)
        self.save_trace('stage1', x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)
        self.save_trace('stage2', x)
        x = self.global_pool(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('prediction', x)
        return x


class SlicePoolModel(BaseModel):
    def __init__(self):
        """"""
        super().__init__()
        self.conv1 = SiConv2d(1, 16, 29, 7)
        self.pool1 = ScalePool(mode='slice')
        self.act1 = nn.ReLU()
        self.conv2 = SiConv2d(16, 32, 26, 7)
        self.pool2 = ScalePool(mode='slice')
        self.act2 = nn.ReLU()
        self.global_pool = nn.MaxPool2d((52, 52))
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act1(x)
        self.save_trace('stage1', x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)
        self.save_trace('stage2', x)
        x = self.global_pool(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('predictions', x)
        return x


class Conv3dModel(BaseModel):
    def __init__(self):
        """"""
        super().__init__()
        self.conv1 = SiConv2d(1, 16, 29, 7)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(16, 32, (5, 5, 5))
        self.act2 = nn.ReLU()
        self.global_pool = nn.MaxPool3d((25, 54, 54))
        self.lin = nn.Linear(32, 36)

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)
        self.save_trace('features', x)
        x = self.lin(x)
        self.save_trace('predictions', x)
        return x
