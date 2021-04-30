""" Training script.

Command-line interface for training models.

optional arguments:
  -h, --help            show this help message and exit
  --model {standard,pixel_pool,slice_pool,conv3d}
                        The model type that should be trained
  --evaluation {1,2,3}  The evaluation scheme that should be used
  --seed SEED           The seed used for random initialization
"""
import argparse
import os.path
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything

from siconvnet.models import StandardModel, PixelPoolModel, SlicePoolModel, Conv3dModel
from siconvnet.data import EmojiDataModule


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for training models.')
    parser.add_argument('--model', help='The model type that should be trained',
                        choices=['standard', 'pixel_pool', 'slice_pool', 'conv3d'], required=True)
    parser.add_argument('--evaluation', help='The evaluation scheme that should be used',
                        type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--seed', help='The seed used for random initialization', type=int, required=True)
    args = parser.parse_args()
    # Set seed for reproducibility
    seed_everything(args.seed)
    # Load network and data modules
    network_map = {
        'standard': StandardModel,
        'pixel_pool': PixelPoolModel,
        'slice_pool': SlicePoolModel,
        'conv3d': Conv3dModel
    }
    network = network_map[args.model]()
    data = EmojiDataModule(batch_size=16, evaluation=args.evaluation)
    # Set up callbacks
    path = os.path.join('sessions_seed{}'.format(args.seed), '{}_eval{}'.format(args.model, args.evaluation))
    logger = CSVLogger(path, name='logs')
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=path, filename='model-{epoch}', monitor='val_acc_epoch', mode='max')
    early_stop = pl.callbacks.EarlyStopping(monitor='val_acc_epoch', mode='max', patience=10)
    # Start training
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint, early_stop], gpus=1, max_epochs=500, deterministic=True)
    start = time.perf_counter()
    trainer.fit(network, data)
    end = time.perf_counter()
    total, average = end - start, (end - start) / trainer.current_epoch
    with open(os.path.join(path, 'timer.txt'), 'w') as file:
        file.write('{} seconds / {} epochs = {}'.format(total, trainer.current_epoch, average))


if __name__ == '__main__':
    main()
