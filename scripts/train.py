""" Training script.

Command-line interface for training models.

This supports the following parameters:

| Name          | Values                                | Description                                     |
| ============= | ===================================== | =============================================== |
| model         | standard,pixel_pool,slice_pool,conv3d | The model type that should be trained           |
| evaluation    | 1,2,3,4,5                             | The evaluation scheme that should be used       |
| interpolation | nearest,bilinear,bicubic,area         | The interpolation technique that should be used |
| seed          | (integer)                             | The seed used for random initialization         |
"""
import argparse
import os.path
import random
import time

import mlflow
import numpy as np
import torch

from siconvnet.data import EmojiDataModule
from siconvnet.models import StandardModel, PixelPoolModel, SlicePoolModel, Conv3dModel, EnsembleModel, SpatialTransformModel


class Metrics:
    """ Metrics class for accumulating epoch loss and accuracy. """
    def __init__(self):
        self.loss = 0.0
        self.correct_count = 0
        self.total_count = 0

    def compute(self, loss, prediction_batch, label_batch):
        assert prediction_batch.shape[0] == label_batch.shape[0]
        num_samples = label_batch.shape[0]
        # Use .item() to convert Tensors to primitive types
        self.loss += loss.item() * num_samples
        self.correct_count += (prediction_batch.argmax(dim=1) == label_batch).sum().item()
        self.total_count += num_samples

    def get_epoch_loss(self):
        return self.loss / self.total_count

    def get_epoch_accuracy(self):
        return 100 * self.correct_count / self.total_count


class Timer:
    """ Timer class to record training times (in seconds). """
    def __init__(self):
        self.start_time = None
        self.times = []

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        assert self.start_time is not None, 'You need to call start before end'
        self.times.append(time.perf_counter() - self.start_time)

    def get_last_time(self):
        return self.times[-1]

    def get_average_time(self):
        return sum(self.times) / len(self.times)


class Record:
    """ Record class that tracks the accuracy, aborting after 10 non-improvements and keeping the best model. """
    def __init__(self, patience=10):
        self.patience = patience
        self.best_accuracy = -1.0
        self.no_improvement_count = 0
        # Create directory and path for PyTorch model file
        run_id = mlflow.active_run().info.run_id
        os.makedirs(os.path.join('temp', run_id), exist_ok=True)
        self.model_path = os.path.join('temp', run_id, 'model.pt')
        self.prediction_path = os.path.join('temp', run_id, 'prediction.npy')

    def track(self, net, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.no_improvement_count = 0
            torch.save(net.state_dict(), self.model_path)
        else:
            self.no_improvement_count += 1
        return self.no_improvement_count >= self.patience

    def get_best_accuracy(self):
        return self.best_accuracy

    def get_model_path(self):
        return self.model_path

    def get_prediction_path(self):
        return self.prediction_path


def seed_everything(seed):
    """ Seed Python, NumPy and PyTorch random modules. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(net, data, lr):
    """ Train a network with training and validation data. """
    # Find the appropriate device (either GPU or CPU depending on availability)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    data.to(device)
    # Define loss criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Loop over maximum number of epochs
    timer = Timer()
    record = Record()
    for epoch in range(200):
        # Loop over each batch
        metrics = Metrics()
        timer.start()
        net.train()
        for image_batch, label_batch in data.train_loader():
            optimizer.zero_grad()
            prediction_batch = net(image_batch)
            loss = criterion(prediction_batch, label_batch)
            loss.backward()
            optimizer.step()
            metrics.compute(loss, prediction_batch, label_batch)
        timer.end()
        # Log training loss and accuracy, plus time
        mlflow.log_metric('train_loss', metrics.get_epoch_loss(), epoch + 1)
        mlflow.log_metric('train_acc', metrics.get_epoch_accuracy(), epoch + 1)
        mlflow.log_metric('train_time', timer.get_last_time(), epoch + 1)
        # Compute validation loss and accuracy
        metrics = Metrics()
        with torch.no_grad():
            net.eval()
            for image_batch, label_batch in data.valid_loader():
                prediction_batch = net(image_batch)
                loss = criterion(prediction_batch, label_batch)
                metrics.compute(loss, prediction_batch, label_batch)
        # Log validation loss and accuracy
        mlflow.log_metric('val_loss', metrics.get_epoch_loss(), epoch + 1)
        mlflow.log_metric('val_acc', metrics.get_epoch_accuracy(), epoch + 1)
        if record.track(net, metrics.get_epoch_accuracy()):
            break  # Stop early once tracker indicates so
    mlflow.log_metric('avg_time', timer.get_average_time())
    mlflow.log_metric('best_acc', record.get_best_accuracy())
    mlflow.log_artifact(record.get_model_path())
    # Reload best parameters
    net.load_state_dict(torch.load(record.get_model_path()))
    # Compute testing loss and accuracy
    metrics = Metrics()
    with torch.no_grad():
        prediction_all = []
        net.eval()
        for image_batch, label_batch in data.test_loader():
            prediction_batch = net(image_batch)
            loss = criterion(prediction_batch, label_batch)
            metrics.compute(loss, prediction_batch, label_batch)
            prediction_all.append(prediction_batch.argmax(dim=1).detach().cpu().numpy())
        np.save(record.get_prediction_path(), np.concatenate(prediction_all))
    mlflow.log_artifact(record.get_prediction_path())
    # Log validation loss and accuracy
    mlflow.log_metric('test_loss', metrics.get_epoch_loss())
    mlflow.log_metric('test_acc', metrics.get_epoch_accuracy())


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for training models.')
    parser.add_argument('--model', help='The model type that should be trained',
                        choices=['standard', 'pixel_pool', 'slice_pool', 'conv3d', 'ensemble', 'spatial_transform'],
                        required=True)
    parser.add_argument('--evaluation', help='The evaluation scheme that should be used',
                        type=int, choices=[1, 2, 3, 4, 5], required=True)
    parser.add_argument('--kernel-size', help='The height and the width of the kernel that should be used',
                        type=int, choices=[3, 7, 11, 15], required=True)
    parser.add_argument('--interpolation', help='The interpolation technique that should be used',
                        choices=['nearest', 'bilinear', 'bicubic', 'area'], required=True)
    parser.add_argument('--lr', help='The learning rate used by the Adam optimizer',
                        type=float, choices=[1e-2, 1e-3], required=True)
    parser.add_argument('--seed', help='The seed used for random initialization', type=int, required=True)
    args = parser.parse_args()
    # Log parameters
    mlflow.log_param('model', args.model)
    mlflow.log_param('evaluation', args.evaluation)
    mlflow.log_param('kernel_size', args.kernel_size)
    mlflow.log_param('interpolation', args.interpolation)
    mlflow.log_param('lr', args.lr)
    mlflow.log_param('seed', args.seed)
    # Set seed for reproducibility
    seed_everything(args.seed)
    # Load network and data modules
    network_map = {
        'standard': StandardModel,
        'pixel_pool': PixelPoolModel,
        'slice_pool': SlicePoolModel,
        'conv3d': Conv3dModel,
        'ensemble': EnsembleModel,
        'spatial_transform': SpatialTransformModel
    }
    network = network_map[args.model](kernel_size=args.kernel_size, interpolation=args.interpolation)
    data = EmojiDataModule(batch_size=16, evaluation=args.evaluation)
    # Train the network with the given data
    train(network, data, lr=args.lr)


if __name__ == '__main__':
    main()
