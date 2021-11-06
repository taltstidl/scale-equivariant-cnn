import argparse

import mlflow
import torch

from scripts.train import Metrics, Record, seed_everything
from trafficsign.data import TrafficSignDataModule
from trafficsign.models import StandardModel, ScaleEquivModel, SpatialTransformerModel, EnsembleModel


def train(net, data, lr):
    # Find the appropriate device (either GPU or CPU depending on availability)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    # Define loss criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Loop over maximum number of epochs
    record = Record()
    for epoch in range(200):
        # Loop over each batch
        metrics = Metrics()
        net.train()
        for image_batch, label_batch in data.train_loader():
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction_batch = net(image_batch)
            loss = criterion(prediction_batch, label_batch)
            loss.backward()
            optimizer.step()
            metrics.compute(loss, prediction_batch, label_batch)
        # Log training loss and accuracy, plus time
        mlflow.log_metric('train_loss', metrics.get_epoch_loss(), epoch + 1)
        mlflow.log_metric('train_acc', metrics.get_epoch_accuracy(), epoch + 1)
        # Compute validation loss and accuracy
        metrics = Metrics()
        with torch.no_grad():
            net.eval()
            for image_batch, label_batch in data.valid_loader():
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                prediction_batch = net(image_batch)
                loss = criterion(prediction_batch, label_batch)
                metrics.compute(loss, prediction_batch, label_batch)
        # Log validation loss and accuracy
        mlflow.log_metric('val_loss', metrics.get_epoch_loss(), epoch + 1)
        mlflow.log_metric('val_acc', metrics.get_epoch_accuracy(), epoch + 1)
        if record.track(net, metrics.get_epoch_accuracy()):
            break  # Stop early once tracker indicates so
    mlflow.log_metric('best_acc', record.get_best_accuracy())
    mlflow.log_artifact(record.get_model_path())
    # Reload best parameters
    net.load_state_dict(torch.load(record.get_model_path()))
    # Compute testing loss and accuracy
    metrics = Metrics()
    with torch.no_grad():
        net.eval()
        for image_batch, label_batch in data.test_loader():
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            prediction_batch = net(image_batch)
            loss = criterion(prediction_batch, label_batch)
            metrics.compute(loss, prediction_batch, label_batch)
    # Log validation loss and accuracy
    mlflow.log_metric('test_loss', metrics.get_epoch_loss())
    mlflow.log_metric('test_acc', metrics.get_epoch_accuracy())


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for training models.')
    parser.add_argument('--model', help='The model type that should be trained',
                        choices=['standard', 'scale_equiv', 'spatial_transformer', 'ensemble'], required=True)
    parser.add_argument('--evaluation', help='The evaluation scheme that should be used',
                        type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--lr', help='The learning rate used by the Adam optimizer',
                        type=float, choices=[1e-2, 1e-3], required=True)
    parser.add_argument('--seed', help='The seed used for random initialization', type=int, required=True)
    args = parser.parse_args()
    # Log parameters
    mlflow.log_param('model', args.model)
    mlflow.log_param('evaluation', args.evaluation)
    mlflow.log_param('lr', args.lr)
    mlflow.log_param('seed', args.seed)
    # Set seed for reproducibility
    seed_everything(args.seed)
    # Load network and data modules
    model_map = {
        'standard': StandardModel,
        'scale_equiv': ScaleEquivModel,
        'spatial_transformer': SpatialTransformerModel,
        'ensemble': EnsembleModel,
    }
    network = model_map[args.model]()
    data = TrafficSignDataModule(batch_size=32, evaluation=args.evaluation)
    # Train the network with the given data
    train(network, data, lr=args.lr)


if __name__ == '__main__':
    main()
