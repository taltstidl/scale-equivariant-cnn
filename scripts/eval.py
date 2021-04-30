""" Evaluation script.

There are multiple targets for evaluation:
* Classification accuracy on the testing data
* Training time for epoch
* Statistics of intra-class distances
* Statistics of activations
"""
import argparse
import os.path
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch

from siconvnet.models import StandardModel, PixelPoolModel, SlicePoolModel, Conv3dModel
from siconvnet.data import EmojiDataModule


# Load the device, depending on whether GPU is available or not
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_network(path, model, evaluation):
    """ Load checkpoint of the given session and prepare for evaluation. """
    # Path containing the session data
    session_data_path = os.path.join(path, '{}_eval{}'.format(model, evaluation))
    # Path containing the model checkpoint
    checkpoint_path = os.path.join(session_data_path,
                                   [f for f in os.listdir(session_data_path) if f.startswith('model')][0])
    # Load network with checkpoint
    network_map = {
        'standard': StandardModel,
        'pixel_pool': PixelPoolModel,
        'slice_pool': SlicePoolModel,
        'conv3d': Conv3dModel
    }
    network = network_map[model].load_from_checkpoint(checkpoint_path)
    network.eval()  # Change network into evaluation mode
    network.to(device)  # Move network to correct device
    return network


def make_predictions(network, loader):
    """ Computes predictions of the network on the given data. """
    with torch.no_grad():  # Disable gradient computation
        labels, predictions = [], []
        for images_batch, labels_batch in loader:
            predictions_batch = network(images_batch.to(device))
            # Copy to prevent memory leak (see https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189)
            labels.append(deepcopy(labels_batch).detach().cpu().numpy())
            predictions.append(predictions_batch.detach().cpu().numpy())
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions).argmax(axis=1)
        return labels, predictions


def make_activations(network, loader):
    """ Computes activations of the network on the given data. """
    with torch.no_grad():  # Disable gradient computation
        network.enable_tracing()
        for images_batch, _ in loader:
            _ = network(images_batch.to(device))
        return network.get_traces()


def perform_evaluation(path, model, evaluation):
    network = load_network(path, model, evaluation)
    data = EmojiDataModule(batch_size=16, evaluation=evaluation)
    data.setup()
    # Compute accuracy
    labels, predictions = make_predictions(network, data.test_dataloader())
    accuracy = 100 * np.sum(labels == predictions) / labels.shape[0]
    print('Accuracy of {} model using eval scheme {} is {}.'.format(model, evaluation, accuracy))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for evaluation models.')
    parser.add_argument('--sessions', help='The directory containing the session files',
                        type=str, required=True)
    args = parser.parse_args()
    # Loop over all models
    for model in ['pixel_pool', 'slice_pool', 'conv3d']:
        for evaluation in [1, 2, 3]:
            perform_evaluation(args.sessions, model, evaluation)


if __name__ == '__main__':
    main()
