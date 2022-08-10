""" Evaluation script.

Command-line interface for evaluating models.

This supports the following parameters:
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from data.dataset import STIRDataset
from siconvnet.metrics import scale_generalization, scale_equivariance, scale_index_correlation
from siconvnet.models import StandardModel, PixelPoolModel, SlicePoolModel, EnergyPoolModel, Conv3dModel, \
    EnsembleModel, SpatialTransformModel


def load_model(state_path, model, num_channels, num_classes):
    # Load correct model class
    model_map = {
        'standard': StandardModel,
        'pixel_pool': PixelPoolModel,
        'slice_pool': SlicePoolModel,
        'energy_pool': EnergyPoolModel,
        'conv3d': Conv3dModel,
        'ensemble': EnsembleModel,
        'spatial_transform': SpatialTransformModel
    }
    model = model_map[model](kernel_size=7, interpolation='bicubic', num_channels=num_channels, num_classes=num_classes)
    # Load state, incl. model weights
    model.load_state_dict(torch.load(state_path))
    return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for training models.')
    parser.add_argument('--runs', help='The path to the exported runs.csv', type=str, required=True)
    parser.add_argument('--models', help='The path to the mlruns folder with all models', type=str, required=True)
    args = parser.parse_args()
    # Find the appropriate device (either GPU or CPU depending on availability)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the experiment data from all runs
    all_runs = pd.read_csv(args.runs)
    all_runs = all_runs[all_runs['status'] == 'FINISHED']  # Filter for finished runs
    generalization_metrics = []
    for data in ['emoji', 'trafficsign']:  # TODO: add all datasets!
        dataset = STIRDataset(path='{}.npz'.format(data))
        runs = all_runs[all_runs['params.data'] == data]  # Filter for dataset
        runs = runs[runs['params.lr'] == 1e-3]  # Filter for learning rate
        for _, run in runs.iterrows():
            model_key, data_key, evaluation = run['params.model'], run['params.data'], run['params.evaluation']
            metadata = [model_key, data_key, evaluation]
            # Create new folder for evaluation results
            eval_path = os.path.join('eval', run['run_id'])
            os.makedirs(eval_path, exist_ok=True)
            # Load model, ensuring it's on the correct device and in evaluation mode
            state_path = os.path.join(args.models, run['run_id'], 'artifacts', 'model.pt')
            model = load_model(state_path, model_key, dataset.num_channels, dataset.num_classes)
            model.to(device)
            model.eval()
            # Compute the different metrics
            # generalization_metrics.append(metadata + scale_generalization(model, dataset, device))
            # Compute scale to index correlation
            accepted_model_keys = ['pixel_pool', 'slice_pool', 'energy_pool']
            if model_key in accepted_model_keys:
                np.savez_compressed(os.path.join(eval_path, 'indices.npz'),
                                    **scale_index_correlation(model, dataset, device))
    # Store results for scale generalization
    # generalization_columns = ['model', 'data', 'eval'] + ['s{}'.format(i) for i in range(17, 65)]
    # generalization_df = pd.DataFrame.from_records(generalization_metrics, columns=generalization_columns)
    # generalization_df.to_csv('generalization.csv')


if __name__ == '__main__':
    main()
