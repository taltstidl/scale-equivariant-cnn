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
    EnsembleModel, SpatialTransformModel, XuModel, KanazawaModel, HermiteModel, DiscoModel


def load_model(state_path, model, num_channels, num_classes):
    # Load correct model class
    model_map = {
        'standard': StandardModel,
        'pixel_pool': PixelPoolModel,
        'slice_pool': SlicePoolModel,
        'energy_pool': EnergyPoolModel,
        'conv3d': Conv3dModel,
        'ensemble': EnsembleModel,
        'spatial_transform': SpatialTransformModel,
        'xu': XuModel,
        'kanazawa': KanazawaModel,
        'hermite': HermiteModel,
        'disco': DiscoModel
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
    parser.add_argument('--data', choices=['emoji', 'mnist', 'trafficsign', 'aerial'], required=True)
    parser.add_argument('--generalization', action='store_true')
    parser.add_argument('--equivariance', action='store_true')
    parser.add_argument('--index-correlation', action='store_true')
    args = parser.parse_args()
    # Find the appropriate device (either GPU or CPU depending on availability)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the experiment data from all runs
    runs = pd.read_csv(args.runs)
    runs = runs[runs['params.data'] == args.data]  # Filter for dataset
    # Load the dataset itself
    dataset = STIRDataset(path='{}.npz'.format(args.data))
    generalization_metrics = []
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
        # Compute scale generalization
        if args.generalization:
            generalization_metrics.append(metadata + scale_generalization(model, dataset, device))
        # Compute scale equivariance
        if args.equivariance:
            accepted_model_keys = ['standard', 'pixel_pool', 'slice_pool', 'energy_pool', 'kanazawa']
            accepted_data_keys = ['emoji', 'mnist', 'trafficsign']
            if model_key in accepted_model_keys and data_key in accepted_data_keys:
                np.savez_compressed(os.path.join(eval_path, 'errors.npz'),
                                    **scale_equivariance(model, dataset, device))
        # Compute scale to index correlation
        if args.index_correlation:
            accepted_run_ids = ['f60d2e453691424ea05bbbe7c5e17289', 'eb15992b30214d36b71b1c00cfd2524d']
            if run['run_id'] in accepted_run_ids:
                np.savez_compressed(os.path.join(eval_path, 'indices.npz'),
                                    **scale_index_correlation(model, dataset, device))
    if args.generalization:
        # Store results for scale generalization
        generalization_columns = ['model', 'data', 'eval'] + ['s{}'.format(i) for i in range(17, 65)]
        generalization_df = pd.DataFrame.from_records(generalization_metrics, columns=generalization_columns)
        generalization_df.to_csv('generalization_{}.csv'.format(args.data))


if __name__ == '__main__':
    main()
