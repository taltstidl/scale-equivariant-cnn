"""
This module implements the metrics discussed in the paper. These are:

* Generalization across scales
* Equivariance of feature maps after convolution
* Correlation between scale and activation index
"""
import numpy as np
import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _crop_image(activation, scale, position):
    # Crop the proper region of the feature map based on ground truth
    padding = 64 - activation.shape[-1]
    px, py, s = position[0], position[1], scale - padding
    return activation[:, py:py + s, px:px + s]


def _interpolate_image(activation, target_scale):
    # Resize and interpolate feature map to common size
    return F.interpolate(activation[None, :], size=(target_scale, target_scale), mode='bicubic')[0]


def scale_generalization(model, dataset, device):
    # Retrieve images and labels from dataset in a data loader
    images, labels = dataset.to_numpy(split='test', scales=range(17, 65), shuffle=False)
    tensor_images = torch.tensor(images, dtype=torch.float).to(device)
    data_loader = DataLoader(tensor_images, batch_size=16, shuffle=False)
    # Compute predictions across all scales
    predictions = []
    with torch.no_grad():
        for image_batch in data_loader:
            prediction_batch = model(image_batch)
            predictions.append(prediction_batch.argmax(dim=1).detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    # Retrieve known scales from dataset
    scales, _ = dataset.get_latents(split='test', scales=range(17, 65), shuffle=False)
    # Compute accuracy for each scale
    scale_accuracies = []
    for scale in range(17, 65):
        scale_predictions, scale_labels = predictions[scales == scale], labels[scales == scale]
        scale_accuracies.append((scale_predictions == scale_labels).sum() / scale_labels.shape[0])
    return scale_accuracies


def scale_equivariance(model, dataset, device):
    # Retrieve images from dataset
    images = dataset.images[2]
    if len(images.shape) == 5:
        images = np.expand_dims(images, axis=-1)  # Single channel
    num_classes, num_instances = images.shape[1:3]
    # Retrieve known scales and positions from dataset
    scales, positions = dataset.scales[2], dataset.positions[2]
    # Go over each individual image
    scale_errors = []
    for ci in range(num_classes):
        for ii in range(num_instances):
            # Retrieve images, scales and positions (should be 5 each)
            instance_images = images[:33:8, ci, ii].transpose((0, 3, 1, 2))
            instance_scales = scales[:33:8, ci, ii]
            instance_positions = positions[:33:8, ci, ii]
            with torch.no_grad():
                # Gather feature maps and vectors
                model.enable_tracing()
                _ = model(torch.tensor(instance_images, dtype=torch.float).to(device))
                traces = model.get_traces()
                model.disable_tracing()
                # Feature map tensors are of shape (num_scales, num_filters, size, size)
                stage1, stage2 = traces['stage1'], traces['stage2']
                # Feature vector tensors are of shape (num_scales, size)
                features, predictions = traces['features'], traces['predictions']
                # Crop feature map tensors such that only object remains
                stage1 = [_crop_image(stage1[i], instance_scales[i], instance_positions[i]) for i in range(5)]
                stage2 = [_crop_image(stage2[i], instance_scales[i], instance_positions[i]) for i in range(5)]
                # Compute pair-wise L2 norm between feature map/vector tensors at different scales
                for i, scale_ref in enumerate(instance_scales):
                    for j, scale in enumerate(instance_scales):
                        if scale == scale_ref:
                            continue
                        # Compute errors for each filter of the first stage
                        ref, interp = stage1[i], _interpolate_image(stage1[j], scale_ref - 6)
                        error = LA.vector_norm(ref - interp)**2 / LA.vector_norm(ref)**2
                        scale_errors.append([ci, ii, 'stage1', scale, scale_ref, error.cpu().item()])
                        # Compute errors for each filter of the second stage
                        ref, interp = stage2[i], _interpolate_image(stage2[j], scale_ref - 12)
                        error = LA.vector_norm(ref - interp)**2 / LA.vector_norm(ref)**2
                        scale_errors.append([ci, ii, 'stage2', scale, scale_ref, error.cpu().item()])
                        # Compute errors for features
                        error = LA.vector_norm(features[i] - features[j])**2 / LA.vector_norm(features[i])**2
                        scale_errors.append([ci, ii, 'features', scale, scale_ref, error.cpu().item()])
                        # Compute errors for predictions
                        error = LA.vector_norm(predictions[i] - predictions[j])**2 / LA.vector_norm(predictions[i])**2
                        scale_errors.append([ci, ii, 'predictions', scale, scale_ref, error.cpu().item()])
    return scale_errors


def scale_index_correlation():
    pass
