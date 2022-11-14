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
    """ Saves performances computed per scale for later analysis.

    Parameters
    ----------
    model: torch.nn.Module
        Image classification network that should be analyzed.
    dataset: STIRDataset
        Dataset from which to pull images for classification.
    device: torch.Device
        Device on which computations take place, usually either CPU or GPU.
    """
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
    """ Saves equivariance of feature maps and vectors for later analysis.

    Equivariance is computed for pairs of images at different scales. Feature maps are resized to the reference scale
    using bicubic interpolation. Feature vectors are left untouched.

    Parameters
    ----------
    model: siconvnet.models.BaseModel
        Image classification network that should be analyzed. Must save traces for 'stage1', 'stage2', 'features'
        and 'predictions', for which equivariance is computed.
    dataset: STIRDataset
        Dataset from which to pull images for classification.
    device: torch.Device
        Device on which computations should take place, usually either CPU or GPU.
    """
    # Retrieve images from dataset
    images = dataset.images[2]
    if len(images.shape) == 5:
        images = np.expand_dims(images, axis=-1)  # Single channel
    num_classes, num_instances = images.shape[1:3]
    # Retrieve known scales and positions from dataset
    scales, positions = dataset.scales[2], dataset.positions[2]
    # Create empty arrays for errors
    errors_stage1 = np.empty(shape=(num_classes, num_instances, 6))
    errors_stage2 = np.empty(shape=(num_classes, num_instances, 6))
    # Go over each individual image
    for ci in range(num_classes):
        for ii in range(num_instances):
            # Retrieve images, scales and positions
            idx = [0, 15, 16, 31, 32, 47]  # Scales 64x64, 49x49, 48x48, 33x33, 32x32, and 17x17
            instance_images = images[idx, ci, ii].transpose((0, 3, 1, 2))
            instance_scales = scales[idx, ci, ii]
            instance_positions = positions[idx, ci, ii]
            with torch.no_grad():
                # Gather feature maps and vectors
                model.enable_tracing()
                _ = model(torch.tensor(instance_images, dtype=torch.float).to(device))
                traces = model.get_traces()
                model.disable_tracing()
                # Feature map tensors are of shape (num_scales, num_filters, size, size)
                stage1, stage2 = traces['stage1'], traces['stage2']
                # Crop feature map tensors such that only object remains
                stage1 = [_crop_image(stage1[i], instance_scales[i], instance_positions[i]) for i in range(6)]
                stage2 = [_crop_image(stage2[i], instance_scales[i], instance_positions[i]) for i in range(6)]
                # Compute pair-wise L2 norm between feature map/vector tensors at different scales
                for stage, errors in zip([stage1, stage2], [errors_stage1, errors_stage2]):
                    for i, (si, sj) in enumerate([(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)]):
                        # Compute errors for each filter of the first stage
                        ref, interp = stage[si], _interpolate_image(stage[sj], stage[si].shape[-1])
                        error = LA.vector_norm(ref - interp)**2 / LA.vector_norm(ref)**2
                        errors[ci, ii, i] = error.cpu().item()
    return {'stage1': errors_stage1, 'stage2': errors_stage2}


def scale_index_correlation(model, dataset, device):
    """ Saves pooling indices along the scale axis for later analysis.

    Parameters
    ----------
    model: torch.nn.Module
        Image classification network that should be analyzed. Must contain scale pooling layers pool1 and pool2, for
        which pooling indices are computed.
    dataset: STIRDataset
        Dataset from which to pull images for classification.
    device: torch.Device
        Device on which computations take place, usually either CPU or GPU.
    """
    # Retrieve images and labels from dataset in a data loader
    images, labels = dataset.to_numpy(split='test', scales=range(17, 65), shuffle=False)
    tensor_images = torch.tensor(images, dtype=torch.float).to(device)
    data_loader = DataLoader(tensor_images, batch_size=16, shuffle=False)
    # Compute predictions across all scales
    pool1_indices, pool2_indices = [], []
    with torch.no_grad():
        for image_batch in data_loader:
            _ = model(image_batch)
            pool1_index_batch = model.pool1.indices
            pool1_indices.append(pool1_index_batch.detach().cpu().numpy().astype(np.uint8))
            pool2_index_batch = model.pool2.indices
            pool2_indices.append(pool2_index_batch.detach().cpu().numpy().astype(np.uint8))
    pool1_indices = np.concatenate(pool1_indices)
    pool2_indices = np.concatenate(pool2_indices)
    return {'pool1': pool1_indices, 'pool2': pool2_indices}
