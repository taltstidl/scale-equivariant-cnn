import argparse
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def select(hparams, model, interp, k):
    """ Filter data table by model, interpolation and kernel size. """
    rows = hparams[(hparams['params.model'] == model) & (hparams['params.interpolation'] == interp) &
                   (hparams['params.kernel_size'] == k)]
    metrics = rows['metrics.test_acc'].to_numpy()
    metrics = metrics[~np.isnan(metrics)]
    if metrics.shape[0] != 50:
        found = metrics.shape[0]
        warnings.warn('Missing runs for model=`{}` interp=`{}` and k=`{}` ({})!'.format(model, interp, k, found))
    return metrics


def filter(hparams, k):
    """ Filter data table and return array of binned runs suitable for plotting. """
    return [
        select(hparams, model='standard', interp='nearest', k=k),
        select(hparams, model='pixel_pool', interp='nearest', k=k),
        select(hparams, model='pixel_pool', interp='bilinear', k=k),
        select(hparams, model='pixel_pool', interp='bicubic', k=k),
        select(hparams, model='pixel_pool', interp='area', k=k),
        select(hparams, model='slice_pool', interp='nearest', k=k),
        select(hparams, model='slice_pool', interp='bilinear', k=k),
        select(hparams, model='slice_pool', interp='bicubic', k=k),
        select(hparams, model='slice_pool', interp='area', k=k),
        select(hparams, model='conv3d', interp='nearest', k=k),
        select(hparams, model='conv3d', interp='bilinear', k=k),
        select(hparams, model='conv3d', interp='bicubic', k=k),
        select(hparams, model='conv3d', interp='area', k=k)
    ]


def configure_boxplot(axis, boxplot):
    colors = ["#002F6C", "#779FB5", "#FFB81C", "#00A3E0", "#43B02A", "#C8102E"]
    colors = ['black'] + 3 * colors[2:]
    hatches = [''] + 3 * ['oo', '//', '\\\\', 'xx']
    for i in range(13):
        box = boxplot['boxes'][i]
        box.set_color(colors[i])
        box.set(hatch=hatches[i], fill=False)
        whisker1, whisker2 = boxplot['whiskers'][2 * i], boxplot['whiskers'][2 * i + 1]
        whisker1.set_color(colors[i])
        whisker2.set_color(colors[i])
        cap1, cap2 = boxplot['caps'][2 * i], boxplot['caps'][2 * i + 1]
        cap1.set_color(colors[i])
        cap2.set_color(colors[i])
    for median in boxplot['medians']:
        median.set_color('#002F6C')
    for mean in boxplot['means']:
        mean.set_markerfacecolor('#002F6C')
        mean.set_markeredgecolor('#002F6C')
    boxes = boxplot['boxes']
    axis.legend([boxes[1], boxes[2], boxes[3], boxes[4]], ['nearest', 'bilinear', 'bicubic', 'area'], ncol=2)
    axis.set_xlabel('Model Type')
    axis.set_ylabel('Testing Accuracy [%]')
    axis.set_xticks([0, 3.5, 8.5, 13.5])
    axis.set_xticklabels(['standard', 'pixel_pool', 'slice_pool', 'conv3d'])


def plot_results(hparams, evaluation, kernel_size):
    hparams = hparams[hparams['params.evaluation'] == evaluation]
    positions = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]
    fig, axis = plt.subplots(nrows=1, ncols=1, sharey=True)
    # axis = axs[0]
    # axis.set_title('Kernel Size 3x3')
    # axis.boxplot(filter(hparams, k=3), showmeans=True, positions=positions)
    # axis = axs[1]
    axis.set_title('Evaluation {0} - Kernel Size {1}x{1}'.format(evaluation, kernel_size))
    boxplot = axis.boxplot(filter(hparams, k=kernel_size), showmeans=True, positions=positions, patch_artist=True)
    configure_boxplot(axis, boxplot)
    # axis = axs[2]
    # axis.set_title('Kernel Size 11x11')
    # axis.boxplot(filter(hparams, k=11), showmeans=True, positions=positions)
    # axis = axs[3]
    # axis.set_title('Kernel Size 15x15')
    # axis.boxplot(filter(hparams, k=15), showmeans=True, positions=positions)
    plt.tight_layout()
    plt.savefig('eval{}_kernel{}.png'.format(evaluation, kernel_size), bbox_inches='tight')


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for generating hyperparam plots.')
    parser.add_argument('--evaluation', help='The evaluation scenario to plot',
                        type=int, choices=[1, 2, 3, 4, 5], required=True)
    parser.add_argument('--kernel-size', help='The kernel size to plot',
                        type=int, choices=[3, 7, 11, 15], required=True)
    args = parser.parse_args()
    # Write plot to file
    hparams = pd.read_csv('../hyperparams.csv')
    for e in [1, 2, 3, 4, 5]:
        for k in [3, 7, 11, 15]:
            plot_results(hparams, e, k)


if __name__ == '__main__':
    main()
