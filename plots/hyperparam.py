import argparse
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def select(hparams, eval, model, data, interp, k, lr):
    """ Filter data table by model, dataset, interpolation and kernel size. """
    rows = hparams[(hparams['params.model'] == model) & (hparams['params.data'] == data) &
                   (hparams['params.interpolation'] == interp) & (hparams['params.kernel_size'] == k) &
                   (hparams['params.lr'] == lr) & (hparams['params.evaluation'] == eval)]
    metrics = rows['metrics.test_acc'].to_numpy()
    metrics = metrics[~np.isnan(metrics)]
    if metrics.shape[0] != 50:
        args = (eval, model, data, interp, k, metrics.shape[0])
        warnings.warn('Missing runs for eval=`{}` model=`{}` data=`{}` interp=`{}` and k=`{}` ({})!'.format(*args))
    return metrics


def color_box(boxplots, i, color):
    """ Color a specific box in a boxplot. """
    box = boxplots['boxes'][i]
    box.set_color(color)
    box.set(fill=False)
    whisker1, whisker2 = boxplots['whiskers'][2 * i], boxplots['whiskers'][2 * i + 1]
    whisker1.set_color(color)
    whisker2.set_color(color)
    cap1, cap2 = boxplots['caps'][2 * i], boxplots['caps'][2 * i + 1]
    cap1.set_color(color)
    cap2.set_color(color)


def plot_for_paper(runs, data, lr):
    """ Plot results for paper (with interp=bicubic and kernel=7x7). """
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(7.5, 2.0))
    for eval in range(3):  # Loop over evaluation scenarios
        axis = axs[eval]
        if eval == 0:
            axis.set_ylabel('Testing Accuracy [%]', fontname='Corbel')
        eval += 1  # Evaluation is 1-based, not 0-based
        axis.set_title('Evaluation {}'.format(eval), fontname='Corbel', fontsize=10)
        boxes = [
            select(runs, eval=eval, data=data, model='standard', interp='bicubic', k=7, lr=lr),
            select(runs, eval=eval, data=data, model='pixel_pool', interp='bicubic', k=7, lr=lr),
            select(runs, eval=eval, data=data, model='slice_pool', interp='bicubic', k=7, lr=lr),
            select(runs, eval=eval, data=data, model='conv3d', interp='bicubic', k=7, lr=lr),
            select(runs, eval=eval, data=data, model='ensemble', interp='bicubic', k=7, lr=lr),
            select(runs, eval=eval, data=data, model='spatial_transform', interp='bicubic', k=7, lr=lr),
        ]
        labels = ['standard', 'pixel_pool', 'slice_pool', 'conv3d', 'ensemble', 'spatial_trans']
        colors = ['#00A3E0', '#43B02A', '#FFB81C', '#C8102E', '#779FB5', '#002F6C']
        boxplots = axis.boxplot(boxes, showfliers=False, patch_artist=True)
        for i in range(6):
            num_runs = len(boxes[i])  # number of repetitions, should be 50, may be smaller due to HPC kill
            pos = np.full(shape=(num_runs,), fill_value=i + 1) + np.random.normal(scale=0.1, size=(num_runs,))
            axis.scatter(pos, boxes[i], s=3, c=colors[i] + '55')
            color_box(boxplots, i, colors[i])
        axis.xaxis.set_visible(False)
        axis.yaxis.grid(zorder=0, c='#eeee')
        kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
        if eval == 1:
            axis.legend([boxplots['boxes'][0], boxplots['boxes'][1]], labels[:2], **kwargs)
        if eval == 2:
            axis.legend([boxplots['boxes'][2], boxplots['boxes'][3]], labels[2:4], **kwargs)
        if eval == 3:
            axis.legend([boxplots['boxes'][4], boxplots['boxes'][5]], labels[4:], **kwargs)
    plt.tight_layout()
    #plt.show()
    plt.savefig('hparams_{}.pdf'.format(data), bbox_inches='tight')


def plot_for_supplemental():
    pass  # TODO: Implement me


def main():
    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description='Command-line interface for generating hyperparam plots.')
    # parser.add_argument('--evaluation', help='The evaluation scenario to plot',
    #                     type=int, choices=[1, 2, 3, 4, 5], required=True)
    # parser.add_argument('--kernel-size', help='The kernel size to plot',
    #                     type=int, choices=[3, 7, 11, 15], required=True)
    # args = parser.parse_args()
    # Write plot to file
    runs = pd.read_csv('../scripts/runs.csv')
    plot_for_paper(runs, data='emoji', lr=1e-2)
    plot_for_paper(runs, data='mnist', lr=1e-2)
    plot_for_paper(runs, data='trafficsign', lr=1e-3)


if __name__ == '__main__':
    main()
