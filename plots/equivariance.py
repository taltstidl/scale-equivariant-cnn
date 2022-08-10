import argparse
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def select(hparams, eval, model, data, node):
    """ Filter data table by model, dataset, interpolation and kernel size. """
    rows = hparams[(hparams['data'] == data) & (hparams['model'] == model) &
                   (hparams['eval'] == eval) & (hparams['node'] == node)]
    metrics = rows['error'].to_numpy()
    metrics = np.log10(metrics)
    return metrics


def plot_for_paper(results, data='emoji', eval=2):
    """ Plot results for paper (with interp=bicubic and kernel=7x7). """
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(7.5, 2.5))
    model_colors = ['#00A3E0', '#43B02A', '#FFB81C']
    for i, model in enumerate(['standard', 'pixel_pool', 'slice_pool']):  # Loop over models
        axis = axs[i]
        axis.set_title(model, fontname='Corbel', fontsize=10)
        boxes = [
            select(results, eval=eval, data=data, model=model, node='stage1'),
            select(results, eval=eval, data=data, model=model, node='stage2'),
            select(results, eval=eval, data=data, model=model, node='features'),
            select(results, eval=eval, data=data, model=model, node='predictions'),
        ]
        violinplots = axis.violinplot(boxes)
        for body in violinplots['bodies']:
            body.set_color(model_colors[i] + '55')
        violinplots['cmins'].set_color(model_colors[i])
        violinplots['cmaxes'].set_color(model_colors[i])
        violinplots['cbars'].set_color(model_colors[i])
        axis.xaxis.set_ticks(range(1, 5))
        axis.xaxis.set_ticklabels(['Conv 1', 'Conv 2', 'Pool', 'Lin'], fontname='Corbel', fontsize=10)
        axis.yaxis.set_visible(False)
    for i in range(3):  # Create "fake" logarithmic axes
        log_axis = axs[i].twinx()
        log_axis.set_yscale('log')
        log_axis.set_ylim(10**np.array(axs[i].get_ylim()))
        if i == 0:  # Left-most axis includes y-axis label
            log_axis.set_ylabel('Equivariance Error', fontname='Corbel')
            log_axis.yaxis.set_label_position('left')
        else:  # Other axis don't include tick labels
            log_axis.set_yticklabels([])
        log_axis.yaxis.tick_left()
        log_axis.yaxis.grid(zorder=0, c='#eeee')
    plt.tight_layout()
    #plt.show()
    plt.savefig('equivariance_{}.pdf'.format(data), bbox_inches='tight')


def main():
    # Write plot to file
    runs = pd.read_csv('equivariance.csv')
    plot_for_paper(runs, data='emoji')


if __name__ == '__main__':
    main()
