import argparse
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def select(results, eval, model, data):
    """ Filter data table by model, dataset, interpolation and kernel size. """
    rows = results[(results['data'] == model) & (results['model'] == data) &
                   (results['eval'] == eval)]
    acc_means = 100 * np.array([rows['s{}'.format(s)].mean() for s in range(17, 65)])
    acc_stds = 100 * np.array([rows['s{}'.format(s)].std() for s in range(17, 65)])
    if acc_means.shape[0] != 48 or acc_stds.shape[0] != 48:
        args = (eval, model, data, acc_means.shape[0], acc_stds.shape[0])
        warnings.warn('Missing runs for eval=`{}` model=`{}` and data=`{}` ({} | {})!'.format(*args))
    return acc_means, acc_stds


def plot_for_paper(results, data='emoji'):
    """ Plot results for paper (with interp=bicubic and kernel=7x7). """
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(7.5, 2.5))
    for eval in range(3):  # Loop over evaluation scenarios
        axis = axs[eval]
        axis.set_xlabel('Scale')
        if eval == 0:
            axis.set_ylabel('Testing Accuracy [%]', fontname='Corbel')
        axis.axvspan([17, 33, 49][eval], [32, 48, 64][eval], fc='#eeee', zorder=-1)
        eval += 1  # Evaluation is 1-based, not 0-based
        axis.set_title('Evaluation {}'.format(eval), fontname='Corbel', fontsize=10)
        lines = [
            select(results, eval=eval, data=data, model='standard'),
            select(results, eval=eval, data=data, model='pixel_pool'),
            select(results, eval=eval, data=data, model='slice_pool'),
            select(results, eval=eval, data=data, model='conv3d'),
            select(results, eval=eval, data=data, model='ensemble'),
            select(results, eval=eval, data=data, model='spatial_transform'),
        ]
        labels = ['standard', 'pixel_pool', 'slice_pool', 'conv3d', 'ensemble', 'spatial_trans']
        colors = ['#00A3E0', '#43B02A', '#FFB81C', '#C8102E', '#779FB5', '#002F6C']
        handles = []
        for i in range(6):
            means, stds = lines[i]
            axis.scatter(range(17, 65), means, s=2, c=colors[i], zorder=5)
            handle, = axis.plot(range(17, 65), means, ':', lw=1, c=colors[i], zorder=5)
            # axis.fill_between(range(17, 65), means - stds, means + stds, fc=colors[i] + '0c', ec=None)
            handles.append(handle)
        axis.yaxis.grid(zorder=-1, c='#eeee')
        kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
        if eval == 1:
            axis.legend([handles[0], handles[1]], labels[:2], **kwargs)
        if eval == 2:
            axis.legend([handles[2], handles[3]], labels[2:4], **kwargs)
        if eval == 3:
            axis.legend([handles[4], handles[5]], labels[4:], **kwargs)
    plt.tight_layout()
    #plt.show()
    plt.savefig('generalization.pdf', bbox_inches='tight')


def main():
    # Write plot to file
    data = pd.read_csv('generalization.csv')
    plot_for_paper(data)


if __name__ == '__main__':
    main()
