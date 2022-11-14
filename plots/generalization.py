import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def select(results, eval, model, data):
    """ Filter data table by model, dataset, interpolation and kernel size. """
    rows = results[(results['data'] == data) & (results['model'] == model) &
                   (results['eval'] == eval)]
    acc_means = 100 * np.array([np.mean(rows['s{}'.format(s)]) for s in range(17, 65)])
    acc_stds = 100 * np.array([rows['s{}'.format(s)].std() for s in range(17, 65)])
    if acc_means.shape[0] != 48 or acc_stds.shape[0] != 48:
        args = (eval, model, data, acc_means.shape[0], acc_stds.shape[0])
        warnings.warn('Missing runs for eval=`{}` model=`{}` and data=`{}` ({} | {})!'.format(*args))
    return acc_means, acc_stds


def plot_for_appendix(results):
    """ Plot results for appendix (with interp=bicubic and kernel=7x7). """
    fig, axs = plt.subplots(nrows=4, ncols=4, sharex='all', sharey='all', figsize=(7.5, 6.5))
    for index, data in enumerate(['emoji', 'mnist', 'trafficsign', 'aerial']):  # Loop over datasets
        for eval in range(4):  # Loop over evaluation scenarios
            axis = axs[index][eval]
            if index == 0:
                eval_titles = ['Small2Large', 'Mid2Rest', 'Large2Small', 'All2All']
                axis.set_title(eval_titles[eval], fontname='Corbel', fontsize=10)
            if eval == 0:
                data_name = ['emoji', 'mnist', 'tsign', 'aerial'][index]
                axis.set_ylabel('{}\nTesting Accuracy [%]'.format(data_name), fontname='Corbel')
            axis.axvspan([17, 33, 49, 17][eval], [32, 48, 64, 64][eval], fc='#eeee', zorder=-1)
            eval += 1  # Evaluation is 1-based, not 0-based
            lines = [
                select(results, eval=eval, data=data, model='standard'),
                select(results, eval=eval, data=data, model='spatial_transform'),
                select(results, eval=eval, data=data, model='ensemble'),
                select(results, eval=eval, data=data, model='xu'),
                select(results, eval=eval, data=data, model='kanazawa'),
                select(results, eval=eval, data=data, model='hermite'),
                select(results, eval=eval, data=data, model='disco'),
                select(results, eval=eval, data=data, model='pixel_pool'),
                select(results, eval=eval, data=data, model='slice_pool'),
                select(results, eval=eval, data=data, model='energy_pool'),
                select(results, eval=eval, data=data, model='conv3d'),
            ]
            labels = ['standard', 'spatial_trans', 'ensemble', 'xu', 'kanazawa', 'hermite', 'disco',
                      'pixel_pool', 'slice_pool', 'energy_pool', 'conv3d']
            # Colors from http://mkweb.bcgsc.ca/colorblind/palettes/12.color.blindness.palette.txt
            colors = ['#009F81', '#00FCCF', '#9F0162', '#FF5AAF', '#8400CD', '#008DF9', '#00C2F9',
                      '#A40122', '#E20134', '#FF6E3A', '#FFC33B', '#FFB2FD']
            handles = []
            for i in range(len(lines)):
                means, stds = lines[i]
                # axis.scatter(range(17, 65), means, s=2, c=colors[i], zorder=5)
                handle, = axis.plot(range(17, 65), means, '-', lw=1, c=colors[i], zorder=5)
                handles.append(handle)
            axis.yaxis.grid(zorder=-1, c='#eeee')
            kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
            if index == 0 and eval == 4:
                axis.legend(handles, labels, ncol=6, loc='lower right', bbox_to_anchor=(1, 1.2), **kwargs)
            if index == 3:
                axis.set_xlabel('Scale', fontname='Corbel')
    plt.tight_layout()
    plt.savefig('generalization_appendix.pdf', bbox_inches='tight')


def plot_summary(results, data='mnist'):
    """ Plot just handwritten digits as representative (with interp=bicubic and kernel=7x7). """
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all', figsize=(7.5, 2.5))
    labels = ['Standard', 'SpatialTrans', 'Ensemble', 'Xu', 'Kanazawa', 'Hermite', 'Disco', 'PixelPool (Ours)',
              'SlicePool (Ours)']
    for eval in range(3):  # Loop over evaluation scenarios
        axis = axs[eval]
        axis.set_title(['Small2Large', 'Mid2Rest', 'Large2Small'][eval], fontname='Corbel', fontsize=10)
        axis.set_xlabel('Scale', fontname='Corbel')
        if eval == 0:
            axis.set_ylabel('Testing Accuracy [%]', fontname='Corbel')
        axis.axvspan([17, 33, 49][eval], [32, 48, 64][eval], fc='#eeee', zorder=-1)
        eval += 1  # Evaluation is 1-based, not 0-based
        lines = [
            select(results, eval=eval, data=data, model='standard'),
            select(results, eval=eval, data=data, model='spatial_transform'),
            select(results, eval=eval, data=data, model='ensemble'),
            select(results, eval=eval, data=data, model='xu'),
            select(results, eval=eval, data=data, model='kanazawa'),
            select(results, eval=eval, data=data, model='hermite'),
            select(results, eval=eval, data=data, model='disco'),
            select(results, eval=eval, data=data, model='pixel_pool'),
            select(results, eval=eval, data=data, model='slice_pool'),
        ]
        colors = ['#7BB725', '#DEEDC8', '#18B4F1', '#C5ECFB', '#E2E7EB', '#C0CBDA', '#04316A',
                  '#C50F3C', '#FDB735']
        styles = ['-', ':', '-', ':', ':', ':', '-', '-', '-']
        orders = [7, 5, 8, 5, 5, 5, 6, 9, 10, 5]
        handles = []
        for i in range(len(lines)):
            means, stds = lines[i]
            # axis.scatter(range(17, 65), means, s=1, c=colors[i], zorder=orders[i])
            handle, = axis.plot(range(17, 65), means, 'o-', lw=0.5, ms=1, c=colors[i], zorder=orders[i])
            handles.append(handle)
        axis.yaxis.grid(zorder=-1, c='#eeee')
        kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
        if eval == 2:
            axis.legend(handles[:5], labels[:5], loc='lower center', **kwargs)
        if eval == 3:
            axis.legend(handles[5:], labels[5:], loc='lower right', **kwargs)
    plt.tight_layout()
    plt.savefig('generalization.pdf', bbox_inches='tight')


def main():
    # Write plot to file
    data = ['emoji', 'mnist', 'trafficsign', 'aerial']
    results = pd.concat([pd.read_csv('generalization_{}.csv'.format(d)) for d in data])
    plot_for_appendix(results)
    plot_summary(results)


if __name__ == '__main__':
    main()
