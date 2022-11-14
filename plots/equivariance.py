import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def remove_outliers(points):
    """ Remove outliers. """
    equivariances, accuracies = points
    q1, q3 = np.quantile(equivariances, 0.25), np.quantile(equivariances, 0.75)
    iqr = q3 - q1
    idx = (equivariances >= q1 - (1.5 * iqr)) & (equivariances <= q3 + (1.5 * iqr))
    return equivariances[idx], accuracies[idx]


def collect(runs, eval, model, data):
    """ Collect equivariance errors from disk. """
    rows = runs[(runs['params.data'] == data) & (runs['params.model'] == model) & (runs['params.evaluation'] == eval)]
    idx = [None, (2, 3, 4, 5), (0, 1, 4, 5), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5)][eval]
    equivariances, accuracies = [], []
    for _, row in rows.iterrows():
        errors_path = os.path.join('eval', row['run_id'], 'errors.npz')
        errors = np.load(errors_path)
        stage1, stage2 = errors['stage1'], errors['stage2']  # each of shape (num_classes, num_instances, 6)
        equivariances.append(np.mean(stage2[:, :, idx]))
        accuracies.append(row['metrics.test_acc'])
    return np.asarray(equivariances), np.asarray(accuracies)


def plot_for_appendix(runs):
    fig, axs = plt.subplots(nrows=3, ncols=4, sharey='row', figsize=(7.5, 5.0))
    for index, data in enumerate(['emoji', 'mnist', 'trafficsign']):  # Loop over datasets
        for eval in range(4):  # Loop over evaluation scenarios
            axis = axs[index][eval]
            if index == 0:
                eval_titles = ['Small2Large', 'Mid2Rest', 'Large2Small', 'All2All']
                axis.set_title(eval_titles[eval], fontname='Corbel', fontsize=10)
            if eval == 0:
                data_name = ['emoji', 'mnist', 'tsign'][index]
                axis.set_ylabel('{}\nTesting Accuracy [%]'.format(data_name), fontname='Corbel')
            eval += 1  # Evaluation is 1-based, not 0-based
            points = [
                collect(runs, eval=eval, data=data, model='standard'),
                collect(runs, eval=eval, data=data, model='kanazawa'),
                collect(runs, eval=eval, data=data, model='pixel_pool'),
                collect(runs, eval=eval, data=data, model='slice_pool'),
            ]
            all_equivariances = np.concatenate([p[0][np.isfinite(p[0])] for p in points])
            q1, q3 = np.quantile(all_equivariances, 0.25), np.quantile(all_equivariances, 0.75)
            iqr = q3 - q1
            labels = ['standard', 'kanazawa', 'pixel_pool', 'slice_pool']
            colors = ['#009F81', '#8400CD', '#E20134', '#FF6E3A']
            for i in range(len(points)):
                axis.scatter(points[i][0], points[i][1], s=2, c=colors[i], zorder=5, label=labels[i])
            axis.yaxis.grid(zorder=-1, c='#eeee')
            axis.xaxis.grid(zorder=-1, c='#eeee')
            axis.set_xscale('log')
            # axis.set_xlim((q1 - (1.5 * iqr), q3 + (1.5 * iqr)))
            kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
            if index == 0 and eval == 4:
                axis.legend(ncol=4, loc='lower right', bbox_to_anchor=(1, 1.2), **kwargs)
            if index == 2:
                axis.set_xlabel('Equivariance Error', fontname='Corbel')
    plt.tight_layout()
    plt.savefig('equivariance_appendix.pdf', bbox_inches='tight')


def plot_summary(runs):
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.5))
    axis.set_title('Mid2Rest Evaluation', fontname='Corbel', fontsize=10)
    axis.set_xlabel('Equivariance Error', fontname='Corbel')
    axis.set_ylabel('Testing Accuracy [%]', fontname='Corbel')
    points = [
        collect(runs, eval=2, data='emoji', model='standard'),
        collect(runs, eval=2, data='emoji', model='kanazawa'),
        collect(runs, eval=2, data='emoji', model='pixel_pool'),
        collect(runs, eval=2, data='emoji', model='slice_pool'),
    ]
    labels = ['standard', 'kanazawa', 'pixel_pool', 'slice_pool']
    colors = ['#7BB725', '#18B4F1', '#C50F3C', '#FDB735']
    markers = ['^', 'D', 's', 'o']
    for i in range(len(points)):
        axis.scatter(points[i][0], points[i][1], marker=markers[i], c=colors[i], s=2, zorder=3, label=labels[i])
    axis.yaxis.grid(zorder=-1, c='#eeee')
    axis.xaxis.grid(zorder=-1, c='#eeee')
    axis.set_xlim((-0.1, 3.1))
    kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'handletextpad': 0.2, 'frameon': True}
    axis.legend(ncol=2, columnspacing=0.5, **kwargs)
    plt.tight_layout()
    plt.savefig('equivariance.pdf', bbox_inches='tight')


def main():
    # Write plot to file
    runs = pd.read_csv('../scripts/clean.csv')
    plot_for_appendix(runs)
    plot_summary(runs)


if __name__ == '__main__':
    main()
