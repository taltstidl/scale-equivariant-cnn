import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def select(hparams, eval, model, data, interp, k):
    """ Filter data table by model, dataset, interpolation and kernel size. """
    rows = hparams[(hparams['params.model'] == model) & (hparams['params.data'] == data) &
                   (hparams['params.interpolation'] == interp) & (hparams['params.kernel_size'] == k) &
                   (hparams['params.evaluation'] == eval)]
    metrics = rows['metrics.test_acc'].to_numpy()
    metrics = metrics[~np.isnan(metrics)]
    if metrics.shape[0] != 50:
        args = (eval, model, data, interp, k, metrics.shape[0])
        # warnings.warn('Missing runs for eval=`{}` model=`{}` data=`{}` interp=`{}` and k=`{}` ({})!'.format(*args))
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
    median = boxplots['medians'][i]
    median.set_color('black')


def plot_for_appendix(runs):
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
            eval += 1  # Evaluation is 1-based, not 0-based
            boxes = [
                select(runs, eval=eval, data=data, model='standard', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='spatial_transform', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='ensemble', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='xu', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='kanazawa', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='hermite', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='disco', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='pixel_pool', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='slice_pool', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='energy_pool', interp='bicubic', k=7),
                select(runs, eval=eval, data=data, model='conv3d', interp='bicubic', k=7),
            ]
            labels = ['standard', 'spatial_trans', 'ensemble', 'xu', 'kanazawa', 'hermite', 'disco',
                      'pixel_pool', 'slice_pool', 'energy_pool', 'conv3d']
            # Colors from http://mkweb.bcgsc.ca/colorblind/palettes/12.color.blindness.palette.txt
            colors = ['#009F81', '#00FCCF', '#9F0162', '#FF5AAF', '#8400CD', '#008DF9', '#00C2F9',
                      '#A40122', '#E20134', '#FF6E3A', '#FFC33B', '#FFB2FD']
            boxplots = axis.boxplot(boxes, showfliers=False, patch_artist=True)
            for i in range(len(boxes)):
                num_runs = len(boxes[i])  # number of repetitions, should be 50, may be smaller due to HPC kill
                pos = np.full(shape=(num_runs,), fill_value=i + 1) + np.random.normal(scale=0.1, size=(num_runs,))
                axis.scatter(pos, boxes[i], s=3, c=colors[i] + '25')
                color_box(boxplots, i, colors[i])
            axis.xaxis.set_visible(False)
            axis.yaxis.grid(zorder=0, c='#eeee')
            kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
            if index == 0 and eval == 4:
                axis.legend(boxplots['boxes'], labels, ncol=6, loc='lower right', bbox_to_anchor=(1, 1.2), **kwargs)
            if index == 3:
                axis.xaxis.set_visible(True)
                axis.xaxis.set_ticks(range(1, 12))
                axis.xaxis.set_ticklabels(labels, rotation=60, ha='right')
                for tick in axis.get_xticklabels():
                    tick.set_fontname('Corbel')
                    tick.set_fontsize(8)
    plt.tight_layout()
    plt.savefig('hparams_appendix.pdf', bbox_inches='tight')


def plot_summary(runs):
    """ Plot simplified overview for paper (with interp=bicubic and kernel=7x7). """
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey='all', figsize=(7.5, 2.5))
    models = ['standard', 'spatial_transform', 'ensemble', 'xu', 'kanazawa', 'hermite', 'disco', 'pixel_pool',
              'slice_pool'][::-1]
    model_labels = ['Standard', 'SpatialTrans', 'Ensemble', 'Xu', 'Kanazawa', 'Hermite', 'Disco', 'PixelPool (Ours)',
              'SlicePool (Ours)'][::-1]
    datasets = ['emoji', 'mnist', 'trafficsign', 'aerial']
    dataset_labels = ['emoji', 'mnist', 'tsign', 'aerial']
    colors = ['#FDB735', '#C50F3C', '#18B4F1', '#7BB725']
    markers = ['o', 's', 'D', '^']
    axis = axs[0]  # First axis for mid to rest evaluation
    axis.set_xlim((36, 104))
    axis.set_title('Mid2Rest Evaluation', fontname='Corbel', fontsize=10)
    axis.set_xlabel('Testing Accuracy [%]', fontname='Corbel')
    for i, dataset in enumerate(datasets):
        acc = [np.mean(select(runs, eval=2, data=dataset, model=m, interp='bicubic', k=7)) for m in models]
        idx = sorted(range(len(acc)), key=acc.__getitem__, reverse=True)[:3]  # top 3 indices
        fc = [colors[i] if id in idx else 'None' for id in range(len(acc))]
        axis.axvline(x=acc[-1], c=colors[i], ls=':', lw=1)
        axis.scatter(acc, model_labels, marker=markers[i], fc=fc, ec=colors[i], s=10, zorder=3, label=dataset_labels[i])
    axis.xaxis.grid(zorder=0, c='#eeee')
    axis.yaxis.grid(zorder=0, c='#eeee', ls=':')
    for tick in axis.get_yticklabels():
        tick.set_fontname('Corbel')
        tick.set_fontsize(10)
    axis = axs[1]  # Second axis for all to all evaluation
    axis.set_xlim((36, 104))
    axis.set_title('All2All Evaluation', fontname='Corbel', fontsize=10)
    axis.set_xlabel('Testing Accuracy [%]', fontname='Corbel')
    for i, dataset in enumerate(datasets):
        acc = [np.median(select(runs, eval=4, data=dataset, model=m, interp='bicubic', k=7)) for m in models]
        idx = sorted(range(len(acc)), key=acc.__getitem__, reverse=True)[:3]  # top 3 indices
        fc = [colors[i] if id in idx else 'None' for id in range(len(acc))]
        axis.axvline(x=acc[-1], c=colors[i], ls=':', lw=1)
        axis.scatter(acc, model_labels, marker=markers[i], fc=fc, ec=colors[i], s=8, zorder=3, label=dataset_labels[i])
    axis.xaxis.grid(zorder=0, c='#eeee')
    axis.yaxis.grid(zorder=0, c='#eeee', ls=':')
    kwargs = {'prop': {'family': 'Corbel', 'size': 9}, 'handlelength': 0.8, 'frameon': False}
    handles, labels = axis.get_legend_handles_labels()
    for handle in handles:
        handle.set_facecolor('none')  # Ensure all symbols are non-filled
    axis.legend(handles, labels, **kwargs)
    plt.tight_layout()
    plt.savefig('hparams.pdf', bbox_inches='tight')


def main():
    # Write plot to file
    runs = pd.read_csv('../scripts/clean.csv')
    plot_for_appendix(runs)
    plot_summary(runs)


if __name__ == '__main__':
    main()
