import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress


def find_model(runs, eval, model, data):
    rows = runs[(runs['params.data'] == data) & (runs['params.model'] == model) & (runs['params.evaluation'] == eval)]
    run_id = rows.loc[rows['metrics.test_acc'].idxmax()]['run_id']
    return run_id


def plot_for_appendix(pixel_id):
    """ Plot pixel pooling indices. """
    pixel_indices = np.load(os.path.join('eval', pixel_id, 'indices.npz'))['pool2']  # (1728, 32, 52, 52)
    num_scales = 48
    num_instances = pixel_indices.shape[0] // num_scales
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("fau", ['#FDB735', '#04316A'])
    pixel_norm = Normalize(vmin=7, vmax=7 + 2 * pixel_indices.max())
    ii = 33  # Selected emoji index
    # Start actual plotting
    width_ratios = [1, 1, 1, 1, 0.4, 1, 1, 1, 1, 0.4, 1, 1, 1, 1]
    fig, axs = plt.subplots(nrows=11, ncols=14, figsize=(7.5, 7.5), gridspec_kw={'width_ratios': width_ratios})
    for ki in range(32):
        kernel_scales = 7 + 2 * pixel_indices[ii::num_instances, ki]
        kernel_scales = kernel_scales[[0, 16, 31, 47]]  # Select only every 2nd
        for si in range(4):
            row, col = ki // 3, 5 * (ki % 3) + si
            axis = axs[row][col]
            axis.imshow(kernel_scales[si], cmap=cmap, vmin=pixel_norm.vmin, vmax=pixel_norm.vmax)
            if ki in [0, 1, 2]:
                axis.set_title('{0}x{0}'.format([17, 33, 48, 64][si]), fontname='Corbel', fontsize=8)
            if si == 0:
                axis.set_ylabel('Kernel #{}'.format(ki + 1), fontname='Corbel', fontsize=8)
            # Disable all axis elements
            for loc in ['top', 'right', 'bottom', 'left']:
                axis.spines[loc].set_visible(False)
            axis.tick_params(bottom=False, labelbottom=False)
            axis.tick_params(left=False, labelleft=False)
    for axis in [axs[10, i] for i in range(10, 14)]:
        axis.axis('off')
    for axis in [axs[i, 4] for i in range(11)]:
        axis.axis('off')  # phantom axis for padding
    for axis in [axs[i, 9] for i in range(11)]:
        axis.axis('off')  # phantom axis for padding
    # divider = make_axes_locatable()
    cax = inset_axes(axs[10][10], width='430%', height='20%', loc='center left', borderpad=0)
    cb = plt.colorbar(ScalarMappable(norm=pixel_norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([pixel_norm.vmin, pixel_norm.vmax])
    cb.ax.set_title('Kernel Size', fontname='Corbel', fontsize=8)
    for tick in cb.ax.get_xticklabels():
        tick.set_fontname('Corbel')
        tick.set_fontsize(8)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('indices_appendix.pdf', bbox_inches='tight')


def plot_summary(slice_id):
    """ Plot slice pooling indices. """
    slice_indices = np.load(os.path.join('eval', slice_id, 'indices.npz'))['pool2']  # (1728, 32)
    num_scales = 48
    num_instances = slice_indices.shape[0] // num_scales
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("fau", ['#FDB735', '#04316A'])
    slice_norm = Normalize(vmin=7, vmax=7 + 2 * slice_indices.max())
    ii = 33  # Selected emoji index
    # Start actual plotting
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.5))
    axis.set_title('Mid2Rest Evaluation', fontname='Corbel', fontsize=10)
    axis.set_xlabel('Scale', fontname='Corbel')
    axis.set_ylabel('Output Channel', fontname='Corbel')
    object_scales = np.arange(17, 65)
    r_values = [linregress(object_scales, 7 + 2 * slice_indices[ii::num_instances, i])[2] for i in range(32)]
    kis = sorted(range(len(r_values)), key=r_values.__getitem__)
    for pi, ki in enumerate(kis):
        object_scales = np.arange(17, 65)
        kernel_scales = 7 + 2 * slice_indices[ii::num_instances, ki]
        axis.scatter(object_scales, np.full(num_scales, pi), c=kernel_scales, cmap=cmap, vmin=7, vmax=37,
                     s=2, zorder=5, clip_on=False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.set_xlim((17, 64))
    axis.yaxis.set_ticks(np.arange(0, 32))
    axis.yaxis.set_ticklabels([])
    axis.yaxis.set_ticks_position('none')
    axis.yaxis.grid(zorder=-1, c='#eeee')
    cb = plt.colorbar(ScalarMappable(norm=slice_norm, cmap=cmap))
    cb.outline.set_visible(False)
    cb.set_ticks([7, 37])
    cb.ax.set_ylabel('Kernel Size', fontname='Corbel', rotation=270)
    plt.tight_layout()
    plt.savefig('indices.pdf', bbox_inches='tight')


def main():
    # Write plot to file
    runs = pd.read_csv('../scripts/clean.csv')
    pixel_id = find_model(runs, eval=2, model='pixel_pool', data='emoji')
    slice_id = find_model(runs, eval=2, model='slice_pool', data='emoji')
    plot_for_appendix(pixel_id)
    plot_summary(slice_id)


if __name__ == '__main__':
    main()
