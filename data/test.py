import unittest

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class GeneratorTest(unittest.TestCase):
    @staticmethod
    def _check_data(path, is_color, num_classes):
        data = np.load(path)
        assert set(data.files) == {'imgs', 'lbls', 'scls', 'psts', 'metadata', 'lbldata'}
        imgs = data['imgs']
        assert imgs.shape == (3, 48, num_classes, 64, 64, 3) if is_color else (3, 48, num_classes, 64, 64)
        assert imgs.dtype == np.uint8
        lbls = data['lbls']
        assert lbls.shape == (3, 48, num_classes)
        assert lbls.dtype == np.int32
        assert lbls.min() == 0
        assert lbls.max() == num_classes - 1
        scls = data['scls']
        assert scls.shape == (3, 48, num_classes)
        assert scls.dtype == np.int32
        psts = data['psts']
        assert psts.shape == (3, 48, num_classes, 2)
        assert psts.dtype == np.int32
        lbldata = data['lbldata']
        assert lbldata.shape == (num_classes,)
        assert lbldata.dtype.kind == 'U'

    @staticmethod
    def test_emoji():
        GeneratorTest._check_data('emoji.npz', False, 36)

    @staticmethod
    def test_mnist():
        GeneratorTest._check_data('mnist.npz', False, 10)

    @staticmethod
    def test_trafficsign():
        GeneratorTest._check_data('trafficsign.npz', True, 40)


class DatasetTest(unittest.TestCase):
    @staticmethod
    def _plot_bounding_box(path, target):
        data = np.load(path)
        # Get five sample images at size 32
        test_images = data['imgs'][0, 32, :5]
        test_scales = data['scls'][0, 32, :5]
        test_positions = data['psts'][0, 32, :5]
        fig, axs = plt.subplots(nrows=1, ncols=5)
        for i in range(5):
            s, tx, ty = test_scales[i], test_positions[i, 0], test_positions[i, 1]
            axs[i].set_title('({}, {}) {}'.format(tx, ty, s))
            axs[i].imshow(test_images[i])
            rect = patches.Rectangle((tx - 1, ty - 1), s, s, linewidth=1, edgecolor='r', facecolor='none')
            axs[i].add_patch(rect)
        plt.tight_layout()
        plt.savefig(target, bbox_inches='tight')

    @staticmethod
    def test_emoji():
        DatasetTest._plot_bounding_box('emoji.npz', 'emoji_bbox.png')

    @staticmethod
    def test_mnist():
        DatasetTest._plot_bounding_box('mnist.npz', 'mnist_bbox.png')

    @staticmethod
    def test_trafficsign():
        DatasetTest._plot_bounding_box('trafficsign.npz', 'trafficsign_bbox.png')


if __name__ == '__main__':
    unittest.main()
