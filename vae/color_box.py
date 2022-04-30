from torch.nn import functional as F
import torch
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

import gflags
FLAGS = gflags.FLAGS


gflags.DEFINE_string('filename', None, '')


palette = ['83E76B', 'CDE584', 'DEDB6A', 'F39D58', 'C44028']
#palette = ['EA1B20', '121111']
#palette = ['1E5652']


def save_images(images, name, nrow, ncol):
    fig, axes = plt.subplots(nrow, ncol)
    nplot = len(images)
    assert nrow * ncol == nplot, f'{nrow}, {ncol}, {nplot}'
    for i in range(nrow):
        for j in range(ncol):
            c = i * ncol + j
            image = images[c]
            if nplot == 1 and i == 0 and j == 0:
                ax = axes
            else:
                ax = axes[i][j]
            # clr = np.array(colors.to_rgb(f'#{palette[c % len(palette)]}'))
            #spot = 1. * (image < 0.95)
            #clr_image = np.ones_like(image) * clr
            #spot_image = np.ones_like(image)
            #image = (1 - spot) * spot_image + spot * clr_image
            image_tensor = torch.tensor(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.permute(0, 3, 1, 2)
            upsampled = F.interpolate(
                image_tensor, scale_factor=32, mode='nearest')
            upsampled = upsampled.permute(0, 2, 3, 1).squeeze(0)
            ax.imshow(upsampled.numpy())
            #ax.imshow(image, norm=None)
            # ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(name)
    plt.close('all')


def load(filename):
    assert filename.endswith('.npz'), filename
    dict_data = np.load(filename)
    data = dict_data['arr_0']
    return data


def main():
    data = load(FLAGS.filename)
    save_images(data[:1], 'replot.png', 1, 1)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
