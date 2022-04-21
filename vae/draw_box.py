import random
import numpy as np
from matplotlib import pyplot as plt


def draw():
    color = (0, 0, 0)
    color = [x / 255. for x in color]
    size = 25
    sample = np.ones((size, size, 3))
    nbox = 1  # 15
    scale = 3  # 30
    center = np.random.normal(loc=size / 2, scale=scale, size=nbox)
    width = np.random.normal(loc=size / 4, scale=scale, size=nbox)
    height = np.random.normal(loc=size / 3, scale=scale, size=nbox)
    for c, w, h in zip(center, width, height):
        left = np.clip(int(c - w / 2), 0, size - 1)
        right = np.clip(int(c + w / 2), 0, size - 1)
        top = np.clip(int(c - h / 2), 0, size - 1)
        bottom = np.clip(int(c + h / 2), 0, size - 1)
        for i in range(left, right):
            sample[top, i, :] = color
            sample[bottom, i, :] = color
        for j in range(top, bottom):
            sample[j, left, :] = color
            sample[j, right, :] = color
    return sample


def main():
    nrow = 2
    ncol = 2
    nplot = nrow * ncol
    name = 'box.png'
    fig, axes = plt.subplots(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            sample = draw()
            if nplot == 1 and i == 0 and j == 0:
                ax = axes
            else:
                ax = axes[i][j]
            ax.imshow(sample, norm=None)
            ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(name)
    plt.close('all')


if __name__ == '__main__':
    main()
