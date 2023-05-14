import sys
import os
import os.path as osp
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from typing import Union


def plot_coalition(sample, coalition, save_folder, save_name, **kwargs):
    if len(sample.shape) == 2:
        assert sample.shape[0] == 1
        sample = sample[0]
    assert len(sample.shape) == 1

    os.makedirs(save_folder, exist_ok=True)
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()

    background = np.zeros(sample.shape[0], dtype=bool)
    background[coalition] = True
    background = np.arange(sample.shape[0])[~background]

    plt.figure()
    plt.barh(background, sample[background], color="gray", alpha=0.5)
    plt.barh(coalition, sample[coalition], color="tab:blue", alpha=0.7)
    ax = plt.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    for i in range(sample.shape[0]):
        plt.hlines(0.5 + i, *x_lim, colors='gray', alpha=0.3, linewidth=0.5)
    plt.vlines(0, *y_lim, colors='black', alpha=1.0, linewidth=0.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    plt.tight_layout()

    format = "png" if 'save_format' not in kwargs else kwargs['save_format']
    plt.savefig(osp.join(save_folder, f"{save_name}.{format}"), dpi=200)
    plt.close("all")


if __name__ == '__main__':
    sample = torch.randn(1, 10)
    coalition = [3, 5]
    plot_coalition(sample, coalition, ".", "test_tabular_coalition")