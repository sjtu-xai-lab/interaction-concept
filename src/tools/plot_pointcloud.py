import os
import os.path as osp
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Union

import sys
sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
from common_harsanyi.src.harsanyi.interaction_utils import flatten



def visualize_easy(point_cloud, save_path):
    """
    Visualize the point cloud
    :param point_cloud: [3, n_pts]
    :param save_path:
    :return:
    """
    assert point_cloud.shape[0] == 3
    x, y, z = point_cloud[0], point_cloud[1], point_cloud[2]
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111)
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c='gray', s=10, alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def visualize_parts(point_cloud, part_labels, save_folder, save_name, part_names=None):
    """
    Visualize a point cloud by parts
    :param point_cloud: [3, n_pts]
    :param part_labels: per-point label for the point cloud [n_pts,]
    :param save_folder: str
    :param save_name: str
    :param part_names: dict, a mapping from part label to part name
    :return:
    """
    os.makedirs(save_folder, exist_ok=True)
    all_labels = sorted(np.unique(part_labels))
    assert point_cloud.shape[0] == 3
    x, y, z = point_cloud[0], point_cloud[1], point_cloud[2]

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111)
    ax = Axes3D(fig)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    for label in all_labels:
        indices = part_labels == label
        if part_names is not None:
            name = part_names[label]
            ax.scatter(x[indices], y[indices], z[indices], s=10, alpha=0.7, label=name)
        else:
            ax.scatter(x[indices], y[indices], z[indices], s=10, alpha=0.7)
    if part_names is not None:
        plt.legend()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def plot_coalition(point_cloud, coalition, save_folder, save_name, player_names=None, **kwargs):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    if len(point_cloud.shape) == 3:
        assert point_cloud.shape[0] == 1
        point_cloud = point_cloud.squeeze(0)
    assert point_cloud.shape[0] == 3
    n_points = point_cloud.shape[1]
    x, y, z = point_cloud[0], point_cloud[1], point_cloud[2]
    # print(x.shape, y.shape, z.shape)
    if player_names is not None:
        assert len(coalition) == len(player_names)

    os.makedirs(save_folder, exist_ok=True)

    # split foreground indices and background indices
    foreground = list(flatten(coalition))
    indices = np.ones(n_points, dtype=bool)
    indices[foreground] = False
    background = np.arange(n_points)[indices].tolist()

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111)
    ax = Axes3D(fig, elev=120, azim=-90)
    # ax = Axes3D(fig)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    # plot background
    ax.scatter(x[background], y[background], z[background], s=10, c="gray", alpha=0.3, rasterized=True,
               edgecolors=None, linewidths=0, depthshade=False)
    # plot each player (foreground)
    for i in range(len(coalition)):
        player = coalition[i]
        if player_names is not None:
            name = player_names[i]
            ax.scatter(x[player], y[player], z[player], s=30, alpha=0.7, label=name, rasterized=True,
                       edgecolors=None, linewidths=0, depthshade=False)
        else:
            ax.scatter(x[player], y[player], z[player], s=30, alpha=0.7, rasterized=True,
                       edgecolors=None, linewidths=0, depthshade=False)

    if player_names is not None:
        plt.legend()

    if 'title' in kwargs:
        plt.title(kwargs['title'])

    format = "png" if 'save_format' not in kwargs else kwargs['save_format']
    transparent = False if "transparent" not in kwargs else kwargs["transparent"]
    plt.savefig(osp.join(save_folder, f"{save_name}.{format}"),
                dpi=200, transparent=transparent)
    plt.close("all")





