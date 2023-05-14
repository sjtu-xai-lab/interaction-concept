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
from .utils import makedirs

from typing import Union

sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
from common_harsanyi.src.harsanyi.interaction_utils import flatten


def _convert_mean_std_to_mat(image: Union[np.ndarray, torch.Tensor], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            assert image.shape[0] == len(mean) == len(std), f"image shape: {image.shape}"
            mean = np.array(mean).reshape((-1, 1, 1))
            std = np.array(std).reshape((-1, 1, 1))
        elif len(image.shape) == 4:
            assert image.shape[1] == len(mean) == len(std), f"image shape: {image.shape}"
            mean = np.array(mean).reshape((1, -1, 1, 1))
            std = np.array(std).reshape((1, -1, 1, 1))
        else:
            raise NotImplementedError(f"Unknown shape of data: {image.shape}")
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            assert image.shape[0] == len(mean) == len(std), f"image shape: {image.shape}"
            mean = torch.Tensor(mean).view((-1, 1, 1))
            std = torch.Tensor(std).view((-1, 1, 1))
        elif len(image.shape) == 4:
            assert image.shape[1] == len(mean) == len(std), f"image shape: {image.shape}"
            mean = torch.Tensor(mean).view((1, -1, 1, 1))
            std = torch.Tensor(std).view((1, -1, 1, 1))
        else:
            raise NotImplementedError(f"Unknown shape of data: {image.shape}")
    else:
        raise NotImplementedError(f"Unknown type of data: {type(image)}")

    return mean, std


def denormalize_image(image: Union[np.ndarray, torch.Tensor], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean, std = _convert_mean_std_to_mat(image, mean, std)
    image = image * std + mean
    return image


def normalize_image(image: Union[np.ndarray, torch.Tensor], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean, std = _convert_mean_std_to_mat(image, mean, std)
    image = (image - mean) / std
    return image


def get_selected_region_mask(sqrt_grid_num, selected_regions_id):
    if isinstance(selected_regions_id, int): selected_regions_id = [selected_regions_id]
    mask = np.zeros(shape=(sqrt_grid_num, sqrt_grid_num))
    for region_id in selected_regions_id:
        row_id = region_id // sqrt_grid_num
        column_id = region_id % sqrt_grid_num
        mask[row_id, column_id] = 1
    return mask


def _plot_selected_region_boundary(ax, selected_regions_mask, sqrt_grid_num, grid_width, linecolor="red", **kwargs):
    if "linewidth" in kwargs:
        linewidth = kwargs['linewidth']
    else:
        linewidth = 4
    # n_line = 0
    for i in range(selected_regions_mask.shape[0]):
        for j in range(selected_regions_mask.shape[1]):
            if selected_regions_mask[i, j] == 0: continue
            # edge on the left
            if (j != 0 and selected_regions_mask[i, j-1] == 0) or j == 0:
                line = Line2D(
                    xdata=[j*grid_width - 0.5, j*grid_width - 0.5],
                    ydata=[i*grid_width - 0.5, (i+1)*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
                # n_line += 1
            # edge on the right
            if (j != sqrt_grid_num-1 and selected_regions_mask[i, j+1] == 0) or j == sqrt_grid_num-1:
                line = Line2D(
                    xdata=[(j+1)*grid_width - 0.5, (j+1)*grid_width - 0.5],
                    ydata=[i*grid_width - 0.5, (i+1)*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
                # n_line += 1
            # edge on the top
            if (i != 0 and selected_regions_mask[i-1, j] == 0) or i == 0:
                line = Line2D(
                    xdata=[j*grid_width - 0.5, (j+1)*grid_width - 0.5],
                    ydata=[i*grid_width - 0.5, i*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
                # n_line += 1
            # edge on the bottom
            if (i != sqrt_grid_num-1 and selected_regions_mask[i+1, j] == 0) or i == sqrt_grid_num-1:
                line = Line2D(
                    xdata=[j*grid_width - 0.5, (j+1)*grid_width - 0.5],
                    ydata=[(i+1)*grid_width - 0.5, (i+1)*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
                # n_line += 1
    # print("n_line", n_line)


def plot_coalition(image, grid_width, coalition, save_folder, save_name, alpha=0.5, overlay_color=(0, 0, 0), **kwargs):
    makedirs(save_folder)
    # linecolors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    # ]
    linecolors = ["r"]

    image_channel = image.shape[0]
    image_width = image.shape[2]
    sqrt_grid_num = int(np.ceil(image_width / grid_width))
    selected_regions_mask = get_selected_region_mask(sqrt_grid_num, flatten(coalition))

    # masked image
    overlay_color = np.array(overlay_color).reshape((-1, 1, 1))
    if image_channel == 1:
        image_ = image.copy()
        image_ = np.concatenate([image_, image_, image_], axis=0)
    else:
        image_ = image.copy()
    for i in range(selected_regions_mask.shape[0]):
        for j in range(selected_regions_mask.shape[1]):
            if selected_regions_mask[i, j] == 0:
                image_[:, i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] *= (1 - alpha)
                image_[:, i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] += overlay_color * alpha

    if "figsize" in kwargs:
        plt.figure(figsize=kwargs['figsize'])
    else:
        plt.figure(figsize=(5, 5))
    ax = plt.gca()

    if "title" in kwargs.keys():
        if "fontsize" in kwargs.keys():
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])
        else:
            plt.title(kwargs["title"])

    if image_channel == 1:
        if overlay_color.sum() != 0:
            plt.imshow(image_.transpose(1, 2, 0).clip(0, 1))
            plt.axis("off")
        else:
            # plt.imshow(image_.squeeze().clip(0, 1), cmap="gray", vmin=0.0, vmax=1.0)
            # plt.imshow(image_.squeeze().clip(0, 1), cmap="gray", vmin=0.0, vmax=image.max())
            im = plt.imshow(image.squeeze().clip(0, 1), cmap="gray", vmin=0.0, vmax=image.max())
            plt.axis("off")
            # _plot_colorbar(ax=plt.gca(), im=im)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.07)
            plt.colorbar(im, cax=cax, orientation='vertical')
    else:
        plt.imshow(image_.transpose(1, 2, 0).clip(0, 1))
        plt.axis("off")

    for i, player in enumerate(coalition):
        _plot_selected_region_boundary(
            ax, get_selected_region_mask(sqrt_grid_num, player),
            sqrt_grid_num, grid_width, linecolor=linecolors[i%len(linecolors)], **kwargs
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(osp.join(save_folder, f"{save_name}.png"), transparent=True)
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def plot_curves(save_folder, res_dict):
    """plot curves for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
    """
    for key in res_dict.keys():   
        # define the path
        path = os.path.join(save_folder, "{}-curve.png".format(key))
        # plot the fig
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(res_dict[key])) + 1, res_dict[key])
        ax.set(xlabel = 'epoch', ylabel = key, 
            title = '{}\'s curve'.format(key))
        ax.grid()
        plt.tight_layout()
        fig.savefig(path)
        plt.close("all")


def plot_simple_line_chart(data, xlabel, ylabel, title, save_folder, save_name, X=None):
    makedirs(save_folder)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if isinstance(data, torch.Tensor):
        data = data.clone().detach().cpu().numpy()
    if X is None: X = np.arange(len(data))
    plt.plot(X, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def plot_simple_scatter_chart(x, y, xlabel, ylabel, title, save_folder, save_name):
    makedirs(save_folder)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.scatter(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def plot_simple_bar_chart(data, xlabel, ylabel, title, save_folder, save_name, ordering=None, value_label=False):
    makedirs(save_folder)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ordering is None:
        plt.bar(np.arange(len(data)), data)
    else:
        plt.bar(np.arange(len(data)), sorted(data, key=ordering))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if value_label:
        add_value_labels(plt.gca())
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def add_value_labels(ax, spacing=2):
    """Add labels to the end of each bar in a bar chart.

    Reference: <https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart>

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.



def compare_simple_line_chart(data_series, xlabel, ylabel, legends, title, save_folder, save_name, X=None):
    makedirs(save_folder)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    for data, legend in zip(data_series, legends):
        if X is None: X = np.arange(len(data))
        plt.plot(X, data, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def visualize_all_interaction_descending(interactions, save_path="test.png"):
    plt.figure(figsize=(8, 6))
    X = np.arange(1, interactions.shape[0] + 1)
    plt.hlines(0, 0, X.shape[0], linestyles="dotted", colors="red")
    plt.plot(X, interactions[np.argsort(-interactions)])
    plt.xlabel("patterns (with I(S) descending)")
    plt.ylabel("I(S)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


##
def _ax_plot_single_image(ax, image, **kwargs):

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if len(image.shape) not in [2, 3]:
        raise NotImplementedError(f"Cannot visualize a {len(image.shape)}D tensor.")

    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    if len(image.shape) == 2:
        cmap = "gray" if "cmap" not in kwargs else kwargs["cmap"]
        if "vmin" in kwargs and "vmax" in kwargs:
            im = ax.imshow(image, cmap=cmap, vmin=kwargs["vmin"], vmax=kwargs["vmax"])
        else:
            im = ax.imshow(image, cmap=cmap)
        # To adjust the position of the colorbar
        # Reference: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(ax)
        colorbar_size = 5 * min(image.shape[0] / image.shape[1], 1)
        cax = divider.append_axes("right", size=f"{colorbar_size:.2f}%", pad=0.05)
        plt.colorbar(im, ax=ax, cax=cax)
    else:
        image = image.clip(0, 1)
        ax.imshow(image.transpose(1, 2, 0))

    if "axis_off" not in kwargs or kwargs["axis_off"]:
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

    # set the title of this sub-figure
    if "title" in kwargs.keys():
        if "fontsize" in kwargs.keys():
            ax.set_title(kwargs["title"], fontsize=kwargs["fontsize"])
        else:
            ax.set_title(kwargs["title"])


##
def plot_image_batch(image_batch, save_folder, save_name, n_col=5, **kwargs):
    n_image = len(image_batch)
    if n_image == 0:
        return
    n_row = int(np.ceil(n_image / n_col))

    plt.figure(figsize=(n_col * 3.5, n_row * 3.55))

    for i in range(n_image):
        if image_batch[i] is None: continue
        plt.subplot(n_row, n_col, i + 1)
        ax = plt.gca()
        single_kwargs = {}
        if "cmap" in kwargs: single_kwargs["cmap"] = kwargs["cmap"]
        if "title_batch" in kwargs: single_kwargs["title"] = kwargs["title_batch"][i]
        if "title" in kwargs: single_kwargs["title"] = kwargs["title"]
        if "fontsize" in kwargs: single_kwargs["fontsize"] = kwargs["fontsize"]
        if "axis_off" in kwargs: single_kwargs["axis_off"] = kwargs["axis_off"]
        if "vmin" in kwargs: single_kwargs["vmin"] = kwargs["vmin"]
        if "vmax" in kwargs: single_kwargs["vmax"] = kwargs["vmax"]
        _ax_plot_single_image(ax, image_batch[i], **single_kwargs)

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    transparent = "transparent" in kwargs and kwargs["transparent"]
    if "format" in kwargs:
        save_format = kwargs["format"]
        plt.savefig(osp.join(save_folder, f"{save_name}.{save_format}"), transparent=transparent)
    else:
        plt.savefig(osp.join(save_folder, f"{save_name}.png"), transparent=transparent)
    plt.close("all")


##
def plot_single_image(image, save_folder, save_name, figsize=None, **kwargs):
    os.makedirs(save_folder, exist_ok=True)
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    ax = plt.gca()
    _ax_plot_single_image(ax, image, **kwargs)
    plt.tight_layout()
    if "format" in kwargs:
        save_format = kwargs["format"]
        plt.savefig(osp.join(save_folder, f"{save_name}.{save_format}"), dpi=200)
    else:
        plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


##
def _ax_plot_coalition_single_image(ax, image, grid_width, coalition, **kwargs):
    alpha = 0.5 if 'alpha' not in kwargs else kwargs['alpha']
    linecolors = ["r"] if 'linecolors' not in kwargs else kwargs['linecolors']
    if 'linecolor' in kwargs: linecolors = [kwargs['linecolor']]
    linewidth = 4 if 'linewidth' not in kwargs else kwargs['linewidth']
    overlay_color = (0, 0, 0) if 'overlay_color' not in kwargs else kwargs['overlay_color']

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if len(image.shape) not in [2, 3]:
        raise NotImplementedError(f"Cannot visualize a {len(image.shape)}D tensor.")

    if len(image.shape) == 2:
        image = image[None]

    assert len(image.shape) == 3
    assert image.shape[1] == image.shape[2]  # square image

    image_channel = image.shape[0]
    image_width = image.shape[2]
    sqrt_grid_num = int(np.ceil(image_width / grid_width))
    selected_regions_mask = get_selected_region_mask(sqrt_grid_num, flatten(coalition))

    # masked image
    overlay_color = np.array(overlay_color).reshape((-1, 1, 1))
    if image_channel == 1:
        image_ = image.copy()
        image_ = np.concatenate([image_, image_, image_], axis=0)
    else:
        image_ = image.copy()
    for i in range(selected_regions_mask.shape[0]):
        for j in range(selected_regions_mask.shape[1]):
            if selected_regions_mask[i, j] == 0:
                image_[:, i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] *= (1 - alpha)
                image_[:, i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] += overlay_color * alpha

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    # plot image
    plt.imshow(image_.transpose(1, 2, 0).clip(0, 1))
    plt.axis("off")

    # plot coalition boundary
    for i, player in enumerate(coalition):
        player_mask = get_selected_region_mask(sqrt_grid_num, player)
        _plot_selected_region_boundary(
            ax, player_mask, sqrt_grid_num, grid_width,
            linecolor=linecolors[i%len(linecolors)],
            linewidth=linewidth
        )

##
def plot_coalition_single_image(image, grid_width, coalition, save_folder, save_name, **kwargs):
    os.makedirs(save_folder, exist_ok=True)
    if 'figsize' not in kwargs:
        plt.figure()
    else:
        plt.figure(figsize=kwargs['figsize'])
    ax = plt.gca()
    _ax_plot_coalition_single_image(ax, image, grid_width, coalition, **kwargs)
    plt.tight_layout()
    transparent = False if "transparent" not in kwargs else kwargs["transparent"]
    save_format = "png" if "format" not in kwargs else kwargs["format"]
    plt.savefig(osp.join(save_folder, f"{save_name}.{save_format}"),
                dpi=200, transparent=transparent)
    plt.close("all")


##
def plot_coalition_image_batch(image_batch, grid_width, coalition_batch, save_folder, save_name, n_col=5, **kwargs):
    n_image = len(image_batch)
    if n_image == 0:
        return
    n_row = int(np.ceil(n_image / n_col))

    plt.figure(figsize=(n_col * 3.5, n_row * 3.55))

    for i in range(n_image):
        # print(i)
        if image_batch[i] is None: continue
        plt.subplot(n_row, n_col, i + 1)
        ax = plt.gca()
        single_kwargs = {}
        if 'alpha' in kwargs: single_kwargs['alpha'] = kwargs['alpha']
        if 'linecolors' in kwargs: single_kwargs['linecolors'] = kwargs['linecolors']
        if 'linecolor' in kwargs: single_kwargs['linecolor'] = kwargs['linecolor']
        if 'linewidth' in kwargs: single_kwargs['linewidth'] = kwargs['linewidth']
        if 'overlay_color' in kwargs: single_kwargs['overlay_color'] = kwargs['overlay_color']
        if "title_batch" in kwargs: single_kwargs["title"] = kwargs["title_batch"][i]
        if "title" in kwargs: single_kwargs["title"] = kwargs["title"]
        _ax_plot_coalition_single_image(ax, image_batch[i], grid_width, coalition_batch[i], **single_kwargs)

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    if "format" in kwargs:
        save_format = kwargs["format"]
        plt.savefig(osp.join(save_folder, f"{save_name}.{save_format}"))
    else:
        plt.savefig(osp.join(save_folder, f"{save_name}.png"))
    plt.close("all")
    return


def _ax_plot_pattern_activation_grid(ax, attribute_names, pattern_masks, sign_interaction, **kwargs):
    if isinstance(pattern_masks, torch.Tensor):
        pattern_masks = pattern_masks.detach().cpu().numpy()
    if isinstance(sign_interaction, torch.Tensor):
        sign_interaction = sign_interaction.detach().cpu().numpy()

    n_pattern, n_dim = pattern_masks.shape
    assert len(attribute_names) == n_dim
    assert len(sign_interaction) == n_dim
    x = np.arange(n_pattern)
    y = np.arange(n_dim)
    ax.set_xticks(x, [])
    ax.set_yticks(y, attribute_names)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_xlabel(r"index of interactive concepts $S$")
    ax.set_ylabel("attributes")

    patch_colors = {
        True: {
            'pos': 'red',
            'neg': 'blue'
        },
        False: 'gray'
    }
    patch_width = 0.8
    patch_height = 0.9

    for i in range(n_pattern):
        pattern = pattern_masks[i]
        for d in range(n_dim):
            is_selected = pattern[d]
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if sign_interaction[i] > 0:
                    facecolor = patch_colors[is_selected]['pos']
                else:
                    facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(i - patch_width / 2, d - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax.add_patch(rect)

    return ax