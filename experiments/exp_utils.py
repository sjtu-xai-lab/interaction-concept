import torch
import numpy as np
import os
import os.path as osp
import re
import functools
from pprint import pprint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bisect import bisect
from typing import List, Tuple, Callable
import json
from tqdm import tqdm


def load_interactions(dataset, result_folder, selected_classes=None):
    if dataset in ["wifi", "tictactoe"]:
        interaction_dict = _load_interactions_tabular(result_folder, selected_classes)
    elif dataset in ["simpleisthree", "celeba_eyeglasses", "bg_bird", "redbg_bluebird"]:
        interaction_dict = _load_interactions_image(result_folder, selected_classes)
    elif dataset in ["shapenet"]:
        interaction_dict = _load_interactions_pointcloud(result_folder, selected_classes)
    else:
        raise NotImplementedError
    return interaction_dict


def _load_interactions_tabular(result_folder, selected_classes=None):
    all_interactions = {}  # cluster_name -> sample_name -> [I_and, I_or]
    if selected_classes is None:
        class_names = [folder for folder in os.listdir(result_folder)
                       if re.compile(r"class_(.+)").match(folder)]
    else:
        class_names = selected_classes
    class_names = sorted(class_names)
    for class_name in class_names:
        all_interactions[class_name] = {}
        sample_names = [folder for folder in os.listdir(osp.join(result_folder, class_name))
                        if re.compile(r"sample_(.+)").match(folder)]
        sample_names = sorted(sample_names)
        pbar = tqdm(sample_names, desc="Loading", ncols=100)
        for sample_name in pbar:
            pbar.set_postfix_str(f"LOADING {sample_name} FROM {class_name}")
            all_interactions[class_name][sample_name] = [
                torch.load(osp.join(result_folder, class_name, sample_name, "I_and.pth"),
                           map_location=torch.device("cpu")),
                torch.load(osp.join(result_folder, class_name, sample_name, "I_or.pth"),
                           map_location=torch.device("cpu"))
            ]
    return all_interactions


def _load_interactions_image(result_folder, selected_classes=None):
    all_interactions = {}  # cluster_name -> sample_name -> [I_and, I_or]
    if selected_classes is None:
        class_names = [folder for folder in os.listdir(result_folder)
                         if re.compile(r"class_(.+)").match(folder)]
    else:
        class_names = selected_classes
    class_names = sorted(class_names)
    for class_name in class_names:
        all_interactions[class_name] = {}
        sample_names = [folder for folder in os.listdir(osp.join(result_folder, class_name))
                        if re.compile(r"sample_(.+)").match(folder)]
        sample_names = sorted(sample_names)
        pbar = tqdm(sample_names, ncols=100, desc="loading")
        for sample_name in pbar:
            pbar.set_postfix_str(f"--> LOADING {sample_name} FROM {class_name}")
            all_interactions[class_name][sample_name] = [
                torch.load(osp.join(result_folder, class_name, sample_name, "I_and.pth"),
                           map_location=torch.device("cpu")),
                torch.load(osp.join(result_folder, class_name, sample_name, "I_or.pth"),
                           map_location=torch.device("cpu"))
            ]
    return all_interactions


def _load_interactions_pointcloud(result_folder, selected_classes=None):
    all_interactions = {}  # cluster_name -> sample_name -> [I_and, I_or]
    if selected_classes is None:
        class_names = [folder for folder in os.listdir(result_folder) if folder in ["motorbike"]]
    else:
        class_names = selected_classes
    class_names = sorted(class_names)
    for class_name in class_names:
        all_interactions[class_name] = {}
        sample_names = [folder for folder in os.listdir(osp.join(result_folder, class_name))
                        if re.compile(r"sample_(.+)").match(folder)]
        sample_names = sorted(sample_names)
        pbar = tqdm(sample_names, ncols=100, desc="loading")
        for sample_name in pbar:
            pbar.set_postfix_str(f"--> LOADING {sample_name} FROM {class_name}")
            all_interactions[class_name][sample_name] = [
                torch.load(osp.join(result_folder, class_name, sample_name, "I_and.pth"),
                           map_location=torch.device("cpu")),
                torch.load(osp.join(result_folder, class_name, sample_name, "I_or.pth"),
                           map_location=torch.device("cpu"))
            ]
    return all_interactions


def judge_tictactoe_pattern_id(sample: torch.Tensor):
    if sample[0] == 1. and sample[1] == 1. and sample[2] == 1.:
        return 1
    elif sample[3] == 1. and sample[4] == 1. and sample[5] == 1.:
        return 2
    elif sample[6] == 1. and sample[7] == 1. and sample[8] == 1.:
        return 3
    elif sample[0] == 1. and sample[3] == 1. and sample[6] == 1.:
        return 4
    elif sample[1] == 1. and sample[4] == 1. and sample[7] == 1.:
        return 5
    elif sample[2] == 1. and sample[5] == 1. and sample[8] == 1.:
        return 6
    elif sample[0] == 1. and sample[4] == 1. and sample[8] == 1.:
        return 7
    elif sample[2] == 1. and sample[4] == 1. and sample[6] == 1.:
        return 8
    raise NotImplementedError


def filter_raw_interaction_dict_tictactoe(raw_interactions, X_all, selected_patterns=None):
    name_to_id = lambda s: int(s.split("_")[-1])
    if selected_patterns is None:
        judge = lambda x: True
    else:
        judge = lambda x: x in selected_patterns
    filtered_dict = {}
    for cluster_name in raw_interactions.keys():
        for sample_name in raw_interactions[cluster_name].keys():
            if selected_patterns is not None:
                sample_id = name_to_id(sample_name)
                pattern_id = judge_tictactoe_pattern_id(X_all[sample_id])
            else:
                pattern_id = None
            if not judge(pattern_id): continue

            filtered_dict[f"{cluster_name}_{sample_name}"] = raw_interactions[cluster_name][sample_name]

    return filtered_dict


def get_significant_threshold(I_vector: torch.Tensor, significant_ratio: float = 0.01):
    max_strength = torch.abs(I_vector).max()
    return significant_ratio * max_strength


def judge_pattern_type_single_sample(I_vector: torch.Tensor, significant_threshold: float):
    type_of_patterns = torch.zeros_like(I_vector).long()
    type_of_patterns[I_vector >= significant_threshold] = 1
    type_of_patterns[I_vector <= -significant_threshold] = -1
    return type_of_patterns


def filter_raw_interaction_dict(raw_interactions, selected_clusters=None):
    """
    filter the raw interaction dict according to `selected_clusters`
    :param raw_interactions: cluster_name -> sample_name -> [I_and, I_or]
    :param selected_clusters: List or None
    :return:
    """
    if selected_clusters is None:
        judge = lambda x: True
    else:
        judge = lambda x: x in selected_clusters
    filtered_dict = {}
    for cluster_name in raw_interactions.keys():
        if not judge(cluster_name): continue
        for sample_name in raw_interactions[cluster_name].keys():
            filtered_dict[f"{cluster_name}_{sample_name}"] = raw_interactions[cluster_name][sample_name]
    return filtered_dict


def get_overlap(i_1, i_2, eps=1e-7, normalize=False):
    """
    The overlap metric is defined as follows, which is similar to IoU,
        overlap(I1, I2) := numerator / denorminator
          -> numerator := \sum_S min(|I1^+(S)|, |I2^+(S)|) + min(|I1^-(S)|, |I2^-(S)|)
          -> denominator := \sum_S max(|I1^+(S)|, |I2^+(S)|) + max(|I1^-(S)|, |I2^-(S)|)
    :param i_1:
    :param i_2:
    :param eps:
    :return:
    """
    if normalize:
        i_1 = i_1 / (torch.sum(torch.abs(i_1)) + eps)
        i_2 = i_2 / (torch.sum(torch.abs(i_2)) + eps)
    i_1_pos = torch.clamp(i_1, min=0)
    i_1_neg = - torch.clamp(i_1, max=0)
    i_2_pos = torch.clamp(i_2, min=0)
    i_2_neg = - torch.clamp(i_2, max=0)
    numerator = torch.minimum(i_1_pos, i_2_pos).sum() + torch.minimum(i_1_neg, i_2_neg).sum()
    denominator = torch.maximum(i_1_pos, i_2_pos).sum() + torch.maximum(i_1_neg, i_2_neg).sum()
    return numerator / (denominator + eps)


def get_cover_ratio(i_1, i_2, eps=1e-7):
    """
    用 i_2 来 cover i_1
    :param i_1:
    :param i_2:
    :param eps:
    :return:
    """
    i_1 = torch.abs(i_1)
    i_2 = torch.abs(i_2)
    iou = torch.minimum(i_1, i_2).sum() / (i_1.sum() + eps)
    return iou

