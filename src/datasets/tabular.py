import os
import os.path as osp
import re

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import json


class ClassBalancedSampler(Sampler):
    def __init__(self, data_source: TensorDataset, num_samples=None):
        super(ClassBalancedSampler, self).__init__(data_source)
        self.num_samples = len(data_source) if num_samples is None else num_samples

        targets = [data_source.__getitem__(i)[1].item() for i in range(len(data_source))]
        label_count = [0] * len(np.unique(targets))
        for idx in range(len(data_source)):
            label = targets[idx]
            label_count[label] += 1

        weights_per_cls = 1.0 / np.array(label_count)
        weights = [weights_per_cls[targets[idx]]
                   for idx in range(len(data_source))]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True
        ).tolist())

    def __len__(self):
        return self.num_samples


def _is_classification_dataset(dataset_name):
    return dataset_name in ["census", "commercial", "yeast", "wine", "glass", "telescope",
                            "tictactoe", "raisin", "phishing_binary", "wifi"] \
           or re.match(r"binary_rule_(.+)_classification_(.+)", dataset_name) \
           or re.match(r"uniform_1_rule_(.+)_classification_(.+)", dataset_name) \
           or re.match(r"gaussian_rule_(.+)_classification_(.+)", dataset_name) \
           or re.match(r"wifi_corrupt_data_(.+)", dataset_name) \
           or re.match(r"wifi_corrupt_label_(.+)", dataset_name)


def _is_regression_dataset(dataset_name):
    return dataset_name in ["bike"] \
           or re.match(r"gaussian_rule_(.+)_regression_(.+)", dataset_name) \
           or re.match(r"census_rule_(.+)_regression_(.+)", dataset_name) \
           or re.match(r"binary_rule_(.+)_regression_(.+)", dataset_name) \
           or re.match(r"zero_one_rule_(.+)_regression_(.+)", dataset_name) \
           or re.match(r"uniform_2_rule_(.+)_regression_(.+)", dataset_name)


class TabularDataset(object):
    def __init__(self, data_root, dataset_name):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self._is_data_loaded = False

    def _load_data(self):
        data_folder = osp.join(self.data_root, self.dataset_name)
        X_train = np.load(osp.join(data_folder, "X_train.npy"))
        y_train = np.load(osp.join(data_folder, "y_train.npy"))
        X_test = np.load(osp.join(data_folder, "X_test.npy"))
        y_test = np.load(osp.join(data_folder, "y_test.npy"))

        if _is_classification_dataset(self.dataset_name):
            self.X_train = torch.from_numpy(X_train).float()
            self.y_train = torch.from_numpy(y_train).long()
            self.X_test = torch.from_numpy(X_test).float()
            self.y_test = torch.from_numpy(y_test).long()
        elif _is_regression_dataset(self.dataset_name):
            self.X_train = torch.from_numpy(X_train).float()
            self.y_train = torch.from_numpy(y_train).float()
            self.X_test = torch.from_numpy(X_test).float()
            self.y_test = torch.from_numpy(y_test).float()
        else:
            raise NotImplementedError(f"Please setup dataset: {self.dataset_name} in [TabularDataset].")

        self._is_data_loaded = True

    def get_data(self):
        if not self._is_data_loaded:
            self._load_data()
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_dataloader(self, batch_size, balance=False, shuffle_train=True):
        if not self._is_data_loaded:
            self._load_data()
        # create dataloader
        train_set = TensorDataset(self.X_train, self.y_train)
        test_set = TensorDataset(self.X_test, self.y_test)

        if balance:
            print("################## Use balanced dataset ##################")
            sampler = ClassBalancedSampler(train_set)
            assert shuffle_train, f"Sampler is incompatible with the current 'shuffle_train' value: {shuffle_train}"
            train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, sampler=sampler)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def get_attributes(self):
        with open(osp.join(self.data_root, self.dataset_name, "info.json"), "r") as f:
            dataset_info = json.load(f)

        return dataset_info["attributes"]

    def get_data_mean_std(self):
        with open(osp.join(self.data_root, self.dataset_name, "info.json"), "r") as f:
            dataset_info = json.load(f)
        return np.array(dataset_info['X_mean_original']), np.array(dataset_info['X_std_original'])

    def get_rule(self):
        rule_path = osp.join(self.data_root, self.dataset_name, "rule.json")
        if not osp.exists(rule_path):
            raise Exception(f"No rule exists for dataset {self.dataset_name}")
        with open(rule_path, "r") as f:
            rule = json.load(f)
        return rule


class TabularDatasetCorrupted(TabularDataset):
    def __init__(self, data_root, dataset_name, corrupted_ratio):
        super(TabularDatasetCorrupted, self).__init__(data_root=data_root, dataset_name=dataset_name)
        assert 0 <= corrupted_ratio <= 1, f"Corrupted ratio should be in [0, 1], current: {corrupted_ratio}"
        self.corrupted_ratio = corrupted_ratio

    def _load_data(self):
        data_folder = osp.join(self.data_root, self.dataset_name)
        X_train = np.load(osp.join(data_folder, "X_train.npy"))
        y_train = np.load(osp.join(data_folder, "y_train.npy"))
        X_test = np.load(osp.join(data_folder, "X_test.npy"))
        y_test = np.load(osp.join(data_folder, "y_test.npy"))

        if _is_classification_dataset(self.dataset_name):
            self.X_train = torch.from_numpy(X_train).float()
            self.y_train = torch.from_numpy(y_train).long()
            self.X_test = torch.from_numpy(X_test).float()
            self.y_test = torch.from_numpy(y_test).long()
        else:
            raise NotImplementedError(f"Please setup dataset: {self.dataset_name} in [TabularDataset].\n"
                                      f"Note: we have only implemented data corruption for classification datasets.")

        # do data corruption here
        print(f"Load data and corrupt the labels of {self.corrupted_ratio:.2%} training data")
        self.y_train_clean = self.y_train.clone()
        # print(self.y_train)
        n_category = torch.unique(self.y_train).shape[0]
        n_train = self.y_train.shape[0]
        self.y_train[:int(self.corrupted_ratio * n_train)] += 1
        self.y_train[:int(self.corrupted_ratio * n_train)] %= n_category
        # print(n_category)
        # print(self.y_train)
        r = torch.mean((self.y_train != self.y_train_clean).float()).item()
        print(f"Finish, corrupted: {r:.2%}")

        self._is_data_loaded = True


if __name__ == '__main__':
    corrupted_wifi = TabularDatasetCorrupted(
        data_root="/data2/lmj/data/tabular",
        dataset_name="wifi",
        corrupted_ratio=0.5
    )

    corrupted_wifi._load_data()