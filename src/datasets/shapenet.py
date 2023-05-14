import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, List, Callable
import os
import os.path as osp
import json
import numpy as np


class ShapeNetClsDataset(Dataset):
    """
        Reference: https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/dataset.py
    """
    def __init__(
            self,
            data_root: str,
            n_points: int = 2500,
            class_choice: Optional[List] = None,
            split: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(ShapeNetClsDataset, self).__init__()

        self.n_points = n_points
        self.transform = transform
        self.target_transform = target_transform
        self.data_root = data_root
        self.catfile = osp.join(data_root, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(data_root, 'train_test_split', f'shuffled_{split}_file_list.json')
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(
                    [os.path.join(data_root, category, 'points', f'{uuid}.pts'),
                     os.path.join(data_root, category, 'points_label', f'{uuid}.seg')]
                )

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # print(self.classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        choice = np.random.choice(len(point_set), self.n_points, replace=True)
        # resample
        point_set = point_set[choice, :]

        # normalize
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), axis=0)
        point_set = point_set / dist  # scale

        point_set = torch.from_numpy(point_set)
        if self.transform is not None:
            point_set = self.transform(point_set)
        point_set = point_set.transpose(0, 1).float()

        if self.target_transform is not None:
            cls = self.target_transform(cls)

        return point_set, cls

    def __len__(self):
        return len(self.datapath)


class RandomRotate3d(nn.Module):
    def __init__(self, rotation_range):
        super(RandomRotate3d, self).__init__()
        self.rotation_range = rotation_range

    def forward(self, point_set):
        """
        Randomly rotate a point cloud
        :param point_set: torch.Size([n_points, 3])
        :return: rotated point cloud
        """
        theta_1 = np.random.uniform(*self.rotation_range)
        theta_2 = np.random.uniform(*self.rotation_range)
        theta_3 = np.random.uniform(*self.rotation_range)

        rot_mat_1 = np.array([
            [1,               0,                0],
            [0, np.cos(theta_1), -np.sin(theta_1)],
            [0, np.sin(theta_1),  np.cos(theta_1)]
        ])
        rot_mat_2 = np.array([
            [ np.cos(theta_2), 0, np.sin(theta_2)],
            [               0, 1,               0],
            [-np.sin(theta_2), 0, np.cos(theta_2)]
        ])
        rot_mat_3 = np.array([
            [np.cos(theta_3), -np.sin(theta_3), 0],
            [np.sin(theta_3),  np.cos(theta_3), 0],
            [              0,                0, 1]
        ])
        rot_mat = torch.from_numpy(rot_mat_1 @ rot_mat_2 @ rot_mat_3).float()
        point_set = point_set @ rot_mat
        return point_set

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(theta={self.rotation_range})"


class RandomRotateXZ(nn.Module):
    def __init__(self, rotation_range):
        super(RandomRotateXZ, self).__init__()
        self.rotation_range = rotation_range

    def forward(self, point_set):
        """
        Randomly rotate a point cloud
        :param point_set: torch.Size([n_points, 3])
        :return: rotated point cloud
        """
        theta = np.random.uniform(*self.rotation_range)

        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rot_mat = torch.from_numpy(rot_mat).float()
        point_set[:, [0, 2]] = point_set[:, [0, 2]] @ rot_mat
        return point_set

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(theta={self.rotation_range})"



class RandomTranslate3d(nn.Module):
    def __init__(self, sigma):
        """
        translation noise: N(0, s^2)
        :param sigma: standard deviation of the translation
        """
        super(RandomTranslate3d, self).__init__()
        self.sigma = sigma

    def forward(self, point_set):
        """
        Randomly translate a point cloud
        :param point_set: torch.Size([n_points, 3])
        :return: translated point cloud
        """
        point_set += np.random.normal(0, self.sigma, size=point_set.shape)
        return point_set

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s={self.sigma})"


class ShapeNet(object):
    def __init__(self, data_root, n_points=2500, augmentation=True):
        self.data_root = osp.join(data_root, "shapenetcore_partanno_segmentation_benchmark_v0")
        self.n_points = n_points
        self.augmentation = augmentation
        self._is_data_loaded = False

    def _load_data(self):
        if self.augmentation:
            train_transform = transforms.Compose([
                RandomRotateXZ(rotation_range=(0, 0.5 * np.pi)),
                RandomTranslate3d(sigma=0.02)
            ])
        else:
            train_transform = None
        test_transform = None
        self.train_set = ShapeNetClsDataset(self.data_root, n_points=self.n_points,
                                            split='train', transform=train_transform)
        self.test_set = ShapeNetClsDataset(self.data_root, n_points=self.n_points,
                                           split='test', transform=test_transform)
        self._is_data_loaded = True

    def get_dataloader(self, batch_size):
        if not self._is_data_loaded:
            self._load_data()

        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=4)

        return train_loader, test_loader


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def visualize_easy(data, save_path):
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        fig = plt.figure(figsize=(15, 15), dpi=50)
        plt.subplot(221)
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c='gray', s=20, alpha=0.7)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()
        plt.savefig(save_path, dpi=200)
        plt.close("all")

    data_root = "/data2/lmj/data/ShapeNet-raw/ShapeNetPart"

    shapenet = ShapeNet(data_root)
    train_loader, test_loader = shapenet.get_dataloader(batch_size=1)
    print(shapenet.train_set.classes.__len__())
    print(shapenet.train_set.classes)
    print(shapenet.test_set.classes)

    for i, (points, target) in enumerate(train_loader):
        print(points.shape, target.shape)
        visualize_easy(points.squeeze().numpy(), "try-vis.png")
        exit()