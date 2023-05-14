import os
import os.path as osp
import torch
import torchvision
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import PIL
from typing import Union, Optional, List, Callable, Tuple, Any
from tqdm import tqdm


class BgBird(object):
    def __init__(self, data_root, augmentation=True):
        self.data_root = data_root
        self.augmentation = augmentation
        self._is_data_loaded = False

    def _load_data(self):
        if self.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(224, 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.train_set = ImageFolder(osp.join(self.data_root, "bg-bird", "train"), train_transform)
        self.test_set = ImageFolder(osp.join(self.data_root, "bg-bird", "test"), test_transform)
        self._is_data_loaded = True

    def get_dataloader(self, batch_size):
        if not self._is_data_loaded:
            self._load_data()

        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=1)

        return train_loader, test_loader


class RedBgBlueBird(object):
    def __init__(self, data_root, augmentation=True):
        self.data_root = data_root
        self.augmentation = augmentation
        self._is_data_loaded = False

    def _load_data(self):
        if self.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(224, 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.train_set = ImageFolder(osp.join(self.data_root, "redbg-bluebird", "train"), train_transform)
        self.test_set = ImageFolder(osp.join(self.data_root, "redbg-bluebird", "test"), test_transform)
        self._is_data_loaded = True

    def get_dataloader(self, batch_size):
        if not self._is_data_loaded:
            self._load_data()

        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=1)

        return train_loader, test_loader



if __name__ == '__main__':
    import os
    import os.path as osp
    import matplotlib.pyplot as plt
    import sys
    sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
    from common_harsanyi.src.tools.plot import denormalize_image
    os.makedirs("tmp", exist_ok=True)

    bgbird_dataset = BgBird("/data2/lmj/data/")

    train_loader, test_loader = bgbird_dataset.get_dataloader(batch_size=512)
    print(len(train_loader.dataset), len(test_loader.dataset), train_loader.dataset.class_to_idx)