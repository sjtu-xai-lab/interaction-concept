import os
import torch
import torchvision
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import transforms
import numpy as np
import PIL
from typing import Union, Optional, List, Callable, Tuple, Any
from tqdm import tqdm


CelebA_attrs = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
]


class CelebA_binary_dataset(CelebA):
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_attribute: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(CelebA_binary_dataset, self).__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
        )
        self.target_attribute = target_attribute
        self.target_attr_id = self._get_target_attr_id(target_attribute)

    def _get_target_attr_id(self, target_attribute):
        for attr_id, attr in enumerate(self.attr_names):
            if target_attribute == attr.lower():
                return attr_id
        raise Exception(f"Cannot find the attribute: {target_attribute}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        assert len(self.target_type) == 1 and self.target_type[0] == "attr"
        target = self.attr[index, :][self.target_attr_id]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def get_label(self, index: int) -> Any:  # used to speed up the initialization of sampler
        assert len(self.target_type) == 1 and self.target_type[0] == "attr"
        target = self.attr[index, :][self.target_attr_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target


class ClassBalancedSampler(Sampler):
    def __init__(self, data_source: CelebA_binary_dataset, num_samples=None):
        super(ClassBalancedSampler, self).__init__(data_source)
        self.num_samples = len(data_source) if num_samples is None else num_samples

        targets = [data_source.get_label(i) for i in range(len(data_source))]  # to accelerate
        if isinstance(targets[0], torch.Tensor):
            targets = [target.item() for target in targets]

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


class CelebA_binary(object):
    def __init__(self, data_root, target_attribute, augmentation=True):
        self.data_root = data_root
        assert target_attribute in [attr_name.lower() for attr_name in CelebA_attrs]
        self.target_attribute = target_attribute
        self.augmentation = augmentation
        self._is_data_loaded = False

    def _load_data(self):
        if self.augmentation:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
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
        self.train_set = CelebA_binary_dataset(root=self.data_root, split="train",
                                               target_attribute=self.target_attribute,
                                               transform=train_transform)
        self.test_set = CelebA_binary_dataset(root=self.data_root, split="test",
                                              target_attribute=self.target_attribute,
                                              transform=test_transform)
        self._is_data_loaded = True

    def get_dataloader(self, batch_size):
        if not self._is_data_loaded:
            self._load_data()

        sampler = ClassBalancedSampler(self.train_set)
        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  num_workers=1, drop_last=True, sampler=sampler)
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

    celeba_dataset = CelebA_binary("/data2/lmj/data", "big_nose")

    train_loader, test_loader = celeba_dataset.get_dataloader(batch_size=512)
    print(len(train_loader.dataset), len(test_loader.dataset))
    for images, labels in train_loader:
        print("ratio of positive samples", labels.float().mean())
        break

    train_loader, test_loader = celeba_dataset.get_dataloader(batch_size=1)
    for i, (image, label) in enumerate(train_loader):
        # print(image.shape)
        # print(label.shape)
        # print(label)

        plt.title(f"{celeba_dataset.target_attribute}: {label.item()}")
        image = denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        plt.imshow(image.squeeze(0).numpy().transpose(1, 2, 0))
        plt.savefig(f"./tmp/test-celeba-{i:>03d}.png")
        if i == 20: break
