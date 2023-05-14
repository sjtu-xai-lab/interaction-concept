import os
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ClassBalancedSampler(Sampler):
    def __init__(self, data_source: Dataset, num_samples=None):
        super(ClassBalancedSampler, self).__init__(data_source)
        self.num_samples = len(data_source) if num_samples is None else num_samples

        targets = [data_source[i][1] for i in range(len(data_source))]
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


# MNIST dataset for One-VS-Rest
class SimpleMNIST_OVR(object):
    def __init__(self, data_root, target_class):
        self.data_root = data_root
        assert target_class in list(range(10))
        self.target_class = target_class
        self._is_data_loaded = False

    def _load_data(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        target_transform = lambda y: 1 if y == self.target_class else 0
        self.train_set = datasets.MNIST(root=self.data_root, train=True, download=True,
                                        transform=transform, target_transform=target_transform)
        self.test_set = datasets.MNIST(root=self.data_root, train=False, download=True,
                                       transform=transform, target_transform=target_transform)
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


class SimpleIsThree(SimpleMNIST_OVR):
    def __init__(self, data_root):
        super(SimpleIsThree, self).__init__(data_root=data_root, target_class=3)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("tmp", exist_ok=True)

    isthree_dataset = SimpleIsThree("/data2/lmj/data")
    train_loader, test_loader = isthree_dataset.get_dataloader(batch_size=512)
    print(len(train_loader.dataset), len(test_loader.dataset))

    for images, labels in train_loader:
        print(labels.shape)
        print("Ratio of positive samples:", torch.mean(labels.float()))
        break

    for images, labels in test_loader:

        for i in range(20):
            image = images[i]
            label = labels[i].item()
            image = image.squeeze().numpy()

            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title(f"label: {label}")
            plt.colorbar()
            plt.savefig(f"./tmp/isthree-test-{i}.png")
            plt.close("all")

        break
