import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNIST(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self._is_data_loaded = False

    def _load_data(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.train_set = datasets.MNIST(root=self.data_root, train=True, download=True, transform=transform)
        self.test_set = datasets.MNIST(root=self.data_root, train=False, download=True, transform=transform)
        self._is_data_loaded = True

    def get_dataloader(self, batch_size, shuffle_train=True):
        if not self._is_data_loaded:
            self._load_data()

        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=shuffle_train, num_workers=1, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=1)

        return train_loader, test_loader


# SimpleMnist 没有加归一化，用 0 做 baseline value 更 make sense 一些
class SimpleMNIST(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self._is_data_loaded = False

    def _load_data(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        self.train_set = datasets.MNIST(root=self.data_root, train=True, download=True, transform=transform)
        self.test_set = datasets.MNIST(root=self.data_root, train=False, download=True, transform=transform)
        self._is_data_loaded = True

    def get_dataloader(self, batch_size, shuffle_train=True):
        if not self._is_data_loaded:
            self._load_data()

        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  shuffle=shuffle_train, num_workers=1, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=1)

        return train_loader, test_loader



if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mnist_dataset = MNIST("/data2/lmj/data")
    train_loader, test_loader = mnist_dataset.get_dataloader(batch_size=64)
    print(len(train_loader.dataset), len(test_loader.dataset))

    for images, _ in test_loader:
        image = images[0]
        print(image.shape)
        image = (image.squeeze().numpy() * 0.5) + 0.5

        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.savefig("mnist-test.png")


        exit()

