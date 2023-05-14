import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Sigmoid(nn.Module):

    def __init__(self, in_dim, hidd_dim, out_dim, n_layer):
        super(MLP_Sigmoid, self).__init__()

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.layers = self._make_layers(in_dim, hidd_dim, out_dim, n_layer)

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
        for i in range(n_layer - 2):
            if i == n_layer - 3:
                layers.extend([nn.Linear(hidd_dim, hidd_dim), nn.Sigmoid()])
            else:
                layers.extend([nn.Linear(hidd_dim, hidd_dim), nn.ReLU()])
        layers.append(nn.Linear(hidd_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        return self.layers(x)


# ===========================
#   wrapper
# ===========================
def mlp5_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=5)


def mlp2_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=2)


def mlp3_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=3)


def mlp4_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=4)


def mlp6_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=6)


def mlp7_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=7)


def mlp8_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=8)


def mlp9_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=9)


def mlp10_sigmoid(in_dim, hidd_dim, out_dim):
    return MLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=10)



if __name__ == '__main__':
    x = torch.rand(1000,10)
    net = mlp5_sigmoid(in_dim=10, hidd_dim=100, out_dim=2)
    print(net)
    print(net(x).shape)