import time
import torch
from torch import nn, optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation="relu"):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        if in_dim == out_dim:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=bias)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.linear(x)
        if self.shortcut is None:
            out = out + x
        else:
            out = out + self.shortcut(x)
        out = self.activation(out)
        return out

    def forward_wo_relu(self, x):
        out = self.linear(x)
        if self.shortcut is None:
            out = out + x
        else:
            out = out + self.shortcut(x)
        return out


class ResMLP_Sigmoid(nn.Module):
    def __init__(self, in_dim, hidd_dim, out_dim, n_layer):
        super(ResMLP_Sigmoid, self).__init__()

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.layers = self._make_layers(in_dim, hidd_dim, out_dim, n_layer)

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
        for i in range(n_layer - 2):
            if i == n_layer - 3:
                layers.append(ResidualBlock(hidd_dim, hidd_dim, activation="sigmoid"))
            else:
                layers.append(ResidualBlock(hidd_dim, hidd_dim, activation="relu"))
        layers.append(nn.Linear(hidd_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)


# ===========================
#   wrapper
# ===========================
def resmlp5_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=5)


def resmlp2_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=2)


def resmlp3_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=3)


def resmlp4_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=4)


def resmlp6_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=6)


def resmlp7_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=7)


def resmlp8_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=8)


def resmlp9_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=9)


def resmlp10_sigmoid(in_dim, hidd_dim, out_dim):
    return ResMLP_Sigmoid(in_dim, hidd_dim, out_dim, n_layer=10)



if __name__ == '__main__':
    x = torch.rand(1000, 10)
    net = resmlp5_sigmoid(in_dim=10, hidd_dim=100, out_dim=2)
    print(net)
    print(net(x).shape)
