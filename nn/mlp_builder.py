import torch.nn as nn


def build_mlp(dims, act, norm=None):
    layers = []

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))

        if i != len(dims) - 2:
            if norm is not None:
                layers.append(norm(dims[i + 1]))
            layers.append(act())

    return nn.Sequential(*layers)
