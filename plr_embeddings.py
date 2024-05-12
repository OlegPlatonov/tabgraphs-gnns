"""
Adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/main/lib/deep.py.
"""

import torch
from torch import nn


class PeriodicEmbeddings(nn.Module):
    def __init__(self, num_features, num_frequencies, frequency_scale):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(num_features, num_frequencies) * frequency_scale)

    def forward(self, x):
        x = 2 * torch.pi * self.frequencies[None, ...] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)

        return x


class NLinear(nn.Module):
    def __init__(self, num_features, input_dim, output_dim, bias=True):
        super().__init__()

        init_max = 1 / input_dim ** 0.5
        self.weight = nn.Parameter(torch.Tensor(num_features, input_dim, output_dim).uniform_(-init_max, init_max))
        self.bias = nn.Parameter(torch.Tensor(num_features, output_dim).uniform_(-init_max, init_max)) if bias else None

    def forward(self, x):
        x = (x[..., None] * self.weight[None, ...]).sum(axis=2)
        if self.bias is not None:
            x = x + self.bias[None, ...]

        return x


class PLREmbeddings(nn.Module):
    def __init__(self, num_features, num_frequencies, frequency_scale, embedding_dim, lite=False):
        super().__init__()

        linear_layer = nn.Linear(in_features=num_frequencies * 2, out_features=embedding_dim) if lite \
            else NLinear(num_features=num_features, input_dim=num_frequencies * 2, output_dim=embedding_dim)

        self.plr_embeddings = nn.Sequential(
            PeriodicEmbeddings(num_features=num_features, num_frequencies=num_frequencies,
                               frequency_scale=frequency_scale),
            linear_layer,
            nn.ReLU()
        )

    def forward(self, x):
        return self.plr_embeddings(x)
