"""
Copied from https://github.com/yandex-research/tabular-dl-tabr/blob/main/lib/deep.py.
"""

import torch
from torch import nn


class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
        self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, d_in, d_out))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PLREmbeddings(nn.Sequential):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )
