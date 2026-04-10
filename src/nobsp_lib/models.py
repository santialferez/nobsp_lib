"""Simple models that match the tuple conventions expected by NObSP."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _build_mlp(input_dim: int, hidden_dims: Iterable[int]) -> nn.Sequential:
    layers = []
    in_features = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        in_features = hidden_dim
    return nn.Sequential(*layers)


class TabularRegressorNet(nn.Module):
    """Small MLP regressor with the `(prediction, activations)` output format."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        self.features = _build_mlp(input_dim, hidden_dims)
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = self.features(x)
        prediction = self.head(activations)
        return prediction, activations


class TabularClassifierNet(nn.Module):
    """Small MLP classifier with the `(probabilities, activations, logits)` format."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.features = _build_mlp(input_dim, hidden_dims)
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.head = nn.Linear(last_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activations = self.features(x)
        logits = self.head(activations)
        probabilities = F.softmax(logits, dim=1)
        return probabilities, activations, logits


class SmallConvNet(nn.Module):
    """Small CNN with `features` and `classifier`, compatible with NObSPVision."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
