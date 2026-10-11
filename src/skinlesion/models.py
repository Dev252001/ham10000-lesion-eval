"""
models.py – model definitions and utilities.

Public API
----------
BaselineCNN        : 4-block CNN (32-64-128-256 channels)
build_model        : factory for baseline_cnn / resnet18 / efficientnet_b0
count_params       : total trainable parameter count
compute_class_weights : N / (C * n_c), computed from train partition only
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from skinlesion.data import CLASSES

NUM_CLASSES: int = len(CLASSES)  # 7


# ---------------------------------------------------------------------------
# BaselineCNN
# ---------------------------------------------------------------------------


class BaselineCNN(nn.Module):
    """Four-block convolutional network for 7-class skin-lesion classification.

    Architecture
    ------------
    Block 1: Conv(3->32, 3x3, pad=1) - BN - ReLU - MaxPool(2)
    Block 2: Conv(32->64, 3x3, pad=1) - BN - ReLU - MaxPool(2)
    Block 3: Conv(64->128, 3x3, pad=1) - BN - ReLU - MaxPool(2)
    Block 4: Conv(128->256, 3x3, pad=1) - BN - ReLU - MaxPool(2)
    Global Average Pooling -> Dropout(0.5) -> Linear(256, 7)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Build a model by name with a 7-class output head.

    Supported names: ``baseline_cnn``, ``resnet18``, ``efficientnet_b0``.

    Parameters
    ----------
    name:
        One of the three supported model names.
    pretrained:
        If ``True``, load ImageNet weights (requires network access).
        If ``False``, initialise randomly (weights=None in torchvision).
        Tests should always pass ``pretrained=False``.

    Returns
    -------
    nn.Module with ``num_classes=7`` output.

    Raises
    ------
    ValueError if *name* is not recognised.
    """
    if name == "baseline_cnn":
        return BaselineCNN()

    if name == "resnet18":
        import torchvision.models as tvm
        weights = "DEFAULT" if pretrained else None
        model = tvm.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model

    if name == "efficientnet_b0":
        import torchvision.models as tvm
        weights = "DEFAULT" if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        return model

    raise ValueError(
        f"Unknown model name {name!r}. "
        f"Choose from: baseline_cnn, resnet18, efficientnet_b0."
    )


# ---------------------------------------------------------------------------
# count_params
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Compute per-class weights as N / (C * n_c) from the training partition.

    This is the standard inverse-frequency formula:
    - N  = total number of training images
    - C  = number of classes (7)
    - n_c = number of training images for class c

    Weights are ordered according to ``skinlesion.data.CLASSES`` (sorted).

    Parameters
    ----------
    train_df:
        Training-partition DataFrame (must have a ``dx`` column).
        Must NOT include val or test rows.

    Returns
    -------
    torch.Tensor of shape (7,) with dtype float32.
    """
    n_total = len(train_df)
    n_classes = len(CLASSES)
    weights = []
    for cls in CLASSES:
        n_c = int((train_df["dx"] == cls).sum())
        if n_c == 0:
            # Class absent from training split; assign weight 0.0 so it
            # does not interfere (caller should be warned separately via
            # splits.py's MIN_LESIONS_WARN mechanism).
            weights.append(0.0)
        else:
            weights.append(n_total / (n_classes * n_c))
    return torch.tensor(weights, dtype=torch.float32)