"""
transforms.py – image pre-processing pipelines for HAM10000.

ImageNet mean/std are fixed constants, not derived from the dataset.
Using the ImageNet statistics is standard practice for torchvision
pretrained models and is appropriate here because we fine-tune those
models and because the HAM10000 images are natural-ish colour photographs.

Public API
----------
IMAGENET_MEAN  : tuple[float, float, float]
IMAGENET_STD   : tuple[float, float, float]
train_transform(image_size) -> torchvision.transforms.Compose
eval_transform(image_size)  -> torchvision.transforms.Compose
"""

from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (fixed, not dataset-derived)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 224) -> T.Compose:
    """Return the training augmentation pipeline.

    Augmentations applied:
    - Resize to ``image_size × image_size``
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random rotation ±15°
    - ToTensor + Normalize with ImageNet statistics

    The flip and rotation choices are appropriate for skin-lesion images,
    which have no natural orientation.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224) -> T.Compose:
    """Return the evaluation (val/test) pipeline.

    No augmentation is applied: resize, ToTensor, normalise.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])