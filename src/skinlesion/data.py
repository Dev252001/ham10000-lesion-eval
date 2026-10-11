"""
data.py – metadata loading, image discovery, Dataset, and dataset statistics.

Public API
----------
CLASSES          : tuple[str, ...] – seven sorted diagnostic labels
MALIGNANT        : tuple[str, ...]
load_metadata    : load CSV + map image stems to paths
HAMDataset       : torch Dataset returning (tensor, label_idx)
dataset_stats    : per-class image/lesion counts + imbalance ratios
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES: Tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MALIGNANT: Tuple[str, ...] = ("mel", "bcc", "akiec")

_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


def load_metadata(data_dir: Path | str) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and attach an ``image_path`` column.

    Steps
    -----
    1. Read ``HAM10000_metadata.csv`` from *data_dir*.
    2. Recursively glob all ``*.jpg`` / ``*.jpeg`` files (case-insensitive)
       under *data_dir* and build a stem → path mapping.
    3. Attach the resolved path as ``image_path``; raise ``FileNotFoundError``
       for any ``image_id`` whose file cannot be found.
    4. Raise ``ValueError`` if any ``lesion_id`` maps to more than one ``dx``
       (a sign of a corrupted annotation file).

    Parameters
    ----------
    data_dir:
        Root directory that contains ``HAM10000_metadata.csv`` and the
        ``*.jpg`` image files (may be in subdirectories).

    Returns
    -------
    pd.DataFrame
        One row per image with columns from the CSV plus ``image_path``
        (``pathlib.Path``).
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # Build stem -> path index (case-insensitive extension matching)
    # ------------------------------------------------------------------
    stem_to_path: Dict[str, Path] = {}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg"}:
            stem_to_path[p.stem] = p

    # ------------------------------------------------------------------
    # Attach image_path; collect missing ids for a single clear error
    # ------------------------------------------------------------------
    missing: List[str] = []
    paths: List[Optional[Path]] = []
    for iid in df["image_id"]:
        p = stem_to_path.get(iid)
        if p is None:
            missing.append(iid)
        paths.append(p)

    if missing:
        n = len(missing)
        sample = missing[:5]
        raise FileNotFoundError(
            f"{n} image file(s) listed in the metadata CSV could not be "
            f"found under {data_dir}. "
            f"First up to 5 missing IDs: {sample}"
        )

    df = df.copy()
    df["image_path"] = paths

    # ------------------------------------------------------------------
    # Integrity check: each lesion_id must have exactly one dx
    # ------------------------------------------------------------------
    lesion_dx = df.groupby("lesion_id")["dx"].nunique()
    bad = lesion_dx[lesion_dx > 1]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} lesion_id(s) have more than one dx value, which "
            f"indicates a corrupted annotation file. "
            f"Offending lesion_ids: {bad.index.tolist()[:5]}"
        )

    log.info("Loaded %d images, %d lesions, %d classes",
             len(df), df["lesion_id"].nunique(), df["dx"].nunique())
    return df


# ---------------------------------------------------------------------------
# HAMDataset
# ---------------------------------------------------------------------------


class HAMDataset:
    """Torch-style dataset returning ``(tensor, label_idx)`` pairs.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`load_metadata` (must have
        ``image_path`` and ``dx`` columns).
    transform:
        Optional callable applied to the PIL image before converting to
        tensor.  When *None* the image is returned as a float32 tensor in
        [0, 1] with shape ``(3, H, W)``.
    """

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self._df = df.reset_index(drop=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        import torch
        import torchvision.transforms.functional as F  # noqa: N812

        row = self._df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self._transform is not None:
            img = self._transform(img)
        else:
            img = F.to_tensor(img)

        label = _CLASS_TO_IDX[row["dx"]]
        return img, label


# ---------------------------------------------------------------------------
# dataset_stats
# ---------------------------------------------------------------------------


def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute per-class image and lesion counts plus imbalance ratios.

    The imbalance ratio is ``majority_count / minority_count`` computed
    separately at the image level and at the lesion level (one row per
    ``lesion_id``).

    Parameters
    ----------
    df:
        DataFrame with at least ``dx`` and ``lesion_id`` columns.

    Returns
    -------
    dict
        Structure::

            {
                "per_class": {
                    "<class>": {"images": int, "lesions": int},
                    ...
                },
                "image_level": {
                    "majority_class": str, "majority_count": int,
                    "minority_class": str, "minority_count": int,
                    "imbalance_ratio": float,
                },
                "lesion_level": {
                    "majority_class": str, "majority_count": int,
                    "minority_class": str, "minority_count": int,
                    "imbalance_ratio": float,
                },
            }

        Suitable for ``json.dumps`` and saving to
        ``outputs/dataset_stats.json``.
    """
    # Per-class image counts (use all rows)
    image_counts: Dict[str, int] = (
        df.groupby("dx").size().to_dict()
    )

    # Per-class lesion counts (one row per unique lesion)
    lesion_df = df.drop_duplicates("lesion_id")
    lesion_counts: Dict[str, int] = (
        lesion_df.groupby("dx").size().to_dict()
    )

    per_class = {
        cls: {
            "images": int(image_counts.get(cls, 0)),
            "lesions": int(lesion_counts.get(cls, 0)),
        }
        for cls in CLASSES
    }

    def _ratio_info(counts: Dict[str, int], label: str) -> dict:
        if not counts:
            return {}
        maj_cls = max(counts, key=counts.__getitem__)
        min_cls = min(counts, key=counts.__getitem__)
        maj_n = counts[maj_cls]
        min_n = counts[min_cls]
        return {
            "majority_class": maj_cls,
            "majority_count": int(maj_n),
            "minority_class": min_cls,
            "minority_count": int(min_n),
            "imbalance_ratio": round(maj_n / min_n, 4) if min_n else None,
        }

    return {
        "per_class": per_class,
        "image_level": _ratio_info(image_counts, "image"),
        "lesion_level": _ratio_info(lesion_counts, "lesion"),
    }