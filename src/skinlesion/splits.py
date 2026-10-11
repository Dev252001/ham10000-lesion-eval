"""
splits.py – leakage-aware train/val/test splitting for HAM10000.

Public API
----------
MIN_LESIONS_WARN   : int    – warn when a class has fewer lesions than this
make_split         : stratified split at lesion or image level
split_manifest     : build a DataFrame recording each image's partition
manifest_sha256    : stable, OS-independent hash of the manifest
save_manifest      : write manifest CSV to outputs/splits/
audit_split        : per-class, per-split counts + leakage detection
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import warnings
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# Heuristic: warn when a split has fewer than this many lesions for a class.
# 20 is a rough lower bound; real coverage depends on the dataset.
MIN_LESIONS_WARN: int = 20

# ---------------------------------------------------------------------------
# make_split
# ---------------------------------------------------------------------------


def make_split(
    df: pd.DataFrame,
    split_mode: Literal["lesion", "image"] = "lesion",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test partitions.

    Lesion mode (correct, leakage-free)
    ------------------------------------
    1. Collapse *df* to one row per ``lesion_id`` (keeping ``dx``).
    2. Two-step stratified split on ``dx`` at the **lesion** level:
       a. Separate test lesions from the rest.
       b. Separate val lesions from the remaining train lesions.
    3. Map back to the original image rows: every image that belongs to a
       test lesion goes to test, etc.
    This guarantees that no image of a given lesion appears in more than one
    partition, preventing the label-leakage present in naive image-level splits.

    Image mode (intentionally leaky – kept only for comparison)
    -----------------------------------------------------------
    Stratified split directly on image rows.  Images of the same lesion may
    appear in both train and test, which inflates evaluation metrics.

    Parameters
    ----------
    df:
        DataFrame with at least ``image_id``, ``lesion_id``, and ``dx``.
    split_mode:
        ``"lesion"`` (recommended) or ``"image"`` (leaky comparison only).
    val_size:
        Fraction of the dataset for validation (applied to lesions/images).
    test_size:
        Fraction of the dataset for test.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df): DataFrames of image rows.

    Raises
    ------
    ValueError
        If *split_mode* is not ``"lesion"`` or ``"image"``.
    """
    if split_mode not in ("lesion", "image"):
        raise ValueError(
            f"split_mode must be 'lesion' or 'image', got {split_mode!r}"
        )

    if split_mode == "lesion":
        return _lesion_split(df, val_size=val_size, test_size=test_size, seed=seed)
    else:
        return _image_split(df, val_size=val_size, test_size=test_size, seed=seed)


def _lesion_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # One row per lesion
    lesion_df = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]].reset_index(drop=True)

    # Step 1: split off test lesions
    # The relative test fraction out of the full set is test_size.
    # sklearn's train_test_split uses test_size as a fraction of what's passed in.
    train_val_lesions, test_lesions = train_test_split(
        lesion_df,
        test_size=test_size,
        stratify=lesion_df["dx"],
        random_state=seed,
    )

    # Step 2: split val out of the remainder.
    # val_size is a fraction of the *full* dataset, so we adjust relative to
    # the train+val pool.
    n_total = len(lesion_df)
    n_trainval = len(train_val_lesions)
    val_relative = val_size * n_total / n_trainval

    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=val_relative,
        stratify=train_val_lesions["dx"],
        random_state=seed,
    )

    # Map lesion sets back to image rows
    test_ids = set(test_lesions["lesion_id"])
    val_ids = set(val_lesions["lesion_id"])
    # Everything else is train
    test_df = df[df["lesion_id"].isin(test_ids)].copy()
    val_df = df[df["lesion_id"].isin(val_ids)].copy()
    train_df = df[~df["lesion_id"].isin(test_ids | val_ids)].copy()

    _log_split_stats(train_df, val_df, test_df, split_mode="lesion")
    return train_df, val_df, test_df


def _image_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_total = len(df)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["dx"],
        random_state=seed,
    )

    n_trainval = len(train_val_df)
    val_relative = val_size * n_total / n_trainval

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=train_val_df["dx"],
        random_state=seed,
    )

    _log_split_stats(train_df, val_df, test_df, split_mode="image")
    return train_df, val_df, test_df


def _log_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_mode: str,
) -> None:
    """Log per-class image and lesion counts; warn on sparse val/test classes."""
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for cls, grp in split_df.groupby("dx"):
            n_img = len(grp)
            n_les = grp["lesion_id"].nunique()
            log.info(
                "[%s|%s] %s: %d images, %d lesions",
                split_mode, split_name, cls, n_img, n_les,
            )
            if split_name in ("val", "test") and n_les < MIN_LESIONS_WARN:
                warnings.warn(
                    f"[{split_mode}|{split_name}] class '{cls}' has only "
                    f"{n_les} lesion(s), below MIN_LESIONS_WARN={MIN_LESIONS_WARN}. "
                    f"Evaluation metrics for this class may be unreliable.",
                    UserWarning,
                    stacklevel=4,
                )


# ---------------------------------------------------------------------------
# split_manifest
# ---------------------------------------------------------------------------


def split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_mode: str,
    seed: int,
) -> pd.DataFrame:
    """Build a manifest DataFrame recording which partition each image belongs to.

    Parameters
    ----------
    train_df, val_df, test_df:
        DataFrames as returned by :func:`make_split`.
    split_mode:
        The mode string (stored in the manifest for traceability).
    seed:
        The random seed (stored for traceability).

    Returns
    -------
    pd.DataFrame
        Columns: ``image_id``, ``lesion_id``, ``dx``, ``split``.
    """
    parts = []
    for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tmp = part_df[["image_id", "lesion_id", "dx"]].copy()
        tmp["split"] = name
        parts.append(tmp)

    manifest = pd.concat(parts, ignore_index=True)
    manifest["split_mode"] = split_mode
    manifest["seed"] = seed
    return manifest


# ---------------------------------------------------------------------------
# manifest_sha256
# ---------------------------------------------------------------------------


def manifest_sha256(manifest: pd.DataFrame) -> str:
    """Stable, OS-independent SHA-256 hash of the manifest.

    The hash is computed over the canonical CSV text:
    rows sorted by ``image_id``, columns ``image_id,lesion_id,dx,split``,
    LF line endings, UTF-8.  The ``split_mode`` and ``seed`` columns are
    *not* included so that the hash captures only the actual partition
    assignment rather than metadata about how it was produced.

    This function always returns a 64-character hex string and is never null.
    """
    cols = ["image_id", "lesion_id", "dx", "split"]
    canonical = (
        manifest[cols]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    buf = io.StringIO()
    canonical.to_csv(buf, index=False, lineterminator="\n")
    csv_text = buf.getvalue()
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# save_manifest
# ---------------------------------------------------------------------------


def save_manifest(
    manifest: pd.DataFrame,
    outputs_dir: Path | str,
    split_mode: str,
    seed: int,
    sha: str,
) -> Path:
    """Write the manifest CSV to ``outputs/splits/``.

    File name: ``manifest_<mode>_seed<N>.csv``.
    The SHA-256 is stored as the first comment line so it can be verified
    without re-computing.

    Returns the path of the written file.
    """
    outputs_dir = Path(outputs_dir)
    splits_dir = outputs_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    fname = splits_dir / f"manifest_{split_mode}_seed{seed}.csv"

    with open(fname, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# sha256={sha}\n")
        manifest.to_csv(fh, index=False, lineterminator="\n")

    log.info("Manifest written to %s (sha256=%s…)", fname, sha[:8])
    return fname


# ---------------------------------------------------------------------------
# audit_split
# ---------------------------------------------------------------------------


def audit_split(
    manifest: pd.DataFrame,
    split_mode: str,
    seed: int,
    outputs_dir: Path | str | None = None,
) -> dict:
    """Audit the split for class balance and lesion leakage.

    Returns
    -------
    dict
        Structure::

            {
                "split_mode": str,
                "seed": int,
                "per_class": {
                    "<class>": {
                        "train":  {"images": int, "lesions": int},
                        "val":    {"images": int, "lesions": int},
                        "test":   {"images": int, "lesions": int},
                    },
                    ...
                },
                "leakage": {
                    "lesions_in_train_and_val":  int,
                    "lesions_in_train_and_test": int,
                    "lesions_in_val_and_test":   int,
                    "val_images_with_train_lesion":  int,
                    "test_images_with_train_lesion": int,
                    "val_share_with_train_lesion":   float,
                    "test_share_with_train_lesion":  float,
                },
            }

        If *outputs_dir* is given, the dict is written as
        ``audit_<mode>_seed<N>.json``.
    """
    splits = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}

    # Per-class, per-split counts
    all_classes = sorted(manifest["dx"].unique())
    per_class = {}
    for cls in all_classes:
        per_class[cls] = {}
        for split_name, sdf in splits.items():
            cls_rows = sdf[sdf["dx"] == cls]
            per_class[cls][split_name] = {
                "images": int(len(cls_rows)),
                "lesions": int(cls_rows["lesion_id"].nunique()),
            }

    # Leakage analysis
    train_lesions = set(splits["train"]["lesion_id"])
    val_lesions = set(splits["val"]["lesion_id"])
    test_lesions = set(splits["test"]["lesion_id"])

    tv_overlap = len(train_lesions & val_lesions)
    tt_overlap = len(train_lesions & test_lesions)
    vt_overlap = len(val_lesions & test_lesions)

    val_leaked_imgs = splits["val"]["lesion_id"].isin(train_lesions).sum()
    test_leaked_imgs = splits["test"]["lesion_id"].isin(train_lesions).sum()

    n_val = max(len(splits["val"]), 1)
    n_test = max(len(splits["test"]), 1)

    leakage = {
        "lesions_in_train_and_val": int(tv_overlap),
        "lesions_in_train_and_test": int(tt_overlap),
        "lesions_in_val_and_test": int(vt_overlap),
        "val_images_with_train_lesion": int(val_leaked_imgs),
        "test_images_with_train_lesion": int(test_leaked_imgs),
        "val_share_with_train_lesion": round(val_leaked_imgs / n_val, 4),
        "test_share_with_train_lesion": round(test_leaked_imgs / n_test, 4),
    }

    result = {
        "split_mode": split_mode,
        "seed": seed,
        "per_class": per_class,
        "leakage": leakage,
    }

    if outputs_dir is not None:
        out = Path(outputs_dir) / "splits"
        out.mkdir(parents=True, exist_ok=True)
        fpath = out / f"audit_{split_mode}_seed{seed}.json"
        with open(fpath, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        log.info("Audit written to %s", fpath)

    return result