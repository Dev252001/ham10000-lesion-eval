"""
evaluate.py – test-set evaluation, predictions, metrics, and bootstrap CIs.

Public API
----------
predict                  : run inference; return y_true, y_pred, probabilities
final_test_evaluation    : the ONLY function that consumes the test split
cluster_bootstrap_ci     : 95% CI via lesion-cluster resampling
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skinlesion.config import TrainConfig
from skinlesion.data import CLASSES, MALIGNANT
from skinlesion.train import TrainHistory

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return ground-truth labels, predicted labels, and
    class probabilities.

    Parameters
    ----------
    model:
        Trained nn.Module (called with ``model.eval()``).
    loader:
        DataLoader yielding (image_tensor, label_idx) batches.
    device:
        Target device.

    Returns
    -------
    y_true : np.ndarray, shape (N,), dtype int
    y_pred : np.ndarray, shape (N,), dtype int
    probs  : np.ndarray, shape (N, C), dtype float32 – softmax probabilities
    """
    model.eval()
    model = model.to(device)
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            preds = probs.argmax(axis=1)
            all_true.extend(yb.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.append(probs)

    return (
        np.array(all_true, dtype=int),
        np.array(all_pred, dtype=int),
        np.vstack(all_probs) if all_probs else np.empty((0, len(CLASSES))),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit(repo_dir: Path) -> Tuple[str, bool]:
    """Return (commit_hash_or_unknown, is_dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit, bool(dirty_out)
    except Exception:
        return "unknown", False


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> dict:
    """Compute all required evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }
        for cls in CLASSES
    }

    # Confusion matrix (list of lists of int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_list = cm.tolist()

    # Malignant recall: mean recall over mel, bcc, akiec
    malignant_recalls = []
    for cls in MALIGNANT:
        idx = list(CLASSES).index(cls)
        malignant_recalls.append(per_class[cls]["recall"])
    malignant_recall = float(np.mean(malignant_recalls))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "per_class": per_class,
        "confusion_matrix": cm_list,
        "malignant_recall": malignant_recall,
    }


# ---------------------------------------------------------------------------
# final_test_evaluation
# ---------------------------------------------------------------------------


def final_test_evaluation(
    model: nn.Module,
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    cfg: TrainConfig,
    history: TrainHistory,
    manifest_sha256: str,
    repo_dir: Path,
    outputs_dir: Path,
    device: torch.device,
) -> Tuple[dict, Path, Path]:
    """Evaluate the model on the test split.  This is the ONLY code path
    that consumes the test split.

    Writes two files to ``outputs/``:
    - ``predictions/<model>_<mode>_seed<N>.csv``
    - ``metrics/<model>_<mode>_seed<N>.json``

    Parameters
    ----------
    model:
        Trained nn.Module.
    test_df:
        Test-partition DataFrame (must have image_id, lesion_id, dx columns).
    test_loader:
        DataLoader for the test partition (images in the same row order as test_df).
    cfg:
        TrainConfig used for training.
    history:
        TrainHistory returned by train_model / load_or_train.
    manifest_sha256:
        SHA-256 of the split manifest.  Raises if empty.
    repo_dir:
        Root of the git repository (used to obtain the current commit hash).
    outputs_dir:
        Root outputs directory (e.g. ``outputs/``).
    device:
        Target device.

    Returns
    -------
    (metrics_dict, predictions_path, metrics_path)

    Raises
    ------
    ValueError
        If *manifest_sha256* is empty.
    """
    if not manifest_sha256:
        raise ValueError(
            "manifest_sha256 is required for final_test_evaluation. "
            "Pass the value returned by splits.manifest_sha256()."
        )

    # --- Run inference ---
    y_true, y_pred, probs = predict(model, test_loader, device)

    # --- Build predictions DataFrame ---
    test_df_reset = test_df.reset_index(drop=True)
    pred_rows = {
        "image_id": test_df_reset["image_id"].tolist(),
        "lesion_id": test_df_reset["lesion_id"].tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    for i, cls in enumerate(CLASSES):
        pred_rows[f"p_{cls}"] = probs[:, i].tolist()
    pred_df = pd.DataFrame(pred_rows)

    # --- Compute metrics ---
    metrics = _compute_metrics(y_true, y_pred, probs)

    # --- Library versions ---
    import sklearn
    import torchvision
    lib_versions = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit-learn": sklearn.__version__,
    }

    # --- Git provenance ---
    git_commit, git_dirty = _git_commit(repo_dir)

    # --- Device name ---
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda"
        else "cpu"
    )

    # --- train_time_s: use history value; never fake a 0.0 placeholder ---
    train_time_s = history.train_time_s if history.train_time_s > 0 else None

    # --- Build JSON ---
    from skinlesion.models import count_params
    metrics_doc = {
        "model_name": cfg.model_name,
        "split_mode": cfg.split_mode,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "metrics": metrics,
        "n_params": count_params(model),
        "train_time_s": train_time_s,
        "epochs_run": history.epochs_run,
        "best_epoch": history.best_epoch,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "manifest_sha256": manifest_sha256,
        "library_versions": lib_versions,
        "device": device_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    # --- Determine file names ---
    stem = f"{cfg.model_name}_{cfg.split_mode}_seed{cfg.seed}"
    preds_dir = Path(outputs_dir) / "predictions"
    metrics_dir = Path(outputs_dir) / "metrics"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    preds_path = preds_dir / f"{stem}.csv"
    metrics_path = metrics_dir / f"{stem}.json"

    # --- Write predictions CSV ---
    pred_df.to_csv(preds_path, index=False, lineterminator="\n")

    # --- Write metrics JSON ---
    with open(metrics_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(metrics_doc, fh, indent=2)
        fh.write("\n")

    log.info("Predictions written to %s", preds_path)
    log.info("Metrics written to %s", metrics_path)

    return metrics_doc, preds_path, metrics_path


# ---------------------------------------------------------------------------
# cluster_bootstrap_ci
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(
    pred_df: pd.DataFrame,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Estimate a 95% percentile CI by resampling lesion_id clusters.

    Resampling at the lesion level (rather than the image level) respects the
    cluster structure of the data.  Resampling images independently would
    inflate the effective sample size and produce falsely narrow confidence
    intervals when multiple images of the same lesion are present.

    Parameters
    ----------
    pred_df:
        Predictions DataFrame with columns:
        image_id, lesion_id, y_true, y_pred, and p_<class> for all 7 classes.
    metric:
        One of ``"macro_f1"`` or a class name from ``CLASSES`` (for recall).
    n_boot:
        Number of bootstrap iterations.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    (point_estimate, lower_95, upper_95)
    """
    from sklearn.metrics import f1_score, recall_score

    def _compute(df: pd.DataFrame) -> float:
        yt = df["y_true"].to_numpy()
        yp = df["y_pred"].to_numpy()
        if metric == "macro_f1":
            return float(f1_score(yt, yp, average="macro", zero_division=0))
        elif metric in CLASSES:
            cls_idx = list(CLASSES).index(metric)
            # Recall for a single class: fraction of true positives
            mask = yt == cls_idx
            if mask.sum() == 0:
                return 0.0
            return float((yp[mask] == cls_idx).mean())
        else:
            raise ValueError(
                f"Unknown metric {metric!r}. "
                f"Choose 'macro_f1' or a class name from CLASSES."
            )

    point = _compute(pred_df)

    rng = np.random.default_rng(seed)
    unique_lesions = pred_df["lesion_id"].unique()
    n_lesions = len(unique_lesions)

    boot_scores: List[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_lesions, size=n_lesions, replace=True)
        # Build a resampled DataFrame by concatenating rows for each sampled lesion.
        # Using a groupby map for efficiency.
        groups = pred_df.groupby("lesion_id", group_keys=False)
        parts = [groups.get_group(les) for les in sampled]
        boot_df = pd.concat(parts, ignore_index=True)
        boot_scores.append(_compute(boot_df))

    lower = float(np.percentile(boot_scores, 2.5))
    upper = float(np.percentile(boot_scores, 97.5))
    return point, lower, upper