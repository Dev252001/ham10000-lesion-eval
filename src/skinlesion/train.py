"""
train.py – training loop with early stopping, checkpointing, and provenance.

Public API
----------
seed_everything(seed)          : seed Python, numpy, torch, CUDA
train_model(...)               : training loop; returns TrainHistory
load_or_train(...)             : cache-aware wrapper
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skinlesion.config import TrainConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU and CUDA).

    Note: bitwise determinism is not guaranteed across different PyTorch
    versions, CUDA versions, or hardware.  This sets the seeds but does not
    enable ``torch.use_deterministic_algorithms(True)``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Per-DataLoader worker seed function to improve reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# TrainHistory
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)
    best_epoch: int = 0          # 1-based
    epochs_run: int = 0
    train_time_s: float = 0.0    # wall-clock seconds for the epoch loop only


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    class_weights: torch.Tensor,
    device: torch.device,
    ckpt_path: Path,
) -> TrainHistory:
    """Train *model* and save the best checkpoint.

    The training loop has NO access to the test split.  Selection and early
    stopping use validation macro-F1 only (tie-break: lower validation loss).

    Timing is measured with ``time.perf_counter()`` around the epoch loop
    only (excludes model/data setup time).

    A ``<ckpt_stem>_meta.json`` file is written beside the checkpoint
    recording the config fingerprint, training metadata, and provenance
    fields.

    Parameters
    ----------
    model:
        nn.Module to train (modified in place; best weights saved to disk).
    train_loader, val_loader:
        DataLoaders for train and validation partitions.
    cfg:
        TrainConfig with all hyperparameters.
    class_weights:
        Tensor of shape (7,) from ``compute_class_weights``.
    device:
        Target device (``torch.device``).
    ckpt_path:
        Path where the best model state dict is saved (``*.pt``).

    Returns
    -------
    TrainHistory
    """
    from sklearn.metrics import f1_score  # local import – optional at module level

    model = model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = TrainHistory()
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    patience_counter = 0

    # ------------------------------------------------------------------
    # Epoch loop – timing starts here
    # ------------------------------------------------------------------
    t_start = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_preds: List[int] = []
        all_true: List[int] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_total += loss.item()
                val_batches += 1
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(yb.cpu().numpy().tolist())
        val_loss = val_loss_total / max(val_batches, 1)

        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_macro_f1.append(val_f1)
        history.epochs_run = epoch

        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f",
            epoch, cfg.max_epochs, train_loss, val_loss, val_f1,
        )

        # --- Selection: best val macro-F1; tie-break lower val loss ---
        improved = (val_f1 > best_val_f1) or (
            val_f1 == best_val_f1 and val_loss < best_val_loss
        )
        if improved:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    # Epoch loop ends here
    t_end = time.perf_counter()
    history.train_time_s = t_end - t_start

    # ------------------------------------------------------------------
    # Save best checkpoint
    # ------------------------------------------------------------------
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        log.info("Checkpoint saved to %s (best epoch %d)", ckpt_path, history.best_epoch)

    # ------------------------------------------------------------------
    # Write _meta.json beside the checkpoint
    # ------------------------------------------------------------------
    meta = {
        "config_fingerprint": cfg.fingerprint(),
        "model_name": cfg.model_name,
        "split_mode": cfg.split_mode,
        "seed": cfg.seed,
        "manifest_sha256": cfg.manifest_sha256,
        "best_epoch": history.best_epoch,
        "epochs_run": history.epochs_run,
        "train_time_s": history.train_time_s,
        "best_val_macro_f1": best_val_f1,
        "hyperparameters": {
            "image_size": cfg.image_size,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        "selection_rule": cfg.selection_rule,
    }
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    log.info("Meta written to %s", meta_path)

    return history


# ---------------------------------------------------------------------------
# load_or_train
# ---------------------------------------------------------------------------


def load_or_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    class_weights: torch.Tensor,
    device: torch.device,
    ckpt_dir: Path,
) -> Tuple[nn.Module, TrainHistory, bool]:
    """Load an existing checkpoint or train from scratch.

    A checkpoint is reused only when:
    1. The ``.pt`` file exists, AND
    2. The corresponding ``_meta.json`` exists, AND
    3. The stored ``config_fingerprint`` matches ``cfg.fingerprint()``.

    If any condition fails the model is retrained and the reason is printed.

    Parameters
    ----------
    model:
        nn.Module (loaded weights are applied in place).
    ckpt_dir:
        Directory in which to look for / save checkpoints.
        File name: ``{model}_{split_mode}_seed{seed}_best.pt``.

    Returns
    -------
    (model_with_weights, history, loaded_from_cache: bool)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"{cfg.model_name}_{cfg.split_mode}_seed{cfg.seed}_best.pt"
    ckpt_path = ckpt_dir / ckpt_name
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")

    target_fp = cfg.fingerprint()

    if ckpt_path.exists() and meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                stored_meta = json.load(fh)
            stored_fp = stored_meta.get("config_fingerprint", "")
            if stored_fp == target_fp:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model = model.to(device)
                history = TrainHistory(
                    best_epoch=stored_meta.get("best_epoch", 0),
                    epochs_run=stored_meta.get("epochs_run", 0),
                    train_time_s=stored_meta.get("train_time_s", 0.0),
                )
                log.info("Loaded checkpoint from %s (fingerprint matches)", ckpt_path)
                return model, history, True
            else:
                print(
                    f"[load_or_train] Retraining {ckpt_name}: "
                    f"stored fingerprint does not match current config."
                )
        except Exception as exc:
            print(f"[load_or_train] Retraining {ckpt_name}: could not read meta ({exc}).")
    elif ckpt_path.exists():
        print(
            f"[load_or_train] Retraining {ckpt_name}: "
            f"checkpoint exists but meta.json is missing."
        )
    else:
        print(f"[load_or_train] Training {ckpt_name}: no existing checkpoint found.")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_weights=class_weights,
        device=device,
        ckpt_path=ckpt_path,
    )
    return model, history, False