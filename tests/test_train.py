"""
tests/test_train.py
-------------------
Tests for skinlesion.config and skinlesion.train.

Uses a tiny CPU model (BaselineCNN) and synthetic loaders with 1-2 epochs
to keep the test suite fast.  No network access.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from skinlesion.config import TrainConfig
from skinlesion.models import BaselineCNN, build_model, compute_class_weights
from skinlesion.train import TrainHistory, load_or_train, seed_everything, train_model

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_CLASSES = 7
_IMG_SIZE = 16   # tiny spatial size to keep CPU tests fast


def _synthetic_loader(n: int, seed: int = 0) -> DataLoader:
    """Return a DataLoader with *n* random (image, label) pairs."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.rand(n, 3, _IMG_SIZE, _IMG_SIZE, generator=rng)
    y = torch.randint(0, _N_CLASSES, (n,), generator=rng)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4, num_workers=0)


def _make_config(
    tmp_path: Path,
    split_mode: str = "lesion",
    seed: int = 42,
    max_epochs: int = 2,
    manifest_sha: str = "a" * 64,
) -> TrainConfig:
    return TrainConfig(
        model_name="baseline_cnn",
        split_mode=split_mode,
        seed=seed,
        image_size=_IMG_SIZE,
        batch_size=4,
        max_epochs=max_epochs,
        patience=5,
        manifest_sha256=manifest_sha,
    )


def _class_weights() -> torch.Tensor:
    # Uniform weights for simplicity
    return torch.ones(_N_CLASSES, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Tests: TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_default_lr_baseline_cnn(self):
        cfg = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42,
                          manifest_sha256="x" * 64)
        assert cfg.lr == pytest.approx(1e-3)

    def test_default_lr_resnet18(self):
        cfg = TrainConfig(model_name="resnet18", split_mode="lesion", seed=42,
                          manifest_sha256="x" * 64)
        assert cfg.lr == pytest.approx(3e-4)

    def test_default_lr_efficientnet(self):
        cfg = TrainConfig(model_name="efficientnet_b0", split_mode="lesion", seed=42,
                          manifest_sha256="x" * 64)
        assert cfg.lr == pytest.approx(3e-4)

    def test_fingerprint_is_64_hex(self):
        cfg = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42,
                          manifest_sha256="b" * 64)
        fp = cfg.fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_missing_manifest_raises(self):
        cfg = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42)
        with pytest.raises(ValueError, match="manifest_sha256"):
            cfg.fingerprint()

    def test_different_split_mode_different_fingerprint(self):
        cfg1 = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42,
                           manifest_sha256="c" * 64)
        cfg2 = TrainConfig(model_name="baseline_cnn", split_mode="image", seed=42,
                           manifest_sha256="c" * 64)
        assert cfg1.fingerprint() != cfg2.fingerprint()

    def test_different_seed_different_fingerprint(self):
        cfg1 = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42,
                           manifest_sha256="d" * 64)
        cfg2 = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=99,
                           manifest_sha256="d" * 64)
        assert cfg1.fingerprint() != cfg2.fingerprint()

    def test_different_manifest_different_fingerprint(self):
        cfg1 = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42,
                           manifest_sha256="e" * 64)
        cfg2 = TrainConfig(model_name="baseline_cnn", split_mode="lesion", seed=42,
                           manifest_sha256="f" * 64)
        assert cfg1.fingerprint() != cfg2.fingerprint()


# ---------------------------------------------------------------------------
# Tests: train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("train")
        model = BaselineCNN()
        cfg = _make_config(tmp, max_epochs=2)
        train_loader = _synthetic_loader(16, seed=1)
        val_loader = _synthetic_loader(8, seed=2)
        cw = _class_weights()
        device = torch.device("cpu")
        ckpt = tmp / "baseline_cnn_lesion_seed42_best.pt"
        history = train_model(model, train_loader, val_loader, cfg, cw, device, ckpt)
        return history, ckpt, tmp

    def test_history_has_correct_keys(self, trained):
        history, _, _ = trained
        assert hasattr(history, "train_loss")
        assert hasattr(history, "val_loss")
        assert hasattr(history, "val_macro_f1")
        assert hasattr(history, "best_epoch")
        assert hasattr(history, "epochs_run")
        assert hasattr(history, "train_time_s")

    def test_history_list_lengths(self, trained):
        history, _, _ = trained
        assert len(history.train_loss) == history.epochs_run
        assert len(history.val_loss) == history.epochs_run
        assert len(history.val_macro_f1) == history.epochs_run

    def test_train_loss_is_float(self, trained):
        history, _, _ = trained
        for v in history.train_loss:
            assert isinstance(v, float)

    def test_train_time_s_positive(self, trained):
        history, _, _ = trained
        assert isinstance(history.train_time_s, float)
        assert history.train_time_s > 0.0

    def test_best_epoch_valid(self, trained):
        history, _, _ = trained
        assert 1 <= history.best_epoch <= history.epochs_run

    def test_checkpoint_written(self, trained):
        _, ckpt, _ = trained
        assert ckpt.exists()

    def test_meta_written(self, trained):
        _, ckpt, _ = trained
        meta_path = ckpt.with_name(ckpt.stem + "_meta.json")
        assert meta_path.exists()

    def test_meta_has_required_fields(self, trained):
        _, ckpt, _ = trained
        meta_path = ckpt.with_name(ckpt.stem + "_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("config_fingerprint", "train_time_s", "best_epoch",
                    "epochs_run", "manifest_sha256", "selection_rule"):
            assert key in meta, f"Missing key: {key}"

    def test_meta_train_time_positive(self, trained):
        _, ckpt, _ = trained
        meta_path = ckpt.with_name(ckpt.stem + "_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["train_time_s"] > 0.0


# ---------------------------------------------------------------------------
# Tests: load_or_train – reuse and retrain logic
# ---------------------------------------------------------------------------

class TestLoadOrTrain:
    def _run_once(self, tmp_path, split_mode="lesion", seed=42, sha="a" * 64):
        model = BaselineCNN()
        cfg = _make_config(tmp_path, split_mode=split_mode, seed=seed,
                           max_epochs=1, manifest_sha=sha)
        train_loader = _synthetic_loader(8, seed=10)
        val_loader = _synthetic_loader(4, seed=11)
        cw = _class_weights()
        device = torch.device("cpu")
        return load_or_train(model, train_loader, val_loader, cfg, cw, device, tmp_path)

    def test_first_run_not_cached(self, tmp_path):
        _, _, cached = self._run_once(tmp_path)
        assert cached is False

    def test_second_run_cached(self, tmp_path):
        self._run_once(tmp_path)
        _, _, cached = self._run_once(tmp_path)
        assert cached is True

    def test_different_split_mode_forces_retrain(self, tmp_path):
        self._run_once(tmp_path, split_mode="lesion")
        _, _, cached = self._run_once(tmp_path, split_mode="image")
        assert cached is False

    def test_different_seed_forces_retrain(self, tmp_path):
        self._run_once(tmp_path, seed=42)
        # Different seed -> different ckpt name and fingerprint
        _, _, cached = self._run_once(tmp_path, seed=99)
        assert cached is False

    def test_different_manifest_forces_retrain(self, tmp_path):
        self._run_once(tmp_path, sha="a" * 64)
        _, _, cached = self._run_once(tmp_path, sha="b" * 64)
        assert cached is False

    def test_matching_fingerprint_skips_training(self, tmp_path):
        # First run: trains and saves checkpoint
        self._run_once(tmp_path)
        # Second run: should load from cache without printing a retrain message.
        # We capture stdout by redirecting sys.stdout.
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            self._run_once(tmp_path)
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        assert "Retraining" not in output, f"Unexpected retrain: {output!r}"
        assert "Training baseline_cnn" not in output, f"Unexpected train msg: {output!r}"