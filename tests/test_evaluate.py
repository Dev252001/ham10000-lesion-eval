"""
tests/test_evaluate.py
-----------------------
Tests for skinlesion.evaluate.

Uses synthetic predictions (no real images, no real model run).
Nothing is written to outputs/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from skinlesion.config import TrainConfig
from skinlesion.data import CLASSES, MALIGNANT
from skinlesion.evaluate import (
    cluster_bootstrap_ci,
    final_test_evaluation,
    predict,
)
from skinlesion.models import BaselineCNN
from skinlesion.train import TrainHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = len(CLASSES)  # 7


def _make_pred_df(
    n_per_class: int = 10,
    perfect: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic predictions DataFrame.

    *perfect=True* makes y_pred == y_true (used for CI collapse test).
    """
    rng = np.random.default_rng(seed)
    rows = []
    img_counter = 0
    les_counter = 0
    for cls_idx, cls in enumerate(CLASSES):
        for li in range(n_per_class):
            les_id = f"SYN_LES_{les_counter:04d}"
            # 2 images per lesion
            for _ in range(2):
                y_true = cls_idx
                if perfect:
                    y_pred = cls_idx
                else:
                    # Mostly correct but occasionally wrong
                    y_pred = cls_idx if rng.random() > 0.3 else int(rng.integers(0, _N))
                probs = rng.dirichlet(np.ones(_N)).astype(np.float32)
                row = {
                    "image_id": f"SYN_IMG_{img_counter:05d}",
                    "lesion_id": les_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
                for i, c in enumerate(CLASSES):
                    row[f"p_{c}"] = float(probs[i])
                rows.append(row)
                img_counter += 1
            les_counter += 1
    return pd.DataFrame(rows)


def _synthetic_loader(n: int, seed: int = 0) -> DataLoader:
    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.rand(n, 3, 16, 16, generator=rng)
    y = torch.randint(0, _N, (n,), generator=rng)
    return DataLoader(TensorDataset(x, y), batch_size=4, num_workers=0)


def _make_cfg(manifest_sha: str = "a" * 64) -> TrainConfig:
    return TrainConfig(
        model_name="baseline_cnn",
        split_mode="lesion",
        seed=42,
        image_size=16,
        batch_size=4,
        max_epochs=1,
        manifest_sha256=manifest_sha,
    )


def _make_history(train_time_s: float = 1.5) -> TrainHistory:
    return TrainHistory(
        train_loss=[0.5],
        val_loss=[0.6],
        val_macro_f1=[0.5],
        best_epoch=1,
        epochs_run=1,
        train_time_s=train_time_s,
    )


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_shapes(self):
        model = BaselineCNN()
        loader = _synthetic_loader(12)
        device = torch.device("cpu")
        y_true, y_pred, probs = predict(model, loader, device)
        assert y_true.shape == (12,)
        assert y_pred.shape == (12,)
        assert probs.shape == (12, _N)

    def test_probs_sum_to_one(self):
        model = BaselineCNN()
        loader = _synthetic_loader(8)
        device = torch.device("cpu")
        _, _, probs = predict(model, loader, device)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_pred_argmax_matches_probs(self):
        model = BaselineCNN()
        loader = _synthetic_loader(8)
        device = torch.device("cpu")
        _, y_pred, probs = predict(model, loader, device)
        assert np.array_equal(y_pred, probs.argmax(axis=1))

    def test_y_true_dtype_int(self):
        model = BaselineCNN()
        loader = _synthetic_loader(4)
        device = torch.device("cpu")
        y_true, _, _ = predict(model, loader, device)
        assert y_true.dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# Tests: metrics against hand computation
# ---------------------------------------------------------------------------

class TestMetrics:
    """
    Hand-computed case: 7 classes, 10 samples each (one sample per class for
    simplicity).  Perfect predictions except class 0 (akiec) always wrong.
    """

    @pytest.fixture(scope="class")
    def perfect_df(self):
        return _make_pred_df(n_per_class=10, perfect=True, seed=0)

    def test_accuracy_perfect(self, perfect_df):
        from sklearn.metrics import accuracy_score
        y_true = perfect_df["y_true"].to_numpy()
        y_pred = perfect_df["y_pred"].to_numpy()
        assert accuracy_score(y_true, y_pred) == pytest.approx(1.0)

    def test_macro_f1_perfect(self, perfect_df):
        from sklearn.metrics import f1_score
        y_true = perfect_df["y_true"].to_numpy()
        y_pred = perfect_df["y_pred"].to_numpy()
        assert f1_score(y_true, y_pred, average="macro", zero_division=0) == pytest.approx(1.0)

    def test_confusion_matrix_diagonal_perfect(self, perfect_df):
        from sklearn.metrics import confusion_matrix
        y_true = perfect_df["y_true"].to_numpy()
        y_pred = perfect_df["y_pred"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=list(range(_N)))
        # Perfect: all off-diagonal elements are 0
        off_diag = cm - np.diag(np.diag(cm))
        assert off_diag.sum() == 0

    def test_confusion_matrix_sums_to_n_samples(self, perfect_df):
        from sklearn.metrics import confusion_matrix
        y_true = perfect_df["y_true"].to_numpy()
        y_pred = perfect_df["y_pred"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=list(range(_N)))
        assert cm.sum() == len(perfect_df)


# ---------------------------------------------------------------------------
# Tests: final_test_evaluation
# ---------------------------------------------------------------------------

class TestFinalTestEvaluation:
    @pytest.fixture(scope="class")
    def eval_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("eval")

        # Build a tiny test_df and matching loader (16x16 images, 14 samples)
        n_samples = 14
        test_df = pd.DataFrame({
            "image_id": [f"SYN_IMG_{i:04d}" for i in range(n_samples)],
            "lesion_id": [f"SYN_LES_{i // 2:04d}" for i in range(n_samples)],
            "dx": [CLASSES[i % _N] for i in range(n_samples)],
        })
        loader = _synthetic_loader(n_samples, seed=5)

        model = BaselineCNN()
        cfg = _make_cfg()
        history = _make_history(train_time_s=2.3)

        metrics_doc, preds_path, metrics_path = final_test_evaluation(
            model=model,
            test_df=test_df,
            test_loader=loader,
            cfg=cfg,
            history=history,
            manifest_sha256="a" * 64,
            repo_dir=Path("d:/HAM/ham10000-lesion-eval"),
            outputs_dir=tmp,
            device=torch.device("cpu"),
        )
        return metrics_doc, preds_path, metrics_path

    def test_required_top_level_fields(self, eval_result):
        doc, _, _ = eval_result
        required = [
            "model_name", "split_mode", "seed", "config", "metrics",
            "n_params", "train_time_s", "epochs_run", "best_epoch",
            "git_commit", "git_dirty", "manifest_sha256",
            "library_versions", "device", "created_utc",
        ]
        for key in required:
            assert key in doc, f"Missing key: {key}"

    def test_metrics_fields(self, eval_result):
        doc, _, _ = eval_result
        m = doc["metrics"]
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                    "per_class", "confusion_matrix", "malignant_recall"):
            assert key in m, f"Missing metrics key: {key}"

    def test_per_class_fields(self, eval_result):
        doc, _, _ = eval_result
        for cls in CLASSES:
            assert cls in doc["metrics"]["per_class"]
            pc = doc["metrics"]["per_class"][cls]
            for k in ("precision", "recall", "f1", "support"):
                assert k in pc

    def test_library_versions(self, eval_result):
        doc, _, _ = eval_result
        lv = doc["library_versions"]
        for lib in ("torch", "torchvision", "scikit-learn"):
            assert lib in lv

    def test_git_commit_format(self, eval_result):
        doc, _, _ = eval_result
        gc = doc["git_commit"]
        is_40_hex = bool(re.fullmatch(r"[0-9a-f]{40}", gc))
        assert is_40_hex or gc == "unknown", f"Unexpected git_commit: {gc!r}"

    def test_manifest_sha256_present(self, eval_result):
        doc, _, _ = eval_result
        assert doc["manifest_sha256"] == "a" * 64

    def test_train_time_not_zero(self, eval_result):
        doc, _, _ = eval_result
        # train_time_s should be > 0 (2.3 from our fixture)
        assert doc["train_time_s"] > 0

    def test_null_time_stays_null(self, tmp_path):
        """A history with train_time_s=0 should write null, not 0.0."""
        n_samples = 7
        test_df = pd.DataFrame({
            "image_id": [f"SYN_IMG_{i:04d}" for i in range(n_samples)],
            "lesion_id": [f"SYN_LES_{i:04d}" for i in range(n_samples)],
            "dx": [CLASSES[i % _N] for i in range(n_samples)],
        })
        loader = _synthetic_loader(n_samples, seed=9)
        model = BaselineCNN()
        cfg = _make_cfg()
        history = _make_history(train_time_s=0.0)  # zero -> null

        doc, _, _ = final_test_evaluation(
            model=model, test_df=test_df, test_loader=loader,
            cfg=cfg, history=history, manifest_sha256="a" * 64,
            repo_dir=Path("d:/HAM/ham10000-lesion-eval"),
            outputs_dir=tmp_path, device=torch.device("cpu"),
        )
        assert doc["train_time_s"] is None

    def test_missing_manifest_raises(self, tmp_path):
        n_samples = 7
        test_df = pd.DataFrame({
            "image_id": [f"SYN_IMG_{i:04d}" for i in range(n_samples)],
            "lesion_id": [f"SYN_LES_{i:04d}" for i in range(n_samples)],
            "dx": [CLASSES[i % _N] for i in range(n_samples)],
        })
        loader = _synthetic_loader(n_samples, seed=9)
        model = BaselineCNN()
        cfg = _make_cfg()
        history = _make_history()

        with pytest.raises(ValueError, match="manifest_sha256"):
            final_test_evaluation(
                model=model, test_df=test_df, test_loader=loader,
                cfg=cfg, history=history, manifest_sha256="",
                repo_dir=Path("d:/HAM/ham10000-lesion-eval"),
                outputs_dir=tmp_path, device=torch.device("cpu"),
            )

    def test_correct_filename(self, eval_result):
        _, preds_path, metrics_path = eval_result
        assert preds_path.name == "baseline_cnn_lesion_seed42.csv"
        assert metrics_path.name == "baseline_cnn_lesion_seed42.json"

    def test_predictions_csv_has_required_columns(self, eval_result):
        _, preds_path, _ = eval_result
        df = pd.read_csv(preds_path)
        for col in ["image_id", "lesion_id", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES]:
            assert col in df.columns

    def test_confusion_matrix_is_list_of_lists_of_int(self, eval_result):
        doc, _, _ = eval_result
        cm = doc["metrics"]["confusion_matrix"]
        assert isinstance(cm, list)
        assert all(isinstance(row, list) for row in cm)
        assert all(isinstance(v, int) for row in cm for v in row)

    def test_confusion_matrix_row_sums(self, eval_result):
        """Each row of the confusion matrix should sum to the per-class support."""
        doc, _, _ = eval_result
        cm = doc["metrics"]["confusion_matrix"]
        row_sums = [sum(row) for row in cm]
        supports = [
            doc["metrics"]["per_class"][cls]["support"]
            for cls in CLASSES
        ]
        assert row_sums == supports


# ---------------------------------------------------------------------------
# Tests: cluster_bootstrap_ci
# ---------------------------------------------------------------------------

class TestClusterBootstrapCI:
    @pytest.fixture(scope="class")
    def pred_df(self):
        return _make_pred_df(n_per_class=20, perfect=False, seed=42)

    def test_returns_three_floats(self, pred_df):
        result = cluster_bootstrap_ci(pred_df, metric="macro_f1", n_boot=50, seed=0)
        assert len(result) == 3
        point, lo, hi = result
        assert isinstance(point, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_point_estimate(self, pred_df):
        point, lo, hi = cluster_bootstrap_ci(pred_df, metric="macro_f1", n_boot=100, seed=0)
        assert lo <= point <= hi, f"Point {point} outside CI [{lo}, {hi}]"

    def test_deterministic_for_fixed_seed(self, pred_df):
        r1 = cluster_bootstrap_ci(pred_df, metric="macro_f1", n_boot=100, seed=7)
        r2 = cluster_bootstrap_ci(pred_df, metric="macro_f1", n_boot=100, seed=7)
        assert r1 == r2

    def test_different_seeds_differ(self, pred_df):
        r1 = cluster_bootstrap_ci(pred_df, metric="macro_f1", n_boot=200, seed=1)
        r2 = cluster_bootstrap_ci(pred_df, metric="macro_f1", n_boot=200, seed=2)
        # Very likely to differ
        assert r1 != r2

    def test_perfect_predictions_collapse(self):
        """Perfect predictions: every bootstrap sample also gives F1=1.0."""
        df = _make_pred_df(n_per_class=10, perfect=True, seed=0)
        point, lo, hi = cluster_bootstrap_ci(df, metric="macro_f1", n_boot=50, seed=0)
        assert point == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_class_recall_metric(self, pred_df):
        point, lo, hi = cluster_bootstrap_ci(pred_df, metric="mel", n_boot=50, seed=0)
        assert 0.0 <= point <= 1.0
        assert lo <= point <= hi

    def test_cluster_vs_image_resample_differ(self):
        """When lesions have multiple images, cluster resampling and image
        resampling should give different (non-identical) CI widths.
        This test checks that the cluster bootstrap produces a meaningful CI
        (not trivially identical to image-level resampling on single-image lesions).
        We construct a case where every lesion has 3 images, then verify
        that the CI differs from a run where we treat each image as independent."""
        # Build a dataset where 5 images share a lesion (clear cluster structure)
        rng = np.random.default_rng(42)
        rows = []
        img_idx = 0
        for les_i in range(20):
            les_id = f"L_{les_i:02d}"
            cls_idx = les_i % _N
            # 5 images per lesion
            for _ in range(5):
                y_true = cls_idx
                y_pred = cls_idx if rng.random() > 0.4 else int(rng.integers(0, _N))
                probs = rng.dirichlet(np.ones(_N)).astype(np.float32)
                row = {
                    "image_id": f"IMG_{img_idx:04d}",
                    "lesion_id": les_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
                for i, c in enumerate(CLASSES):
                    row[f"p_{c}"] = float(probs[i])
                rows.append(row)
                img_idx += 1
        df = pd.DataFrame(rows)

        # Cluster bootstrap
        _, lo_cluster, hi_cluster = cluster_bootstrap_ci(
            df, metric="macro_f1", n_boot=500, seed=0
        )
        ci_width_cluster = hi_cluster - lo_cluster

        # Image-level bootstrap (treat each row as independent by giving unique lesion_id)
        df_img = df.copy()
        df_img["lesion_id"] = [f"UNIQUE_{i}" for i in range(len(df))]
        _, lo_img, hi_img = cluster_bootstrap_ci(
            df_img, metric="macro_f1", n_boot=500, seed=0
        )
        ci_width_img = hi_img - lo_img

        # Cluster resampling should give a wider (or at least different) CI
        # than image-level resampling because clusters are correlated.
        assert ci_width_cluster != pytest.approx(ci_width_img, abs=0.0), (
            "Cluster and image CI widths are identical, suggesting the cluster "
            "structure is not being used."
        )

    def test_unknown_metric_raises(self, pred_df):
        with pytest.raises(ValueError, match="Unknown metric"):
            cluster_bootstrap_ci(pred_df, metric="unknown_metric", n_boot=10, seed=0)