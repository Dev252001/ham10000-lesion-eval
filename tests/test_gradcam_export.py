"""
tests/test_gradcam_export.py – tests for gradcam.py and export.py.

Grad-CAM overlay tests are placed behind pytest.importorskip("pytorch_grad_cam")
so they are skipped cleanly in environments where the package is absent.

select_examples tests run without any optional dependency.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skinlesion.data import CLASSES
from skinlesion.export import make_outputs_zip
from skinlesion.gradcam import select_examples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pred_df(n_per_class: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthetic predictions DataFrame with known correct/incorrect counts."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls_idx, _ in enumerate(CLASSES):
        for i in range(n_per_class):
            image_id = f"SYN_{cls_idx:02d}_{i:04d}"
            y_true = cls_idx
            # First half correct, second half wrong (predict next class).
            if i < n_per_class // 2:
                y_pred = cls_idx
            else:
                y_pred = (cls_idx + 1) % len(CLASSES)
            rows.append({"image_id": image_id, "y_true": y_true, "y_pred": y_pred})
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# select_examples
# ---------------------------------------------------------------------------

class TestSelectExamples:
    @pytest.fixture
    def pred_df(self):
        return _make_pred_df(n_per_class=10)

    def test_correct_only_returns_correct(self, pred_df):
        cls = "mel"
        result = select_examples(pred_df, cls, n=5, seed=0, correct=True)
        cls_idx = CLASSES.index(cls)
        assert (result["y_true"] == cls_idx).all()
        assert (result["y_pred"] == cls_idx).all()

    def test_incorrect_only_returns_incorrect(self, pred_df):
        cls = "mel"
        result = select_examples(pred_df, cls, n=5, seed=0, correct=False)
        cls_idx = CLASSES.index(cls)
        assert (result["y_true"] == cls_idx).all()
        assert (result["y_pred"] != cls_idx).all()

    def test_deterministic_same_seed(self, pred_df):
        r1 = select_examples(pred_df, "nv", n=3, seed=42, correct=True)
        r2 = select_examples(pred_df, "nv", n=3, seed=42, correct=True)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True),
                                      r2.reset_index(drop=True))

    def test_different_seeds_may_differ(self, pred_df):
        r1 = select_examples(pred_df, "nv", n=3, seed=0, correct=True)
        r2 = select_examples(pred_df, "nv", n=3, seed=99, correct=True)
        # With 5 correct rows and n=3, different seeds produce different orders.
        # They could theoretically agree; we just check the function runs.
        assert len(r1) == 3
        assert len(r2) == 3

    def test_returns_at_most_n(self, pred_df):
        result = select_examples(pred_df, "mel", n=100, seed=0, correct=True)
        cls_idx = CLASSES.index("mel")
        available = ((pred_df["y_true"] == cls_idx) & (pred_df["y_pred"] == cls_idx)).sum()
        assert len(result) == available  # capped at available

    def test_warning_when_too_few(self, pred_df):
        # There are only n_per_class//2 = 5 correct for any class; request 100.
        with pytest.warns(UserWarning, match="returning all of them"):
            result = select_examples(pred_df, "mel", n=100, seed=0, correct=True)
        assert len(result) <= 5

    def test_handles_zero_examples_gracefully(self):
        # DataFrame with no rows for a class at all.
        df = pd.DataFrame({
            "y_true": [0, 0],
            "y_pred": [0, 0],
            "image_id": ["SYN_0000", "SYN_0001"],
        })
        with pytest.warns(UserWarning):
            result = select_examples(df, "mel", n=3, seed=0, correct=False)
        assert len(result) == 0

    def test_unknown_class_raises(self, pred_df):
        with pytest.raises(ValueError, match="Unknown class"):
            select_examples(pred_df, "unknown_class", n=1, seed=0, correct=True)

    def test_result_is_subset_of_input(self, pred_df):
        result = select_examples(pred_df, "bcc", n=3, seed=7, correct=True)
        for idx in result.index:
            assert result.loc[idx, "image_id"] in pred_df["image_id"].values


# ---------------------------------------------------------------------------
# Grad-CAM overlay (requires pytorch_grad_cam)
# ---------------------------------------------------------------------------

class TestGradCAMOverlay:
    """Tests that actually call make_gradcam_figure.

    Skipped automatically when pytorch_grad_cam is not installed.
    """

    @pytest.fixture(autouse=True)
    def _require_gradcam(self):
        pytest.importorskip("pytorch_grad_cam")

    def test_overlay_creates_png_and_index(self, tmp_path):
        """make_gradcam_figure writes PNGs and appends to gradcam_index.csv."""
        import torch
        from PIL import Image
        from skinlesion.gradcam import make_gradcam_figure
        from skinlesion.models import build_model

        # Build a tiny model (no pretrained weights).
        model = build_model("baseline_cnn", pretrained=False)
        model.eval()

        # Create two synthetic JPEG images.
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        image_paths = {}
        pred_rows = []
        cls = "mel"
        cls_idx = CLASSES.index(cls)
        for i in range(2):
            stem = f"SYN_{i:04d}"
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            p = img_dir / f"{stem}.jpg"
            img.save(str(p))
            image_paths[stem] = p
            pred_rows.append({
                "image_id": stem,
                "y_true": cls_idx,
                "y_pred": cls_idx,  # all "correct"
            })

        pred_df = pd.DataFrame(pred_rows)
        figures_dir = tmp_path / "figures"

        make_gradcam_figure(
            model=model,
            model_name="baseline_cnn",
            pred_df=pred_df,
            cls=cls,
            n=1,
            seed=0,
            image_paths=image_paths,
            figures_dir=figures_dir,
            device=torch.device("cpu"),
            image_size=32,
        )

        pngs = list(figures_dir.glob("*.png"))
        assert len(pngs) >= 1, "Expected at least one PNG"

        index_path = figures_dir / "gradcam_index.csv"
        assert index_path.exists(), "gradcam_index.csv not created"

        import csv
        with open(index_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["filename", "model", "class", "result",
                            "y_true", "y_pred", "confidence"]
        assert len(rows) >= 2  # header + at least one data row

    def test_index_appends_not_overwrites(self, tmp_path):
        """Calling make_gradcam_figure twice appends to the index."""
        import torch
        from PIL import Image
        from skinlesion.gradcam import make_gradcam_figure
        from skinlesion.models import build_model

        model = build_model("baseline_cnn", pretrained=False)
        model.eval()

        cls = "nv"
        cls_idx = CLASSES.index(cls)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        image_paths = {}
        pred_rows = []
        for i in range(2):
            stem = f"SYN_{i:04d}"
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            p = img_dir / f"{stem}.jpg"
            img.save(str(p))
            image_paths[stem] = p
            pred_rows.append({
                "image_id": stem, "y_true": cls_idx, "y_pred": cls_idx,
            })
        pred_df = pd.DataFrame(pred_rows)
        figures_dir = tmp_path / "figures"

        make_gradcam_figure(
            model=model, model_name="baseline_cnn",
            pred_df=pred_df, cls=cls, n=1, seed=0,
            image_paths=image_paths, figures_dir=figures_dir,
            device=torch.device("cpu"), image_size=32,
        )
        make_gradcam_figure(
            model=model, model_name="baseline_cnn",
            pred_df=pred_df, cls=cls, n=1, seed=1,
            image_paths=image_paths, figures_dir=figures_dir,
            device=torch.device("cpu"), image_size=32,
        )

        index_path = figures_dir / "gradcam_index.csv"
        import csv
        with open(index_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # One header + entries from both calls.
        header_count = sum(1 for r in rows if r == rows[0])
        assert header_count == 1, "Header should appear only once"
        assert len(rows) > 2, "Should have data rows from both calls"


# ---------------------------------------------------------------------------
# make_outputs_zip
# ---------------------------------------------------------------------------

class TestMakeOutputsZip:
    def _setup_repo(self, tmp_path: Path) -> Path:
        """Create a minimal fake repo structure."""
        repo = tmp_path / "repo"
        outputs = repo / "outputs"
        for sub in ("metrics", "predictions", "figures", "splits"):
            (outputs / sub).mkdir(parents=True)
        (outputs / "models").mkdir()
        # Put files in included dirs.
        (outputs / "metrics" / "run1.json").write_text('{"a":1}', encoding="utf-8")
        (outputs / "predictions" / "run1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (outputs / "figures" / "fig1.png").write_bytes(b"\x89PNG")
        (outputs / "splits" / "manifest.csv").write_text("x\n", encoding="utf-8")
        # This should NOT be included.
        (outputs / "models" / "ckpt.pt").write_bytes(b"\x00\x01")
        # dataset_stats.json at outputs root.
        (outputs / "dataset_stats.json").write_text('{"n":7}', encoding="utf-8")
        return repo

    def test_zip_created(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        zp = tmp_path / "out.zip"
        result = make_outputs_zip(repo, zp)
        assert result.exists()
        assert zipfile.is_zipfile(result)

    def test_included_files_present(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        zp = tmp_path / "out.zip"
        make_outputs_zip(repo, zp)
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        assert "outputs/metrics/run1.json" in names
        assert "outputs/predictions/run1.csv" in names
        assert "outputs/figures/fig1.png" in names
        assert "outputs/splits/manifest.csv" in names
        assert "outputs/dataset_stats.json" in names

    def test_models_dir_excluded(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        zp = tmp_path / "out.zip"
        make_outputs_zip(repo, zp)
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        assert not any("models" in n for n in names)

    def test_arcnames_start_with_outputs(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        zp = tmp_path / "out.zip"
        make_outputs_zip(repo, zp)
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                assert name.startswith("outputs/"), f"Bad arcname: {name}"

    def test_gitkeep_excluded(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        (repo / "outputs" / "metrics" / ".gitkeep").write_text("", encoding="utf-8")
        zp = tmp_path / "out.zip"
        make_outputs_zip(repo, zp)
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        assert not any(".gitkeep" in n for n in names)

    def test_missing_dataset_stats_ok(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        (repo / "outputs" / "dataset_stats.json").unlink()
        zp = tmp_path / "out.zip"
        make_outputs_zip(repo, zp)
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        assert "outputs/dataset_stats.json" not in names

    def test_empty_outputs_produces_empty_zip(self, tmp_path):
        repo = tmp_path / "repo"
        for sub in ("metrics", "predictions", "figures", "splits"):
            (repo / "outputs" / sub).mkdir(parents=True)
        zp = tmp_path / "out.zip"
        make_outputs_zip(repo, zp)
        with zipfile.ZipFile(zp) as zf:
            assert zf.namelist() == []

    def test_zip_path_parent_created(self, tmp_path):
        repo = self._setup_repo(tmp_path)
        zp = tmp_path / "nested" / "deep" / "out.zip"
        make_outputs_zip(repo, zp)
        assert zp.exists()
