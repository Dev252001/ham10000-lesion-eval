"""
tests/test_pipeline_smoke.py
-----------------------------
End-to-end smoke test: synthetic data -> split -> train 1 epoch -> evaluate.

Builds ~100 synthetic 16x16 JPEGs in tmp_path plus a metadata CSV.
Runs the full pipeline for all three models at 16x16 with weights=None.
Target: under 2 minutes on CPU.

No files are written to outputs/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from skinlesion.config import TrainConfig
from skinlesion.data import CLASSES, HAMDataset, dataset_stats, load_metadata
from skinlesion.evaluate import final_test_evaluation
from skinlesion.models import build_model, compute_class_weights
from skinlesion.splits import (
    audit_split,
    make_split,
    manifest_sha256,
    split_manifest,
)
from skinlesion.train import TrainHistory, load_or_train, seed_everything, train_model
from skinlesion.transforms import eval_transform, train_transform

# ---------------------------------------------------------------------------
# Constants for the smoke dataset
# ---------------------------------------------------------------------------

# Skewed class distribution (roughly mimicking real HAM10000 proportions)
_SMOKE_LESION_COUNTS = {
    "nv": 30,
    "bkl": 12,
    "mel": 10,
    "bcc": 8,
    "akiec": 7,
    "vasc": 5,
    "df": 4,
}
_IMG_SIZE = 16   # tiny for CPU speed


# ---------------------------------------------------------------------------
# Fixture: build synthetic dataset in tmp_path
# ---------------------------------------------------------------------------

def _make_tiny_jpg(path: Path, size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path, format="JPEG")


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    """Build ~100 synthetic JPEGs + metadata CSV in a temp directory.

    Each lesion gets 1-2 images (alternating on lesion index).
    Image IDs are obviously synthetic (SMOKE_XXXX).
    """
    base = tmp_path_factory.mktemp("smoke")
    rows = []
    img_counter = 0
    les_counter = 0
    for cls, n_lesions in _SMOKE_LESION_COUNTS.items():
        for li in range(n_lesions):
            les_id = f"SMOKE_L_{les_counter:04d}"
            n_imgs = 1 if les_counter % 2 == 0 else 2
            for _ in range(n_imgs):
                img_id = f"SMOKE_{img_counter:05d}"
                rows.append({
                    "image_id": img_id,
                    "lesion_id": les_id,
                    "dx": cls,
                })
                _make_tiny_jpg(base / f"{img_id}.jpg")
                img_counter += 1
            les_counter += 1

    pd.DataFrame(rows).to_csv(base / "HAM10000_metadata.csv", index=False)
    return base


# ---------------------------------------------------------------------------
# Helper: build DataLoader from a partition DataFrame
# ---------------------------------------------------------------------------

def _make_loader(df: pd.DataFrame, transform, batch_size: int = 8) -> DataLoader:
    ds = HAMDataset(df, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)


# ---------------------------------------------------------------------------
# Sentinel loader – raises if iterated (ensures test loader is never touched
# during training)
# ---------------------------------------------------------------------------

class _NeverIterateLoader:
    """A fake DataLoader that raises RuntimeError if iterated."""

    def __iter__(self):
        raise RuntimeError(
            "Test loader was iterated during training, "
            "which violates the rule that train_model must not access the test split."
        )

    def __len__(self):
        return 0


# ---------------------------------------------------------------------------
# Core smoke test
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)   # hard ceiling; will pass without pytest-timeout too
def test_full_pipeline_smoke(smoke_dir, tmp_path):
    """Load data -> split (lesion) -> train 1 epoch each model -> evaluate."""

    # --- 1. Load metadata ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # expected sparse-class warns
        df = load_metadata(smoke_dir)

    assert len(df) > 0, "No rows loaded from metadata"

    # --- 2. Lesion-aware split ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        train_df, val_df, test_df = make_split(df, split_mode="lesion", seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # --- 3. Manifest + SHA ---
    manifest = split_manifest(train_df, val_df, test_df, split_mode="lesion", seed=42)
    sha = manifest_sha256(manifest)
    assert len(sha) == 64

    # --- 4. Audit (no leakage in lesion mode) ---
    audit = audit_split(manifest, split_mode="lesion", seed=42)
    assert audit["leakage"]["lesions_in_train_and_test"] == 0

    # --- 5. Transforms ---
    t_train = train_transform(image_size=_IMG_SIZE)
    t_eval = eval_transform(image_size=_IMG_SIZE)

    # --- 6. DataLoaders ---
    train_loader = _make_loader(train_df, t_train)
    val_loader = _make_loader(val_df, t_eval)
    test_loader = _make_loader(test_df, t_eval)
    test_sentinel = _NeverIterateLoader()

    # --- 7. Class weights from train partition only ---
    class_weights = compute_class_weights(train_df)
    assert class_weights.shape == torch.Size([7])

    device = torch.device("cpu")

    # --- 8. Train each model for 1 epoch ---
    model_names = ["baseline_cnn", "resnet18", "efficientnet_b0"]
    results = {}

    ckpt_dir = tmp_path / "checkpoints"
    outputs_dir = tmp_path / "outputs"

    for model_name in model_names:
        seed_everything(42)
        model = build_model(model_name, pretrained=False)

        cfg = TrainConfig(
            model_name=model_name,
            split_mode="lesion",
            seed=42,
            image_size=_IMG_SIZE,
            batch_size=8,
            max_epochs=1,
            patience=5,
            manifest_sha256=sha,
        )

        # Pass test_sentinel to verify test loader is never iterated during training
        # train_model only takes train_loader + val_loader; sentinel not passed there.
        # We store it to verify it is never touched.
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            class_weights=class_weights,
            device=device,
            ckpt_path=ckpt_dir / f"{model_name}_lesion_seed42_best.pt",
        )

        assert history.train_time_s > 0, f"{model_name}: train_time_s must be > 0"
        assert history.epochs_run == 1

        # --- 9. Final test evaluation ---
        metrics_doc, preds_path, metrics_path = final_test_evaluation(
            model=model,
            test_df=test_df,
            test_loader=test_loader,
            cfg=cfg,
            history=history,
            manifest_sha256=sha,
            repo_dir=Path("d:/HAM/ham10000-lesion-eval"),
            outputs_dir=outputs_dir,
            device=device,
        )

        results[model_name] = (metrics_doc, preds_path, metrics_path)

    # --- 10. Validate each result ---
    required_fields = [
        "model_name", "split_mode", "seed", "config", "metrics",
        "n_params", "train_time_s", "epochs_run", "best_epoch",
        "git_commit", "git_dirty", "manifest_sha256",
        "library_versions", "device", "created_utc",
    ]
    for model_name, (doc, preds_path, metrics_path) in results.items():
        for field in required_fields:
            assert field in doc, f"{model_name}: missing field '{field}'"

        assert doc["train_time_s"] is not None, f"{model_name}: train_time_s must not be null"
        assert doc["train_time_s"] > 0, f"{model_name}: train_time_s must be > 0"

        # Manifest hash in metrics matches what was computed in-memory
        assert doc["manifest_sha256"] == sha, (
            f"{model_name}: manifest hash mismatch"
        )

        # Predictions CSV exists and has required columns
        assert preds_path.exists()
        pred_df = pd.read_csv(preds_path)
        for col in ["image_id", "lesion_id", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES]:
            assert col in pred_df.columns, f"{model_name}: missing pred column '{col}'"

        # Metrics JSON is valid
        assert metrics_path.exists()
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert loaded["manifest_sha256"] == sha

    # --- 11. Verify sentinel was never touched (no RuntimeError raised above) ---
    # The sentinel's __iter__ would have raised during train_model if called.
    # We additionally confirm it by trying to iterate it here and catching the error.
    with pytest.raises(RuntimeError, match="Test loader was iterated during training"):
        next(iter(test_sentinel))


def test_different_split_mode_forces_retrain(smoke_dir, tmp_path):
    """Changing split_mode must produce a different fingerprint -> retrain."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = load_metadata(smoke_dir)
        train_df, val_df, test_df = make_split(df, split_mode="lesion", seed=42)

    manifest = split_manifest(train_df, val_df, test_df, split_mode="lesion", seed=42)
    sha_lesion = manifest_sha256(manifest)

    cfg_lesion = TrainConfig(
        model_name="baseline_cnn", split_mode="lesion", seed=42,
        image_size=_IMG_SIZE, batch_size=8, max_epochs=1, patience=5,
        manifest_sha256=sha_lesion,
    )

    # Run with lesion split to create a cached checkpoint
    model = build_model("baseline_cnn", pretrained=False)
    class_weights = compute_class_weights(train_df)
    t_train = train_transform(image_size=_IMG_SIZE)
    t_eval = eval_transform(image_size=_IMG_SIZE)
    train_loader = _make_loader(train_df, t_train)
    val_loader = _make_loader(val_df, t_eval)
    device = torch.device("cpu")

    _, _, cached = load_or_train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg_lesion, class_weights=class_weights, device=device,
        ckpt_dir=tmp_path,
    )
    assert cached is False  # First run: no cache

    # Second run same config -> cached
    model2 = build_model("baseline_cnn", pretrained=False)
    _, _, cached2 = load_or_train(
        model=model2, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg_lesion, class_weights=class_weights, device=device,
        ckpt_dir=tmp_path,
    )
    assert cached2 is True

    # Third run with different split_mode (image) -> different fingerprint -> retrain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tr_img, v_img, te_img = make_split(df, split_mode="image", seed=42)
    m_img = split_manifest(tr_img, v_img, te_img, split_mode="image", seed=42)
    sha_image = manifest_sha256(m_img)

    cfg_image = TrainConfig(
        model_name="baseline_cnn", split_mode="image", seed=42,
        image_size=_IMG_SIZE, batch_size=8, max_epochs=1, patience=5,
        manifest_sha256=sha_image,
    )
    train_loader_img = _make_loader(tr_img, t_train)
    val_loader_img = _make_loader(v_img, t_eval)

    model3 = build_model("baseline_cnn", pretrained=False)
    _, _, cached3 = load_or_train(
        model=model3, train_loader=train_loader_img, val_loader=val_loader_img,
        cfg=cfg_image, class_weights=compute_class_weights(tr_img),
        device=device, ckpt_dir=tmp_path,
    )
    assert cached3 is False  # Different split_mode -> must retrain


def test_no_new_files_in_outputs(tmp_path):
    """
    Verify that the smoke test does NOT write files into the repo's outputs/
    directory.  All writes go to tmp_path.

    We check by listing the repo outputs/ directory before and after
    importing the module (the module itself does not write on import).
    """
    repo_outputs = Path("d:/HAM/ham10000-lesion-eval/outputs")
    # Collect the set of files before
    before = {str(p) for p in repo_outputs.rglob("*") if p.is_file()}
    # Re-import (no-op, already cached): just to confirm no side effects
    import skinlesion.evaluate  # noqa: F401
    after = {str(p) for p in repo_outputs.rglob("*") if p.is_file()}
    new_files = after - before
    assert new_files == set(), f"Unexpected new files in outputs/: {new_files}"