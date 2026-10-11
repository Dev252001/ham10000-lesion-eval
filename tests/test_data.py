"""
tests/test_data.py
------------------
Tests for skinlesion.data.

All fixtures are synthetic (image IDs like SYN_0001).
Nothing is written to outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skinlesion.data import (
    CLASSES,
    MALIGNANT,
    HAMDataset,
    dataset_stats,
    load_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_jpg(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    """Create a minimal 16x16 RGB JPEG at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(128, 64, 32))
    img.save(path, format="JPEG")


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Synthetic image records – 7 classes, some lesions have multiple images.
# IDs look obviously synthetic (SYN_XXXX prefix).
_RECORDS = [
    {"image_id": "SYN_0001", "lesion_id": "SYN_L001", "dx": "mel"},
    {"image_id": "SYN_0002", "lesion_id": "SYN_L001", "dx": "mel"},   # 2nd image, same lesion
    {"image_id": "SYN_0003", "lesion_id": "SYN_L002", "dx": "bcc"},
    {"image_id": "SYN_0004", "lesion_id": "SYN_L003", "dx": "akiec"},
    {"image_id": "SYN_0005", "lesion_id": "SYN_L004", "dx": "bkl"},
    {"image_id": "SYN_0006", "lesion_id": "SYN_L005", "dx": "df"},
    {"image_id": "SYN_0007", "lesion_id": "SYN_L006", "dx": "nv"},
    {"image_id": "SYN_0008", "lesion_id": "SYN_L006", "dx": "nv"},   # 2nd image
    {"image_id": "SYN_0009", "lesion_id": "SYN_L007", "dx": "vasc"},
]


@pytest.fixture()
def syn_data_dir(tmp_path: Path) -> Path:
    """Flat layout with one nested subfolder to test recursive globbing."""
    sub = tmp_path / "part1"
    sub.mkdir()

    # Place some images in a sub-folder, some directly in tmp_path
    placements = {
        "SYN_0001": sub / "SYN_0001.JPG",   # upper-case extension
        "SYN_0002": sub / "SYN_0002.jpg",
        "SYN_0003": tmp_path / "SYN_0003.jpg",
        "SYN_0004": tmp_path / "SYN_0004.JPEG",  # .JPEG variant
        "SYN_0005": sub / "SYN_0005.jpg",
        "SYN_0006": sub / "SYN_0006.jpg",
        "SYN_0007": tmp_path / "SYN_0007.jpg",
        "SYN_0008": tmp_path / "SYN_0008.jpg",
        "SYN_0009": tmp_path / "SYN_0009.jpg",
    }
    for path in placements.values():
        _make_tiny_jpg(path)

    _write_csv(tmp_path / "HAM10000_metadata.csv", _RECORDS)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_classes_sorted(self):
        assert list(CLASSES) == sorted(CLASSES)

    def test_classes_count(self):
        assert len(CLASSES) == 7

    def test_malignant_subset(self):
        assert set(MALIGNANT) <= set(CLASSES)

    def test_malignant_contents(self):
        assert set(MALIGNANT) == {"mel", "bcc", "akiec"}


# ---------------------------------------------------------------------------
# Tests: load_metadata – happy path
# ---------------------------------------------------------------------------

class TestLoadMetadataHappy:
    def test_row_count(self, syn_data_dir):
        df = load_metadata(syn_data_dir)
        assert len(df) == len(_RECORDS)

    def test_image_path_column_exists(self, syn_data_dir):
        df = load_metadata(syn_data_dir)
        assert "image_path" in df.columns

    def test_all_paths_are_files(self, syn_data_dir):
        df = load_metadata(syn_data_dir)
        for p in df["image_path"]:
            assert Path(p).is_file(), f"Not a file: {p}"

    def test_recursive_glob_finds_nested(self, syn_data_dir):
        """Images in sub-folder must be discovered."""
        df = load_metadata(syn_data_dir)
        nested = [str(p) for p in df["image_path"] if "part1" in str(p)]
        assert len(nested) > 0

    def test_mixed_case_extension(self, syn_data_dir):
        """Upper-case .JPG and .JPEG extensions must be matched."""
        df = load_metadata(syn_data_dir)
        assert df["image_path"].notna().all()


# ---------------------------------------------------------------------------
# Tests: load_metadata – error paths
# ---------------------------------------------------------------------------

class TestLoadMetadataErrors:
    def test_missing_image_raises(self, syn_data_dir, tmp_path):
        """A metadata row whose image file is absent should raise FileNotFoundError."""
        # Add an extra row pointing to a non-existent file
        extra = _RECORDS + [
            {"image_id": "SYN_MISSING", "lesion_id": "SYN_L999", "dx": "nv"}
        ]
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        # Copy images from syn_data_dir fixture manually
        for rec in _RECORDS:
            src = next(syn_data_dir.rglob(f"{rec['image_id']}.*"))
            dst = bad_dir / src.name
            import shutil; shutil.copy(src, dst)
        _write_csv(bad_dir / "HAM10000_metadata.csv", extra)

        with pytest.raises(FileNotFoundError, match="SYN_MISSING"):
            load_metadata(bad_dir)

    def test_two_dx_same_lesion_raises(self, syn_data_dir, tmp_path):
        """A lesion_id appearing with two different dx values should raise ValueError."""
        bad_records = _RECORDS + [
            # Same lesion_id as SYN_L001 but different dx
            {"image_id": "SYN_0010", "lesion_id": "SYN_L001", "dx": "bcc"},
        ]
        bad_dir = tmp_path / "bad2"
        bad_dir.mkdir()
        import shutil
        for rec in _RECORDS:
            src = next(syn_data_dir.rglob(f"{rec['image_id']}.*"))
            dst = bad_dir / src.name
            shutil.copy(src, dst)
        # Create a dummy image for SYN_0010
        _make_tiny_jpg(bad_dir / "SYN_0010.jpg")
        _write_csv(bad_dir / "HAM10000_metadata.csv", bad_records)

        with pytest.raises(ValueError, match="lesion_id"):
            load_metadata(bad_dir)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="HAM10000_metadata.csv"):
            load_metadata(tmp_path)


# ---------------------------------------------------------------------------
# Tests: HAMDataset
# ---------------------------------------------------------------------------

class TestHAMDataset:
    def test_len(self, syn_data_dir):
        torch = pytest.importorskip("torch")
        df = load_metadata(syn_data_dir)
        ds = HAMDataset(df)
        assert len(ds) == len(df)

    def test_item_shape_default(self, syn_data_dir):
        torch = pytest.importorskip("torch")
        df = load_metadata(syn_data_dir)
        ds = HAMDataset(df)
        tensor, label = ds[0]
        assert tensor.shape[0] == 3          # RGB channels
        assert tensor.dtype == torch.float32  # to_tensor gives float32 in [0,1]
        assert isinstance(label, int)

    def test_label_in_range(self, syn_data_dir):
        torch = pytest.importorskip("torch")
        df = load_metadata(syn_data_dir)
        ds = HAMDataset(df)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label < len(CLASSES)

    def test_transform_applied(self, syn_data_dir):
        torch = pytest.importorskip("torch")
        import torchvision.transforms as T
        df = load_metadata(syn_data_dir)
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        ds = HAMDataset(df, transform=transform)
        tensor, _ = ds[0]
        assert tensor.shape == torch.Size([3, 32, 32])


# ---------------------------------------------------------------------------
# Tests: dataset_stats – hand-counted reference table
# ---------------------------------------------------------------------------

class TestDatasetStats:
    """
    Hand-counted reference (from _RECORDS above):
      mel:   2 images, 1 lesion  (SYN_L001 x2)
      bcc:   1 image,  1 lesion
      akiec: 1 image,  1 lesion
      bkl:   1 image,  1 lesion
      df:    1 image,  1 lesion
      nv:    2 images, 1 lesion  (SYN_L006 x2)
      vasc:  1 image,  1 lesion
    Totals: 9 images, 7 lesions

    Image-level: majority=mel|nv (2), minority=any 1-image class (1)
      -> imbalance_ratio = 2.0
    Lesion-level: every class has 1 lesion -> imbalance_ratio = 1.0
    """

    @pytest.fixture()
    def stats(self):
        df = pd.DataFrame(_RECORDS)
        return dataset_stats(df)

    def test_per_class_image_counts(self, stats):
        pc = stats["per_class"]
        assert pc["mel"]["images"] == 2
        assert pc["bcc"]["images"] == 1
        assert pc["nv"]["images"] == 2
        assert pc["vasc"]["images"] == 1

    def test_per_class_lesion_counts(self, stats):
        pc = stats["per_class"]
        # Every class has exactly 1 unique lesion in the synthetic table
        for cls in CLASSES:
            assert pc[cls]["lesions"] == 1, f"{cls}: expected 1 lesion"

    def test_image_imbalance_ratio(self, stats):
        il = stats["image_level"]
        assert il["majority_count"] == 2
        assert il["minority_count"] == 1
        assert il["imbalance_ratio"] == pytest.approx(2.0)

    def test_lesion_imbalance_ratio(self, stats):
        ll = stats["lesion_level"]
        # All classes have 1 lesion -> ratio is 1.0
        assert ll["imbalance_ratio"] == pytest.approx(1.0)

    def test_json_serialisable(self, stats):
        # Must not raise
        json.dumps(stats)

    def test_all_classes_present(self, stats):
        for cls in CLASSES:
            assert cls in stats["per_class"]