"""
tests/test_splits.py
--------------------
Tests for skinlesion.splits.

Synthetic skewed table:
  nv 400 lesions, bkl 60, mel 50, bcc 30, akiec 20, vasc 12, df 8
  Each lesion has 1-4 images (deterministic, keyed on index mod 4).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skinlesion.splits import (
    MIN_LESIONS_WARN,
    audit_split,
    make_split,
    manifest_sha256,
    save_manifest,
    split_manifest,
)


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------

_LESION_COUNTS = {
    "nv": 400,
    "bkl": 60,
    "mel": 50,
    "bcc": 30,
    "akiec": 20,
    "vasc": 12,
    "df": 8,
}
# Total lesions = 580


def _build_synthetic_df(lesion_counts: dict | None = None) -> pd.DataFrame:
    """Build a synthetic DataFrame with multiple images per lesion.

    Each lesion i gets (i % 4) + 1 images (so 1, 2, 3, or 4 images).
    Image IDs look like SYN_IMG_<zero-padded-int>.
    Lesion IDs look like SYN_LES_<zero-padded-int>.
    """
    if lesion_counts is None:
        lesion_counts = _LESION_COUNTS

    rows = []
    img_counter = 0
    les_counter = 0
    for cls, n_lesions in lesion_counts.items():
        for li in range(n_lesions):
            les_id = f"SYN_LES_{les_counter:05d}"
            n_images = (les_counter % 4) + 1
            for _ in range(n_images):
                rows.append({
                    "image_id": f"SYN_IMG_{img_counter:06d}",
                    "lesion_id": les_id,
                    "dx": cls,
                })
                img_counter += 1
            les_counter += 1

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def syn_df():
    return _build_synthetic_df()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_lesions(df):
    return df["lesion_id"].nunique()


def _lesion_set(df):
    return set(df["lesion_id"])


# ---------------------------------------------------------------------------
# Tests: invalid mode
# ---------------------------------------------------------------------------

def test_invalid_mode_raises(syn_df):
    with pytest.raises(ValueError, match="split_mode"):
        make_split(syn_df, split_mode="random")


# ---------------------------------------------------------------------------
# Tests: lesion mode – zero overlap
# ---------------------------------------------------------------------------

class TestLesionModeNoOverlap:
    @pytest.fixture(scope="class")
    def splits(self, syn_df):
        return make_split(syn_df, split_mode="lesion", seed=42)

    def test_no_lesion_in_train_and_val(self, splits):
        train, val, test = splits
        assert _lesion_set(train).isdisjoint(_lesion_set(val))

    def test_no_lesion_in_train_and_test(self, splits):
        train, val, test = splits
        assert _lesion_set(train).isdisjoint(_lesion_set(test))

    def test_no_lesion_in_val_and_test(self, splits):
        train, val, test = splits
        assert _lesion_set(val).isdisjoint(_lesion_set(test))

    def test_every_image_exactly_once(self, splits, syn_df):
        train, val, test = splits
        all_ids = list(train["image_id"]) + list(val["image_id"]) + list(test["image_id"])
        assert len(all_ids) == len(syn_df)
        assert len(set(all_ids)) == len(syn_df)


# ---------------------------------------------------------------------------
# Tests: lesion mode – class fractions within tolerance
# ---------------------------------------------------------------------------

class TestLesionModeFractions:
    """
    Tolerance rationale: we split at lesion level, not image level.
    With small classes (vasc=12, df=8) stratification may not be perfect.
    We use ±5 pp for images (which depends on multi-image lesions) and a
    more generous ±8 pp for rare classes.
    """

    @pytest.fixture(scope="class")
    def splits(self, syn_df):
        return make_split(syn_df, split_mode="lesion", val_size=0.15, test_size=0.15, seed=42)

    @pytest.fixture(scope="class")
    def total_images(self, syn_df):
        return len(syn_df)

    def test_test_fraction_approx(self, splits, total_images):
        _, _, test = splits
        frac = len(test) / total_images
        assert 0.10 <= frac <= 0.22, f"test fraction {frac:.3f} out of [0.10, 0.22]"

    def test_val_fraction_approx(self, splits, total_images):
        _, val, _ = splits
        frac = len(val) / total_images
        assert 0.10 <= frac <= 0.22, f"val fraction {frac:.3f} out of [0.10, 0.22]"

    def test_per_class_fraction_in_test(self, splits, syn_df):
        """Each class should contribute roughly 15% of its images to test."""
        _, _, test = splits
        for cls in syn_df["dx"].unique():
            total = len(syn_df[syn_df["dx"] == cls])
            n_test = len(test[test["dx"] == cls])
            frac = n_test / total
            assert 0.05 <= frac <= 0.30, (
                f"class '{cls}': test fraction {frac:.3f} out of [0.05, 0.30]"
            )


# ---------------------------------------------------------------------------
# Tests: same seed -> same split; different seed -> different
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_split(self, syn_df):
        tr1, v1, te1 = make_split(syn_df, split_mode="lesion", seed=42)
        tr2, v2, te2 = make_split(syn_df, split_mode="lesion", seed=42)
        assert set(tr1["image_id"]) == set(tr2["image_id"])
        assert set(te1["image_id"]) == set(te2["image_id"])

    def test_different_seeds_differ(self, syn_df):
        tr1, _, te1 = make_split(syn_df, split_mode="lesion", seed=42)
        tr2, _, te2 = make_split(syn_df, split_mode="lesion", seed=99)
        # Very unlikely to be identical for a dataset of this size
        assert set(te1["image_id"]) != set(te2["image_id"])


# ---------------------------------------------------------------------------
# Tests: image mode shows lesion overlap > 0
# ---------------------------------------------------------------------------

class TestImageModeHasOverlap:
    def test_overlap_positive(self, syn_df):
        """
        Image-mode split on multi-image lesions must produce overlapping lesions
        (same lesion in train AND test).
        """
        train, val, test = make_split(syn_df, split_mode="image", seed=42)
        overlap = _lesion_set(train) & _lesion_set(test)
        assert len(overlap) > 0, (
            "Image-mode split should have lesion overlap between train and test"
        )


# ---------------------------------------------------------------------------
# Tests: manifest
# ---------------------------------------------------------------------------

class TestManifest:
    @pytest.fixture(scope="class")
    def manifest(self, syn_df):
        train, val, test = make_split(syn_df, split_mode="lesion", seed=42)
        return split_manifest(train, val, test, split_mode="lesion", seed=42)

    def test_columns(self, manifest):
        for col in ("image_id", "lesion_id", "dx", "split"):
            assert col in manifest.columns

    def test_all_images_present(self, manifest, syn_df):
        assert set(manifest["image_id"]) == set(syn_df["image_id"])

    def test_split_values(self, manifest):
        assert set(manifest["split"]) <= {"train", "val", "test"}

    def test_each_image_once(self, manifest):
        assert manifest["image_id"].nunique() == len(manifest)


# ---------------------------------------------------------------------------
# Tests: manifest_sha256
# ---------------------------------------------------------------------------

class TestManifestHash:
    @pytest.fixture(scope="class")
    def manifest(self, syn_df):
        train, val, test = make_split(syn_df, split_mode="lesion", seed=42)
        return split_manifest(train, val, test, split_mode="lesion", seed=42)

    def test_hash_is_64_hex(self, manifest):
        sha = manifest_sha256(manifest)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_hash_independent_of_row_order(self, manifest):
        shuffled = manifest.sample(frac=1, random_state=7).reset_index(drop=True)
        assert manifest_sha256(manifest) == manifest_sha256(shuffled)

    def test_different_splits_different_hash(self, syn_df):
        tr1, v1, te1 = make_split(syn_df, split_mode="lesion", seed=42)
        tr2, v2, te2 = make_split(syn_df, split_mode="lesion", seed=99)
        m1 = split_manifest(tr1, v1, te1, split_mode="lesion", seed=42)
        m2 = split_manifest(tr2, v2, te2, split_mode="lesion", seed=99)
        assert manifest_sha256(m1) != manifest_sha256(m2)


# ---------------------------------------------------------------------------
# Tests: save_manifest
# ---------------------------------------------------------------------------

class TestSaveManifest:
    def test_file_created(self, syn_df, tmp_path):
        train, val, test = make_split(syn_df, split_mode="lesion", seed=42)
        m = split_manifest(train, val, test, split_mode="lesion", seed=42)
        sha = manifest_sha256(m)
        p = save_manifest(m, tmp_path, split_mode="lesion", seed=42, sha=sha)
        assert p.exists()
        assert p.name == "manifest_lesion_seed42.csv"

    def test_sha_in_header(self, syn_df, tmp_path):
        train, val, test = make_split(syn_df, split_mode="lesion", seed=42)
        m = split_manifest(train, val, test, split_mode="lesion", seed=42)
        sha = manifest_sha256(m)
        p = save_manifest(m, tmp_path, split_mode="lesion", seed=42, sha=sha)
        first_line = p.read_text(encoding="utf-8").splitlines()[0]
        assert sha in first_line


# ---------------------------------------------------------------------------
# Tests: audit_split
# ---------------------------------------------------------------------------

class TestAuditSplit:
    @pytest.fixture(scope="class")
    def audit_lesion(self, syn_df):
        train, val, test = make_split(syn_df, split_mode="lesion", seed=42)
        m = split_manifest(train, val, test, split_mode="lesion", seed=42)
        return audit_split(m, split_mode="lesion", seed=42), train, val, test

    def test_no_leakage_in_lesion_mode(self, audit_lesion):
        audit, _, _, _ = audit_lesion
        lk = audit["leakage"]
        assert lk["lesions_in_train_and_test"] == 0
        assert lk["lesions_in_train_and_val"] == 0
        assert lk["test_images_with_train_lesion"] == 0

    def test_brute_force_match(self, audit_lesion, syn_df):
        """audit per_class counts must equal a brute-force recount."""
        audit, train, val, test = audit_lesion
        for cls in syn_df["dx"].unique():
            for split_name, sdf in [("train", train), ("val", val), ("test", test)]:
                rows = sdf[sdf["dx"] == cls]
                expected_imgs = len(rows)
                expected_les = rows["lesion_id"].nunique()
                got = audit["per_class"].get(cls, {}).get(split_name, {})
                assert got["images"] == expected_imgs, (
                    f"{cls}/{split_name}: images {got['images']} != {expected_imgs}"
                )
                assert got["lesions"] == expected_les, (
                    f"{cls}/{split_name}: lesions {got['lesions']} != {expected_les}"
                )

    def test_image_mode_has_positive_leakage(self, syn_df):
        train, val, test = make_split(syn_df, split_mode="image", seed=42)
        m = split_manifest(train, val, test, split_mode="image", seed=42)
        a = audit_split(m, split_mode="image", seed=42)
        assert a["leakage"]["test_images_with_train_lesion"] > 0
        assert a["leakage"]["test_share_with_train_lesion"] > 0

    def test_audit_written_to_file(self, syn_df, tmp_path):
        train, val, test = make_split(syn_df, split_mode="lesion", seed=42)
        m = split_manifest(train, val, test, split_mode="lesion", seed=42)
        audit_split(m, split_mode="lesion", seed=42, outputs_dir=tmp_path)
        p = tmp_path / "splits" / "audit_lesion_seed42.json"
        assert p.exists()


# ---------------------------------------------------------------------------
# Tests: warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_warning_for_sparse_class(self):
        """Very small classes (vasc=12, df=8) should trigger MIN_LESIONS_WARN."""
        df = _build_synthetic_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_split(df, split_mode="lesion", seed=42)
        warning_texts = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        # vasc (12) and df (8) are below MIN_LESIONS_WARN=20; at least one warning expected
        assert len(warning_texts) >= 1, (
            f"Expected at least one UserWarning for sparse classes, got: {warning_texts}"
        )
        # At least one of the known sparse classes should appear in a warning
        sparse_classes = {"vasc", "df"}
        warned_classes = {cls for cls in sparse_classes
                          if any(cls in t for t in warning_texts)}
        assert len(warned_classes) > 0, (
            f"Expected warning mentioning vasc or df, got: {warning_texts}"
        )

    def test_no_warning_for_dense_dataset(self, capsys):
        """A balanced large dataset with all classes > MIN_LESIONS_WARN should not warn."""
        # 200 * 0.15 = 30 lesions per class in val/test, comfortably above MIN_LESIONS_WARN=20
        big_counts = {cls: 200 for cls in ["nv", "bkl", "mel", "bcc", "akiec", "vasc", "df"]}
        df = _build_synthetic_df(big_counts)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_split(df, split_mode="lesion", seed=42)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, (
            f"Expected no UserWarnings, got: {[str(x.message) for x in user_warnings]}"
        )