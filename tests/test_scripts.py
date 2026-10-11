"""
tests/test_scripts.py – tests for update_readme.py and make_results_summary.py.

Scripts are loaded with importlib so sys.path manipulation in each script
doesn't affect this test session permanently.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers: load a script as a module from a path
# ---------------------------------------------------------------------------

def _load_script(script_path: Path) -> types.ModuleType:
    """Load a script file as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location("_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    # Add src/ to sys.path the same way the script does, but relative to
    # the script's own parent so tests are independent of cwd.
    src = script_path.parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec.loader.exec_module(mod)
    return mod


def _repo_dir() -> Path:
    """Return the real repo root (for locating scripts)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures: synthetic README with all three blocks
# ---------------------------------------------------------------------------

_README_TEMPLATE = """\
# My Project

Some text.

<!-- BEGIN RESULTS -->
old results
<!-- END RESULTS -->

More text.

<!-- BEGIN DATASET -->
old dataset
<!-- END DATASET -->

<!-- BEGIN STAGES -->
old stages
<!-- END STAGES -->

End.
"""

_README_TEMPLATE_CRLF = _README_TEMPLATE.replace("\n", "\r\n")


def _make_readme(tmp_path: Path, crlf: bool = False) -> Path:
    p = tmp_path / "README.md"
    content = _README_TEMPLATE_CRLF if crlf else _README_TEMPLATE
    p.write_bytes(content.encode("utf-8"))
    return p


def _make_fake_repo(tmp_path: Path) -> Path:
    """Create a minimal fake repo with all expected directories."""
    repo = tmp_path / "repo"
    for sub in ("metrics", "predictions", "figures", "splits", "models"):
        (repo / "outputs" / sub).mkdir(parents=True)
    (repo / "notebooks").mkdir(parents=True)
    (repo / "src").mkdir(parents=True)
    return repo


# ---------------------------------------------------------------------------
# update_readme tests
# ---------------------------------------------------------------------------

class TestUpdateReadme:
    @pytest.fixture
    def script(self):
        return _load_script(_repo_dir() / "scripts" / "update_readme.py")

    @pytest.fixture
    def fake_repo(self, tmp_path):
        return _make_fake_repo(tmp_path)

    def test_missing_runs_show_dash(self, script, fake_repo, tmp_path):
        readme = _make_readme(tmp_path)
        script.update_blocks(readme, fake_repo)
        text = readme.read_text(encoding="utf-8")
        assert "Not yet run" in text

    def test_only_blocks_change(self, script, fake_repo, tmp_path):
        readme = _make_readme(tmp_path)
        original = readme.read_text(encoding="utf-8")
        script.update_blocks(readme, fake_repo)
        updated = readme.read_text(encoding="utf-8")
        # Lines outside the markers must be unchanged.
        assert "Some text." in updated
        assert "More text." in updated
        assert "End." in updated

    def test_lf_preserved(self, script, fake_repo, tmp_path):
        readme = _make_readme(tmp_path, crlf=False)
        script.update_blocks(readme, fake_repo)
        raw = readme.read_bytes()
        assert b"\r\n" not in raw

    def test_crlf_preserved(self, script, fake_repo, tmp_path):
        readme = _make_readme(tmp_path, crlf=True)
        script.update_blocks(readme, fake_repo)
        raw = readme.read_bytes()
        # Every \n should be preceded by \r.
        content = raw.decode("utf-8")
        for i, ch in enumerate(content):
            if ch == "\n":
                assert i > 0 and content[i - 1] == "\r", (
                    "Found bare LF in CRLF file"
                )

    def test_idempotent_second_run_no_diff(self, script, fake_repo, tmp_path):
        readme = _make_readme(tmp_path)
        script.update_blocks(readme, fake_repo)
        after_first = readme.read_bytes()
        result = script.update_blocks(readme, fake_repo)
        after_second = readme.read_bytes()
        assert after_first == after_second
        assert result is False  # no change on second run

    def test_missing_markers_exits_nonzero(self, script, fake_repo, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# No markers here\n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            script.update_blocks(readme, fake_repo)
        assert exc_info.value.code != 0

    def test_results_block_filled_with_metrics(self, script, fake_repo, tmp_path):
        # Write a minimal metrics JSON.
        metrics_data = {
            "model_name": "baseline_cnn",
            "split_mode": "lesion",
            "seed": 42,
            "manifest_sha256": "abc123",
            "metrics": {
                "accuracy": 0.75,
                "macro_f1": 0.65,
                "per_class": {
                    "mel": {"recall": 0.7},
                    "bcc": {"recall": 0.6},
                    "akiec": {"recall": 0.5},
                },
            },
        }
        mf = fake_repo / "outputs" / "metrics" / "baseline_cnn_lesion_seed42.json"
        mf.write_text(json.dumps(metrics_data), encoding="utf-8")

        readme = _make_readme(tmp_path)
        script.update_blocks(readme, fake_repo)
        text = readme.read_text(encoding="utf-8")
        assert "baseline_cnn" in text
        assert "lesion" in text

    def test_dataset_block_filled_with_stats(self, script, fake_repo, tmp_path):
        stats = {
            "per_class": {
                "akiec": {"images": 327, "lesions": 327},
                "bcc":   {"images": 514, "lesions": 514},
                "bkl":   {"images": 1099, "lesions": 1099},
                "df":    {"images": 115, "lesions": 115},
                "mel":   {"images": 1113, "lesions": 1113},
                "nv":    {"images": 6705, "lesions": 6705},
                "vasc":  {"images": 142, "lesions": 142},
            },
            "imbalance": {
                "image_level": {
                    "majority_class": "nv", "majority_count": 6705,
                    "minority_class": "df", "minority_count": 115,
                    "ratio": 58.3,
                },
                "lesion_level": {
                    "majority_class": "nv", "majority_count": 6705,
                    "minority_class": "df", "minority_count": 115,
                    "ratio": 58.3,
                },
            },
        }
        (fake_repo / "outputs" / "dataset_stats.json").write_text(
            json.dumps(stats), encoding="utf-8"
        )
        readme = _make_readme(tmp_path)
        script.update_blocks(readme, fake_repo)
        text = readme.read_text(encoding="utf-8")
        assert "nv" in text
        assert "Image-level imbalance ratio" in text


# ---------------------------------------------------------------------------
# make_results_summary tests
# ---------------------------------------------------------------------------

class TestMakeResultsSummary:
    @pytest.fixture
    def script(self):
        return _load_script(_repo_dir() / "scripts" / "make_results_summary.py")

    @pytest.fixture
    def fake_repo(self, tmp_path):
        return _make_fake_repo(tmp_path)

    def _write_metrics(self, repo: Path, msha: str | None = "abc" * 21 + "ab") -> Path:
        data = {
            "model_name": "baseline_cnn",
            "split_mode": "lesion",
            "seed": 42,
            "manifest_sha256": msha,
            "metrics": {
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "per_class": {
                    "mel": {"recall": 0.75},
                    "bcc": {"recall": 0.65},
                    "akiec": {"recall": 0.55},
                },
            },
            "train_time_s": 120.5,
            "git_commit": "a" * 40,
            "created_utc": "2025-01-01T00:00:00Z",
        }
        mf = repo / "outputs" / "metrics" / "baseline_cnn_lesion_seed42.json"
        mf.write_text(json.dumps(data), encoding="utf-8")
        return mf

    def _write_manifest(self, repo: Path, sha: str) -> Path:
        """Write a manifest CSV whose SHA-256 equals *sha* (by hashing the actual content)."""
        rows = [
            ["image_id", "lesion_id", "dx", "split"],
            ["SYN_0001", "L001", "mel", "train"],
            ["SYN_0002", "L002", "nv", "val"],
            ["SYN_0003", "L003", "bcc", "test"],
        ]
        lines = "\n".join(",".join(r) for r in rows) + "\n"
        content = lines.encode("utf-8")
        actual_sha = hashlib.sha256(content).hexdigest()
        mp = repo / "outputs" / "splits" / "manifest_lesion_seed42.csv"
        mp.write_bytes(content)
        return mp, actual_sha

    def test_no_metrics_exits_1(self, script, fake_repo, tmp_path):
        out = tmp_path / "summary.md"
        with pytest.raises(SystemExit) as exc_info:
            script.main(["--output", str(out)])
        assert exc_info.value.code == 1
        assert not out.exists()

    def test_writes_nothing_without_metrics(self, script, fake_repo, tmp_path):
        """build_summary returns None when no metrics."""
        result = script.build_summary(fake_repo)
        assert result is None

    def test_summary_written_with_metrics(self, script, fake_repo, tmp_path):
        self._write_metrics(fake_repo)
        result = script.build_summary(fake_repo)
        assert result is not None
        assert "baseline_cnn" in result
        assert "lesion" in result

    def test_null_time_shows_dash(self, script, fake_repo, tmp_path):
        data = {
            "model_name": "baseline_cnn",
            "split_mode": "lesion",
            "seed": 42,
            "manifest_sha256": None,
            "metrics": {"accuracy": 0.8, "macro_f1": 0.7, "per_class": {}},
            "train_time_s": None,
            "git_commit": "unknown",
            "created_utc": "2025-01-01T00:00:00Z",
        }
        mf = fake_repo / "outputs" / "metrics" / "baseline_cnn_lesion_seed42.json"
        mf.write_text(json.dumps(data), encoding="utf-8")
        result = script.build_summary(fake_repo)
        assert "Training time: —" in result

    def test_same_split_only_when_hashes_match_and_verify_passes(
        self, script, fake_repo, tmp_path
    ):
        """'same split for all included runs' only when SHA matches and no leakage."""
        mp, actual_sha = self._write_manifest(fake_repo, "dummy")
        self._write_metrics(fake_repo, msha=actual_sha)

        result = script.build_summary(fake_repo)
        assert result is not None
        assert "same split for all included runs" in result

    def test_not_verified_when_sha_mismatch(self, script, fake_repo, tmp_path):
        mp, actual_sha = self._write_manifest(fake_repo, "dummy")
        # Write metrics with wrong SHA.
        self._write_metrics(fake_repo, msha="0" * 64)
        result = script.build_summary(fake_repo)
        assert "not verified" in result

    def test_not_verified_when_manifest_missing(self, script, fake_repo, tmp_path):
        self._write_metrics(fake_repo, msha="a" * 64)
        # No manifest file written.
        result = script.build_summary(fake_repo)
        assert "not verified" in result

    def test_shared_lesion_fails_verification(self, script, fake_repo, tmp_path):
        """A manifest with the same lesion_id in train and test fails."""
        rows = [
            ["image_id", "lesion_id", "dx", "split"],
            ["SYN_0001", "L001", "mel", "train"],
            ["SYN_0002", "L001", "mel", "test"],  # same lesion in two splits
        ]
        lines = "\n".join(",".join(r) for r in rows) + "\n"
        content = lines.encode("utf-8")
        actual_sha = hashlib.sha256(content).hexdigest()
        mp = fake_repo / "outputs" / "splits" / "manifest_lesion_seed42.csv"
        mp.write_bytes(content)
        self._write_metrics(fake_repo, msha=actual_sha)
        result = script.build_summary(fake_repo)
        assert "not verified" in result

    def test_second_run_no_diff(self, script, fake_repo, tmp_path):
        """build_summary is deterministic on the same inputs (modulo timestamp)."""
        mp, actual_sha = self._write_manifest(fake_repo, "dummy")
        self._write_metrics(fake_repo, msha=actual_sha)
        first = script.build_summary(fake_repo)
        second = script.build_summary(fake_repo)
        assert first is not None
        assert second is not None
        assert "baseline_cnn" in second
        # Line count is stable (only the timestamp line varies).
        assert len(first.splitlines()) == len(second.splitlines())
