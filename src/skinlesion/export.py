"""
export.py – package outputs for download / archival.

Public API
----------
make_outputs_zip : zip outputs/metrics, predictions, figures, splits
                   and dataset_stats.json; never outputs/models.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


# Subdirectories of outputs/ that are included in the zip.
_INCLUDED_SUBDIRS = ("metrics", "predictions", "figures", "splits")


def make_outputs_zip(repo_dir: Path, zip_path: Path) -> Path:
    """Create a zip of tracked output files for download.

    Included:
    - outputs/metrics/**
    - outputs/predictions/**
    - outputs/figures/**
    - outputs/splits/**
    - outputs/dataset_stats.json (if it exists)

    Explicitly excluded:
    - outputs/models/  (checkpoints are large and not tracked)

    Arc-names inside the zip start with "outputs/" so the zip can be
    unzipped at the root of the local clone without path confusion.

    Parameters
    ----------
    repo_dir:
        Root of the git repository.
    zip_path:
        Destination path for the zip file (created or overwritten).

    Returns
    -------
    zip_path (resolved).
    """
    repo_dir = Path(repo_dir).resolve()
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    outputs_dir = repo_dir / "outputs"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Include tracked sub-directories.
        for subdir_name in _INCLUDED_SUBDIRS:
            subdir = outputs_dir / subdir_name
            if not subdir.exists():
                continue
            for fpath in sorted(subdir.rglob("*")):
                if fpath.is_file() and fpath.name != ".gitkeep":
                    arcname = "outputs/" + fpath.relative_to(outputs_dir).as_posix()
                    zf.write(fpath, arcname)

        # Include dataset_stats.json if present.
        stats_path = outputs_dir / "dataset_stats.json"
        if stats_path.exists():
            zf.write(stats_path, "outputs/dataset_stats.json")

    return zip_path.resolve()
