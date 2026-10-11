"""
scripts/update_readme.py – fill three marker blocks in README.md.

Marker format in README.md:
    <!-- BEGIN RESULTS -->
    (content replaced on each run)
    <!-- END RESULTS -->

    <!-- BEGIN DATASET -->
    (content replaced on each run)
    <!-- END DATASET -->

    <!-- BEGIN STAGES -->
    (content replaced on each run)
    <!-- END STAGES -->

Usage:
    python scripts/update_readme.py [--readme PATH]

The script is idempotent: running it twice with the same data produces
no diff.  It preserves the file's existing line ending style (CRLF or LF).
It exits with code 1 and prints a message if any marker is missing.
No emoji are used.

All numbers come from outputs/ files written by real runs; "—" is written
for missing runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Add src/ to sys.path so the script works without installation.
_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR / "src"))

from skinlesion.data import CLASSES, MALIGNANT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _outputs(repo: Path) -> Path:
    return repo / "outputs"


# ---------------------------------------------------------------------------
# RESULTS block
# ---------------------------------------------------------------------------

def render_results(repo: Path) -> str:
    """Return the text that replaces the RESULTS block."""
    metrics_dir = _outputs(repo) / "metrics"
    preds_dir = _outputs(repo) / "predictions"

    lines: list[str] = []

    if not metrics_dir.exists():
        lines.append("Not yet run.")
        return "\n".join(lines)

    json_files = sorted(metrics_dir.glob("*.json"))
    if not json_files:
        lines.append("Not yet run.")
        return "\n".join(lines)

    lines.append(
        "| Model | Split | Accuracy | Macro-F1 | Macro-F1 95% CI |"
        " mel recall | bcc recall | akiec recall |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        model = data.get("model_name", jf.stem)
        split = data.get("split_mode", "—")
        metrics = data.get("metrics", {})

        acc = metrics.get("accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "—"

        mf1 = metrics.get("macro_f1")
        mf1_s = f"{mf1:.4f}" if mf1 is not None else "—"

        # Bootstrap CI for macro-F1
        mf1_ci = _get_ci(repo, jf, "macro_f1")

        # Per-class recall for malignant classes
        per_class = metrics.get("per_class", {})
        mal_recalls: list[str] = []
        for c in ("mel", "bcc", "akiec"):
            r = per_class.get(c, {}).get("recall")
            ci = _get_class_ci(repo, jf, c)
            if r is not None:
                mal_recalls.append(f"{r:.4f}{ci}")
            else:
                mal_recalls.append("—")

        lines.append(
            f"| {model} | {split} | {acc_s} | {mf1_s} | {mf1_ci} |"
            f" {mal_recalls[0]} | {mal_recalls[1]} | {mal_recalls[2]} |"
        )

    return "\n".join(lines)


def _get_ci(repo: Path, metrics_json: Path, metric: str) -> str:
    """Return '(lo, hi)' string from bootstrap CI file, or '—'."""
    # CI files are expected at outputs/metrics/<stem>_ci.json
    ci_path = metrics_json.parent / (metrics_json.stem + "_ci.json")
    if not ci_path.exists():
        return "—"
    try:
        ci = json.loads(ci_path.read_text(encoding="utf-8"))
        lo = ci.get(metric, {}).get("ci_lower")
        hi = ci.get(metric, {}).get("ci_upper")
        if lo is not None and hi is not None:
            return f"({lo:.4f}, {hi:.4f})"
    except Exception:
        pass
    return "—"


def _get_class_ci(repo: Path, metrics_json: Path, cls: str) -> str:
    """Return ' (lo, hi)' for per-class recall CI, or ''."""
    ci_path = metrics_json.parent / (metrics_json.stem + "_ci.json")
    if not ci_path.exists():
        return ""
    try:
        ci = json.loads(ci_path.read_text(encoding="utf-8"))
        key = f"{cls}_recall"
        lo = ci.get(key, {}).get("ci_lower")
        hi = ci.get(key, {}).get("ci_upper")
        if lo is not None and hi is not None:
            return f" ({lo:.4f}, {hi:.4f})"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# DATASET block
# ---------------------------------------------------------------------------

def render_dataset(repo: Path) -> str:
    """Return the text that replaces the DATASET block."""
    stats_path = _outputs(repo) / "dataset_stats.json"
    if not stats_path.exists():
        return "Not yet run."

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return "Could not read dataset_stats.json."

    lines: list[str] = []
    lines.append("| Class | Images | Unique lesions |")
    lines.append("|---|---|---|")

    per_class = stats.get("per_class", {})
    for cls in CLASSES:
        info = per_class.get(cls, {})
        n_img = info.get("images", "—")
        n_les = info.get("lesions", "—")
        lines.append(f"| {cls} | {n_img} | {n_les} |")

    imb = stats.get("imbalance", {})
    img_imb = imb.get("image_level", {})
    les_imb = imb.get("lesion_level", {})

    lines.append("")
    if img_imb:
        maj = img_imb.get("majority_class", "—")
        maj_n = img_imb.get("majority_count", "—")
        minn = img_imb.get("minority_class", "—")
        min_n = img_imb.get("minority_count", "—")
        ratio = img_imb.get("ratio", "—")
        ratio_s = f"{ratio:.1f}" if isinstance(ratio, float) else str(ratio)
        lines.append(
            f"Image-level imbalance ratio: {ratio_s} "
            f"({maj}: {maj_n} vs {minn}: {min_n})"
        )
    if les_imb:
        maj = les_imb.get("majority_class", "—")
        maj_n = les_imb.get("majority_count", "—")
        minn = les_imb.get("minority_class", "—")
        min_n = les_imb.get("minority_count", "—")
        ratio = les_imb.get("ratio", "—")
        ratio_s = f"{ratio:.1f}" if isinstance(ratio, float) else str(ratio)
        lines.append(
            f"Lesion-level imbalance ratio: {ratio_s} "
            f"({maj}: {maj_n} vs {minn}: {min_n})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STAGES block
# ---------------------------------------------------------------------------

# Expected outputs per notebook:
# 01 -> dataset_stats.json + both manifests (lesion + image)
# 02 -> baseline_cnn JSON (lesion mode)
# 03 -> resnet18 and efficientnet_b0 JSONs (lesion mode)
# 04 -> gradcam_index.csv
# 05 -> comparison figure (outputs/figures/)

def _notebook_exists(repo: Path, nb_prefix: str) -> bool:
    nb_dir = repo / "notebooks"
    return any(nb_dir.glob(f"{nb_prefix}*.ipynb"))


def _stage_status(repo: Path, nb_prefix: str, output_paths: list[Path]) -> str:
    """Return 'Complete', 'Code ready, not yet run', or 'Pending'."""
    nb_ok = _notebook_exists(repo, nb_prefix)
    outputs_ok = all(p.exists() and p.stat().st_size > 0 for p in output_paths)
    if nb_ok and outputs_ok:
        return "Complete"
    if nb_ok:
        return "Code ready, not yet run"
    return "Pending"


def render_stages(repo: Path) -> str:
    out = _outputs(repo)
    nb = repo / "notebooks"

    # Detect both manifests for 01.
    manifest_lesion = list((out / "splits").glob("manifest_lesion_*.csv"))
    manifest_image = list((out / "splits").glob("manifest_image_*.csv"))

    stages = [
        (
            "01",
            "01_data_and_splits",
            [
                out / "dataset_stats.json",
                *(manifest_lesion[:1]),
                *(manifest_image[:1]),
            ],
        ),
        (
            "02",
            "02_baseline_cnn",
            list((out / "metrics").glob("baseline_cnn_lesion_*.json")),
        ),
        (
            "03",
            "03_transfer_learning",
            list((out / "metrics").glob("resnet18_lesion_*.json"))
            + list((out / "metrics").glob("efficientnet_b0_lesion_*.json")),
        ),
        (
            "04",
            "04_gradcam",
            [out / "figures" / "gradcam_index.csv"],
        ),
        (
            "05",
            "05_comparison",
            list((out / "figures").glob("comparison*.png")),
        ),
    ]

    lines: list[str] = []
    lines.append("| Notebook | Status |")
    lines.append("|---|---|")
    for nb_num, nb_prefix, output_list in stages:
        status = _stage_status(repo, nb_prefix, output_list)
        lines.append(f"| {nb_num} | {status} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block replacement
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"(<!-- BEGIN (?P<name>RESULTS|DATASET|STAGES) -->)"
    r".*?"
    r"(<!-- END (?P=name) -->)",
    re.DOTALL,
)

_RENDERERS = {
    "RESULTS": render_results,
    "DATASET": render_dataset,
    "STAGES": render_stages,
}


def update_blocks(path: Path, repo: Path) -> bool:
    """Replace all three marker blocks in *path*.

    Returns True if the file was changed, False if it was already up-to-date.
    Exits with code 1 (after printing a message) if any marker is absent.
    """
    raw = path.read_bytes()

    # Detect line-ending style (preserve original).
    if b"\r\n" in raw:
        newline = "\r\n"
    else:
        newline = "\n"

    text = raw.decode("utf-8")

    # Check all three markers are present.
    missing = []
    for name in ("RESULTS", "DATASET", "STAGES"):
        if f"<!-- BEGIN {name} -->" not in text:
            missing.append(name)
    if missing:
        print(
            f"ERROR: README is missing markers for: {', '.join(missing)}\n"
            "Add <!-- BEGIN X --> ... <!-- END X --> blocks and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    def _replace(m: re.Match) -> str:
        name = m.group("name")
        renderer = _RENDERERS[name]
        content = renderer(repo)
        # Normalise to LF internally; we'll fix endings at the end.
        inner = f"\n{content}\n"
        return f"<!-- BEGIN {name} -->{inner}<!-- END {name} -->"

    new_text = _BLOCK_RE.sub(_replace, text)

    # Re-apply original line endings.
    if newline == "\r\n":
        new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")

    new_bytes = new_text.encode("utf-8")
    if new_bytes == raw:
        return False

    path.write_bytes(new_bytes)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(
        description="Fill RESULTS, DATASET and STAGES marker blocks in README.md."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Path to README.md (default: <repo_root>/README.md)",
    )
    args = parser.parse_args(argv)

    repo = _REPO_DIR
    readme = args.readme if args.readme else repo / "README.md"

    if not readme.exists():
        print(f"ERROR: README not found at {readme}", file=sys.stderr)
        sys.exit(1)

    changed = update_blocks(readme, repo)
    if changed:
        print(f"README.md updated ({readme})")
    else:
        print("README.md already up to date.")


if __name__ == "__main__":
    main()
