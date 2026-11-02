"""
scripts/make_leakage_figure.py – bar chart comparing macro-F1 under
lesion-level vs image-level split for each model.

Reads from outputs/metrics/*.json and outputs/metrics/*_ci.json.
Writes to outputs/figures/leakage_comparison.png.

For each model, it plots the lesion-split macro-F1 with 95% CI error
bars.  If image-split metrics exist (produced by re-running notebooks
02-03 with SPLIT_MODE="image"), those are shown as a second bar.
Missing bars are omitted with a note to stderr rather than failing.

Usage:
    python scripts/make_leakage_figure.py [--outputs PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR / "src"))

_MODEL_ORDER = ["baseline_cnn", "resnet18", "efficientnet_b0"]
_DISPLAY_NAMES = {
    "baseline_cnn": "Baseline CNN",
    "resnet18": "ResNet-18",
    "efficientnet_b0": "EfficientNet-B0",
}


def _load_run(metrics_dir: Path, model: str, split: str, seed: int = 42):
    """Return (macro_f1, ci_lower, ci_upper) or None if not found."""
    stem = f"{model}_{split}_seed{seed}"
    mf = metrics_dir / f"{stem}.json"
    cf = metrics_dir / f"{stem}_ci.json"
    if not mf.exists():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        mf1 = data.get("metrics", {}).get("macro_f1")
        if mf1 is None:
            return None
        ci_lo, ci_hi = mf1, mf1  # fallback: no CI
        if cf.exists():
            ci = json.loads(cf.read_text(encoding="utf-8"))
            ci_lo = ci.get("macro_f1", {}).get("ci_lower", mf1)
            ci_hi = ci.get("macro_f1", {}).get("ci_upper", mf1)
        return float(mf1), float(ci_lo), float(ci_hi)
    except Exception as exc:
        print(f"Warning: could not read {stem}: {exc}", file=sys.stderr)
        return None


def make_figure(outputs_dir: Path) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: matplotlib is required: {exc}", file=sys.stderr)
        sys.exit(1)

    metrics_dir = outputs_dir / "metrics"
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = _MODEL_ORDER
    n = len(models)

    lesion_data = []
    image_data = []
    for m in models:
        lesion_data.append(_load_run(metrics_dir, m, "lesion"))
        image_data.append(_load_run(metrics_dir, m, "image"))

    has_image = any(v is not None for v in image_data)

    if not any(v is not None for v in lesion_data):
        print(
            "No lesion-split metrics found in outputs/metrics/. "
            "Run notebooks 02-03 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    x = np.arange(n)
    width = 0.35 if has_image else 0.55

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#f7f8fa")
    fig.patch.set_facecolor("white")

    # Lesion bars
    vals_l = [v[0] if v else 0.0 for v in lesion_data]
    errs_l = [
        ([v[0] - v[1]], [v[2] - v[0]]) if v else ([0], [0])
        for v in lesion_data
    ]
    err_l_lo = [e[0][0] for e in errs_l]
    err_l_hi = [e[1][0] for e in errs_l]

    offset = (-width / 2 if has_image else 0)
    bars_l = ax.bar(
        x + offset,
        vals_l,
        width,
        label="Lesion split (primary)",
        color="#3b82d4",
        alpha=0.85,
        zorder=3,
    )
    ax.errorbar(
        x + offset,
        vals_l,
        yerr=[err_l_lo, err_l_hi],
        fmt="none",
        ecolor="#1e3a5f",
        elinewidth=1.5,
        capsize=4,
        zorder=4,
    )

    # Image bars (if available)
    if has_image:
        vals_i = [v[0] if v else 0.0 for v in image_data]
        errs_i = [
            ([v[0] - v[1]], [v[2] - v[0]]) if v else ([0], [0])
            for v in image_data
        ]
        err_i_lo = [e[0][0] for e in errs_i]
        err_i_hi = [e[1][0] for e in errs_i]

        ax.bar(
            x + width / 2,
            vals_i,
            width,
            label="Image split (leakage reference)",
            color="#f59e0b",
            alpha=0.85,
            zorder=3,
        )
        ax.errorbar(
            x + width / 2,
            vals_i,
            yerr=[err_i_lo, err_i_hi],
            fmt="none",
            ecolor="#92400e",
            elinewidth=1.5,
            capsize=4,
            zorder=4,
        )
    else:
        ax.text(
            0.98, 0.04,
            "Image-split bars require running notebooks\n"
            "02-03 with SPLIT_MODE=\"image\".",
            ha="right", va="bottom",
            transform=ax.transAxes,
            fontsize=7,
            color="#57606a",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_DISPLAY_NAMES[m] for m in models], fontsize=10)
    ax.set_ylabel("Macro-F1 (test set)", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(
        "Macro-F1 by model and split mode (95% cluster-bootstrap CIs)",
        fontsize=10,
        pad=10,
    )
    ax.legend(fontsize=8, framealpha=0.9)

    out_path = figures_dir / "leakage_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(
        description="Write outputs/figures/leakage_comparison.png."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=None,
        help="outputs/ directory (default: <repo>/outputs)",
    )
    args = parser.parse_args(argv)

    outputs_dir = args.outputs if args.outputs else _REPO_DIR / "outputs"
    out = make_figure(outputs_dir)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
