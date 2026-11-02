"""
scripts/compute_paired_bootstrap.py – compute paired cluster-bootstrap
differences between committed prediction CSVs and save to
outputs/metrics/paired_bootstrap.json.

Comparisons computed:
  - efficientnet_b0 vs resnet18  (lesion split, seed 42)
  - resnet18 vs baseline_cnn     (lesion split, seed 42)

For the lesion-vs-image-level comparison, both prediction CSVs for the
same model must exist (e.g. efficientnet_b0_lesion_seed42.csv and
efficientnet_b0_image_seed42.csv).  Those files are produced by
re-running notebooks 02-03 with SPLIT_MODE="image".  Missing comparisons
are recorded as null rather than causing a failure.

Usage:
    python scripts/compute_paired_bootstrap.py [--n-boot N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR / "src"))


def _load_preds(preds_dir: Path, model: str, split: str, seed: int) -> "pd.DataFrame | None":
    try:
        import pandas as pd
    except ImportError:
        return None
    path = preds_dir / f"{model}_{split}_seed{seed}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def run(outputs_dir: Path, n_boot: int, seed: int) -> dict:
    from skinlesion.evaluate import paired_cluster_bootstrap

    preds_dir = outputs_dir / "predictions"
    results: dict = {"n_boot": n_boot, "seed": seed, "interval_method": "percentile",
                     "resampling_unit": "lesion_id", "comparisons": {}}

    # EfficientNet-B0 vs ResNet-18 (lesion split)
    en = _load_preds(preds_dir, "efficientnet_b0", "lesion", 42)
    rn = _load_preds(preds_dir, "resnet18", "lesion", 42)
    if en is not None and rn is not None:
        diff, lo, hi = paired_cluster_bootstrap(en, rn, metric="macro_f1",
                                                n_boot=n_boot, seed=seed)
        results["comparisons"]["efficientnet_b0_vs_resnet18_lesion"] = {
            "metric": "macro_f1",
            "split": "lesion",
            "diff": round(diff, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "note": "EfficientNet-B0 macro-F1 minus ResNet-18 macro-F1",
        }
    else:
        results["comparisons"]["efficientnet_b0_vs_resnet18_lesion"] = None
        print("Warning: missing predictions for efficientnet_b0 or resnet18 (lesion)",
              file=sys.stderr)

    # ResNet-18 vs Baseline CNN (lesion split)
    cnn = _load_preds(preds_dir, "baseline_cnn", "lesion", 42)
    if rn is not None and cnn is not None:
        diff, lo, hi = paired_cluster_bootstrap(rn, cnn, metric="macro_f1",
                                                n_boot=n_boot, seed=seed)
        results["comparisons"]["resnet18_vs_baseline_cnn_lesion"] = {
            "metric": "macro_f1",
            "split": "lesion",
            "diff": round(diff, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "note": "ResNet-18 macro-F1 minus Baseline CNN macro-F1",
        }
    else:
        results["comparisons"]["resnet18_vs_baseline_cnn_lesion"] = None
        print("Warning: missing predictions for resnet18 or baseline_cnn (lesion)",
              file=sys.stderr)

    # Lesion-vs-image-level for EfficientNet-B0
    en_img = _load_preds(preds_dir, "efficientnet_b0", "image", 42)
    if en is not None and en_img is not None:
        diff, lo, hi = paired_cluster_bootstrap(en_img, en, metric="macro_f1",
                                                n_boot=n_boot, seed=seed)
        results["comparisons"]["efficientnet_b0_image_vs_lesion"] = {
            "metric": "macro_f1",
            "split_a": "image",
            "split_b": "lesion",
            "diff": round(diff, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "note": "image-split macro-F1 minus lesion-split macro-F1 (leakage gap)",
        }
    else:
        results["comparisons"]["efficientnet_b0_image_vs_lesion"] = None
        if en_img is None:
            print("Info: efficientnet_b0_image_seed42.csv not found; "
                  "run notebooks 02-03 with SPLIT_MODE=image to compute leakage gap.",
                  file=sys.stderr)

    return results


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(
        description="Compute paired cluster-bootstrap differences between models."
    )
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap replicates (default: 1000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="NumPy random seed (default: 0)")
    parser.add_argument("--outputs", type=Path, default=None,
                        help="outputs/ directory (default: <repo>/outputs)")
    args = parser.parse_args(argv)

    outputs_dir = args.outputs if args.outputs else _REPO_DIR / "outputs"
    results = run(outputs_dir, n_boot=args.n_boot, seed=args.seed)

    out_path = outputs_dir / "metrics" / "paired_bootstrap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
