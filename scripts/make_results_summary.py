"""
scripts/make_results_summary.py – write outputs/RESULTS_SUMMARY.md.

Reads only from outputs/metrics/*.json and outputs/predictions/*.csv.
Writes nothing and exits with code 1 if no metrics files exist.

The summary includes:
- Per-run: model, split, key metrics, provenance (git commit, manifest SHA).
- Split verification: manifest file present, SHA-256 matches JSON,
  zero lesion_id shared between train/val/test.
  Prints "same split for all included runs" only when verified.
- datetime.now(timezone.utc) timestamp.

Usage:
    python scripts/make_results_summary.py [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to sys.path so the script works without installation.
_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of the given file (canonical CSV text)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _verify_manifest(
    manifest_path: Path,
    expected_sha: str,
) -> tuple[bool, str]:
    """Return (ok, reason_if_not_ok)."""
    if not manifest_path.exists():
        return False, f"manifest file not found: {manifest_path.name}"

    # Verify SHA-256 matches.
    actual_sha = _sha256_file(manifest_path)
    if actual_sha != expected_sha:
        return False, (
            f"SHA-256 mismatch: JSON says {expected_sha[:12]}… "
            f"but file is {actual_sha[:12]}…"
        )

    # Verify zero lesion_id shared between splits.
    split_to_lesions: dict[str, set[str]] = {}
    try:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            # Skip comment lines (lines starting with #).
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            sp = row.get("split", "")
            lid = row.get("lesion_id", "")
            split_to_lesions.setdefault(sp, set()).add(lid)
    except Exception as exc:
        return False, f"Could not parse manifest: {exc}"

    splits = list(split_to_lesions.keys())
    shared: list[str] = []
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = split_to_lesions[s1] & split_to_lesions[s2]
            if overlap:
                shared.append(f"{s1}/{s2}: {len(overlap)} shared lesion(s)")

    if shared:
        return False, "lesion_id shared between splits: " + "; ".join(shared)

    return True, ""


def _git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def build_summary(repo: Path) -> str | None:
    """Return the summary markdown string, or None if no metrics exist."""
    metrics_dir = repo / "outputs" / "metrics"
    preds_dir = repo / "outputs" / "predictions"
    splits_dir = repo / "outputs" / "splits"

    json_files = sorted(metrics_dir.glob("*.json")) if metrics_dir.exists() else []
    if not json_files:
        return None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_rev = _git_commit(repo)

    lines: list[str] = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append(f"Generated: {now}  ")
    lines.append(f"Git commit: {git_rev}")
    lines.append("")

    # Collect manifest SHAs for cross-run split verification.
    manifest_shas: set[str] = set()
    verify_failures: list[str] = []
    run_blocks: list[str] = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            run_blocks.append(f"### {jf.stem}\n\nCould not read: {exc}\n")
            continue

        model = data.get("model_name", "—")
        split_mode = data.get("split_mode", "—")
        seed = data.get("seed", "—")
        msha = data.get("manifest_sha256", "not recorded")
        metrics = data.get("metrics", {})
        train_time = data.get("train_time_s")
        git_c = data.get("git_commit", "unknown")
        created = data.get("created_utc", "unknown")

        acc = metrics.get("accuracy")
        mf1 = metrics.get("macro_f1")
        per_class = metrics.get("per_class", {})

        block: list[str] = []
        block.append(f"### {jf.stem}")
        block.append("")
        block.append(f"- Model: {model}")
        block.append(f"- Split mode: {split_mode}, seed: {seed}")
        block.append(f"- Accuracy: {acc:.4f}" if acc is not None else "- Accuracy: —")
        block.append(f"- Macro-F1: {mf1:.4f}" if mf1 is not None else "- Macro-F1: —")

        for cls in ("mel", "bcc", "akiec"):
            r = per_class.get(cls, {}).get("recall")
            block.append(
                f"- {cls} recall: {r:.4f}" if r is not None else f"- {cls} recall: —"
            )

        if train_time is not None:
            block.append(f"- Training time: {train_time:.1f} s")
        else:
            block.append("- Training time: —")

        block.append(f"- Git commit: {git_c}")
        block.append(f"- Created: {created}")
        block.append(f"- Manifest SHA-256 prefix: {msha[:12] if msha and msha != 'not recorded' else msha}")

        # Verify manifest.
        if msha and msha != "not recorded":
            # Find the manifest file.
            manifest_files = list(splits_dir.glob(
                f"manifest_{split_mode}_seed{seed}.csv"
            )) if splits_dir.exists() else []

            if manifest_files:
                ok, reason = _verify_manifest(manifest_files[0], msha)
                if ok:
                    block.append("- Manifest verification: passed")
                    manifest_shas.add(msha)
                else:
                    block.append(f"- Manifest verification: FAILED ({reason})")
                    verify_failures.append(f"{jf.stem}: {reason}")
            else:
                block.append("- Manifest verification: manifest file not found")
                verify_failures.append(f"{jf.stem}: manifest file not found")
        else:
            block.append("- Manifest verification: not recorded")
            verify_failures.append(f"{jf.stem}: manifest SHA not recorded")

        run_blocks.append("\n".join(block))

    lines.append("## Runs")
    lines.append("")
    lines.extend(run_blocks)
    lines.append("")

    # Cross-run split consistency.
    lines.append("## Verification")
    lines.append("")
    if not verify_failures and len(manifest_shas) == 1:
        lines.append("same split for all included runs")
    elif not verify_failures and len(manifest_shas) > 1:
        lines.append(
            "not verified: runs used different manifest SHA-256 values "
            "(different seeds or split modes)"
        )
    else:
        for f in verify_failures:
            lines.append(f"not verified: {f}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(
        description="Write outputs/RESULTS_SUMMARY.md from committed metrics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <repo>/outputs/RESULTS_SUMMARY.md)",
    )
    args = parser.parse_args(argv)

    repo = _REPO_DIR
    out_path = args.output if args.output else repo / "outputs" / "RESULTS_SUMMARY.md"

    summary = build_summary(repo)
    if summary is None:
        print(
            "No metrics files found in outputs/metrics/. "
            "Run notebooks 02-03 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
