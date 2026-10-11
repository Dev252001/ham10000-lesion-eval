"""
gradcam.py – Grad-CAM visualisations for HAM10000 models.

Selection rule (no cherry-picking)
-----------------------------------
select_examples(pred_df, cls, n, seed, correct) draws up to *n* rows from
the test-prediction DataFrame for class *cls* using a fixed random seed.

  correct=True  : rows where y_true == y_pred == cls_index
                  (the model classified a true *cls* image correctly)
  correct=False : rows where y_true == cls_index  and  y_pred != cls_index
                  (the model made a mistake on a true *cls* image)

Rows are shuffled with the given seed; the first *n* are taken.  If fewer
than *n* examples exist a warning is issued and all available rows are
returned.  The seed is the only free parameter: which rows are "correct"
or "incorrect" is fully determined by the saved predictions file.

Public API
----------
select_examples       : deterministic correct/incorrect example selection
make_gradcam_figure   : save PNGs + append to gradcam_index.csv
"""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from skinlesion.data import CLASSES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target-layer helpers
# ---------------------------------------------------------------------------

def _get_target_layer(model, model_name: str):
    """Return the convolutional layer used as the Grad-CAM target.

    ResNet-18       : model.layer4[-1]   (last BasicBlock of stage 4)
    EfficientNet-B0 : model.features[-1] (last MBConv/Conv block)
    BaselineCNN     : model.blocks[-1]   (last Conv-BN-ReLU-MaxPool block)
    """
    name = model_name.lower()
    if name == "resnet18":
        return model.layer4[-1]
    elif name == "efficientnet_b0":
        return model.features[-1]
    elif name == "baseline_cnn":
        return model.blocks[-1]
    else:
        raise ValueError(
            f"Unknown model name for Grad-CAM target layer: {model_name!r}"
        )


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def select_examples(
    pred_df: pd.DataFrame,
    cls: str,
    n: int,
    seed: int,
    correct: bool,
) -> pd.DataFrame:
    """Return up to *n* test-prediction rows for class *cls*.

    Parameters
    ----------
    pred_df:
        Predictions DataFrame (columns: image_id, y_true, y_pred, …).
    cls:
        Class name, must be one of CLASSES.
    n:
        Number of examples requested (up to; see warning).
    seed:
        RNG seed; fully determines which rows are returned.
    correct:
        True  → y_true == y_pred == cls_index
        False → y_true == cls_index and y_pred != cls_index

    Returns
    -------
    DataFrame slice, up to n rows, reset index.
    """
    if cls not in CLASSES:
        raise ValueError(f"Unknown class {cls!r}; expected one of {CLASSES}")
    cls_idx = CLASSES.index(cls)

    mask_true = pred_df["y_true"] == cls_idx
    if correct:
        mask = mask_true & (pred_df["y_pred"] == cls_idx)
        label = "correct"
    else:
        mask = mask_true & (pred_df["y_pred"] != cls_idx)
        label = "incorrect"

    subset = (
        pred_df[mask]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    if len(subset) < n:
        warnings.warn(
            f"Only {len(subset)} {label} example(s) available for class "
            f"'{cls}' (requested {n}); returning all of them."
        )
    return subset.iloc[:n].copy()


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_gradcam_figure(
    model,
    model_name: str,
    pred_df: pd.DataFrame,
    cls: str,
    n: int,
    seed: int,
    image_paths: Dict[str, Path],
    figures_dir: Path,
    device=None,
    image_size: int = 224,
) -> None:
    """Generate Grad-CAM overlay figures for correct and incorrect examples.

    For each example (correct and incorrect):
    - Saves a PNG side-by-side (original | Grad-CAM overlay) to *figures_dir*.
    - Appends one row to ``outputs/figures/gradcam_index.csv``.

    Caption format (neutral):
        model=<name>  class=<cls>  result=correct|incorrect
        true=<cls>  pred=<cls>  confidence=<p:.3f>

    Requires ``pytorch_grad_cam``; raises ImportError if unavailable.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pytorch_grad_cam is required for make_gradcam_figure. "
            "Install with: pip install grad-cam"
        ) from exc

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from skinlesion.transforms import eval_transform

    if device is None:
        device = torch.device("cpu")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    index_path = figures_dir / "gradcam_index.csv"
    index_exists = index_path.exists()

    transform = eval_transform(image_size)
    target_layer = _get_target_layer(model, model_name)
    cam_obj = GradCAM(model=model, target_layers=[target_layer])

    model.eval()
    model.to(device)

    cls_idx = CLASSES.index(cls)

    with open(index_path, "a", newline="\n", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not index_exists:
            writer.writerow(
                ["filename", "model", "class", "result",
                 "y_true", "y_pred", "confidence"]
            )

        for result_label, is_correct in [("correct", True), ("incorrect", False)]:
            examples = select_examples(pred_df, cls, n, seed, correct=is_correct)
            for _, row in examples.iterrows():
                stem = str(row["image_id"])
                if stem not in image_paths:
                    logger.warning("Image %s not found; skipping.", stem)
                    continue

                pil_img = Image.open(image_paths[stem]).convert("RGB")
                pil_resized = pil_img.resize((image_size, image_size))
                rgb_np = np.array(pil_resized, dtype=np.float32) / 255.0

                inp = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                grayscale_cam = cam_obj(
                    input_tensor=inp,
                    targets=[ClassifierOutputTarget(cls_idx)],
                )
                overlay = show_cam_on_image(rgb_np, grayscale_cam[0], use_rgb=True)

                true_cls = CLASSES[int(row["y_true"])]
                pred_cls = CLASSES[int(row["y_pred"])]
                conf = float(probs[int(row["y_pred"])])

                fname = f"gradcam_{model_name}_{cls}_{result_label}_{stem}.png"
                out_path = figures_dir / fname

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(pil_resized)
                axes[0].set_title("Original")
                axes[0].axis("off")
                axes[1].imshow(overlay)
                axes[1].set_title("Grad-CAM")
                axes[1].axis("off")
                caption = (
                    f"model={model_name}  class={cls}  "
                    f"result={result_label}\n"
                    f"true={true_cls}  pred={pred_cls}  "
                    f"confidence={conf:.3f}"
                )
                fig.suptitle(caption, fontsize=9)
                fig.tight_layout()
                fig.savefig(out_path, dpi=100, bbox_inches="tight")
                plt.close(fig)

                writer.writerow([
                    fname, model_name, cls, result_label,
                    true_cls, pred_cls, f"{conf:.4f}",
                ])
