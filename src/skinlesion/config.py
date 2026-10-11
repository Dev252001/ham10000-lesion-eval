"""
config.py – training hyperparameters and configuration dataclass.

TrainConfig records every hyperparameter choice.  Values are set once here
(chosen a priori based on common practice for skin-lesion fine-tuning) and
are never modified based on test-set results.

The config fingerprint (SHA-256 of a canonical JSON snapshot) is used by
train.py to decide whether to retrain or reuse an existing checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict


# Learning rates chosen a priori:
#   - 1e-3  for models trained from scratch (baseline_cnn)
#   - 3e-4  for fine-tuned pretrained models (resnet18, efficientnet_b0)
# These are standard starting points; they are recorded here and not
# adjusted after examining test-set performance.
_DEFAULT_LR_MAP: Dict[str, float] = {
    "baseline_cnn": 1e-3,
    "resnet18": 3e-4,
    "efficientnet_b0": 3e-4,
}


@dataclass
class TrainConfig:
    """All hyperparameters for a single training run.

    Fields
    ------
    model_name    : one of baseline_cnn / resnet18 / efficientnet_b0
    split_mode    : lesion (recommended) or image (leaky comparison)
    seed          : random seed used for splitting and weight init
    image_size    : side length of square input (pixels)
    batch_size    : mini-batch size
    max_epochs    : hard ceiling on training epochs
    patience      : early-stopping patience (validation macro-F1)
    lr            : learning rate (set from _DEFAULT_LR_MAP if not overridden)
    weight_decay  : L2 regularisation coefficient
    selection_rule: human-readable string describing how best epoch is chosen
    manifest_sha256: SHA-256 of the split manifest; mandatory before training
    """

    model_name: str
    split_mode: str
    seed: int

    # Architecture / data
    image_size: int = 224
    batch_size: int = 32

    # Optimisation
    max_epochs: int = 20
    patience: int = 5
    lr: float = field(default=0.0)  # set in __post_init__
    weight_decay: float = 1e-4

    # Evaluation selection rule (documented, never changed post-hoc)
    selection_rule: str = (
        "best validation macro-F1; tie-break lower validation loss"
    )

    # Provenance – must be set before training starts
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        # Assign default LR if caller left it at 0.0
        if self.lr == 0.0:
            self.lr = _DEFAULT_LR_MAP.get(self.model_name, 1e-3)

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable SHA-256 of the canonical config JSON.

        The canonical JSON includes: model_name, split_mode, seed,
        manifest_sha256, and all hyperparameters.  It does NOT include
        mutable fields like selection_rule text (it is always the same)
        or future fields that should not invalidate existing checkpoints.

        The JSON is serialised with sorted keys and no extra whitespace
        so the hash is OS- and Python-version-independent.
        """
        if not self.manifest_sha256:
            raise ValueError(
                "manifest_sha256 must be set before computing a fingerprint. "
                "Call make_split -> split_manifest -> manifest_sha256 first."
            )
        canonical = {
            "model": self.model_name,
            "split_mode": self.split_mode,
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "hyperparameters": {
                "image_size": self.image_size,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
            "selection_rule": self.selection_rule,
        }
        text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)