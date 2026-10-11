# Code Tour

A factual one-liner per module and the key design decisions.

---

## src/skinlesion/

### `data.py`
Loads `HAM10000_metadata.csv`, discovers images by recursively globbing
`*.jpg` (case-insensitive), and maps stems to paths.  Raises clearly on
missing images or on a lesion with two different `dx` values.
`dataset_stats` computes imbalance ratios without any hard-coded class
counts " all numbers are derived from the DataFrame at runtime.

### `splits.py`
`make_split` implements two modes:

- **lesion** (default): collapse to one row per `lesion_id`, stratify on
  `dx` at the lesion level (two-step: train vs rest, then val vs test),
  then map back to images.  This is the key design decision: by splitting
  on lesions, the same lesion never appears in both train and test.
- **image**: plain stratified split on image rows, kept only to quantify
  how much an image-level split inflates metrics.

`manifest_sha256` hashes a canonical (sorted, LF-terminated) CSV of the
split so the hash is identical on every OS and can never be null.

### `transforms.py`
Fixed ImageNet mean/std " not derived from the dataset.  Using ImageNet
statistics is appropriate because the fine-tuned models were pre-trained
on ImageNet-normalised images.

### `models.py`
`BaselineCNN`: four Conv-BN-ReLU-MaxPool blocks (32->64->128->256 channels),
global average pooling, Dropout 0.5, Linear(256, 7).  No pre-trained
weights needed.

`build_model`: wraps torchvision for ResNet-18 and EfficientNet-B0 with
a 7-class head replacing the original classifier.

`compute_class_weights`: `N / (C * n_c)` computed from the training split
only " never from val or test.

### `config.py`
`TrainConfig` dataclass stores all hyperparameters.  Learning rate is
split by model family (1e-3 for from-scratch, 3e-4 for fine-tuned) to
reflect the different gradient scales.  `fingerprint()` hashes a
canonical JSON of `{model, split_mode, seed, manifest_sha256,
hyperparameters, selection rule}` so checkpoints from different
configurations cannot be silently reused.

### `train.py`
`train_model` has no test-loader parameter by design " keeping test data
out of the training loop is enforced at the API level, not by convention.
Early stopping and model selection use val macro-F1 (tie-break: lower val
loss).  `load_or_train` re-uses a checkpoint only when its fingerprint
matches the current config exactly; it prints why it retrains otherwise.
`seed_everything` seeds Python, NumPy, PyTorch and CUDA, but bitwise
determinism is not claimed.

### `evaluate.py`
`final_test_evaluation` is the **only** function that touches the test
split.  It writes both a predictions CSV and a metrics JSON, and raises
if `manifest_sha256` is missing " preventing provenance-free evaluations.
`cluster_bootstrap_ci` resamples at the lesion level, not the image level,
so correlated test images from the same lesion are kept together.

### `gradcam.py`
`select_examples` draws examples by a fixed seed with a documented
no-cherry-picking rule: correct/incorrect is determined entirely by the
saved predictions file.  The `pytorch_grad_cam` dependency is imported
lazily so the rest of the module works without it.

### `export.py`
Zips only the tracked output directories (metrics, predictions, figures,
splits, dataset_stats.json).  `outputs/models/` is excluded because
checkpoints are large and re-creatable.  Arc-names start with `outputs/`
so unzipping at the repo root places files correctly.

---

## scripts/

### `update_readme.py`
Fills three marker blocks (`RESULTS`, `DATASET`, `STAGES`) in README.md
by reading outputs/ JSON files.  Writes """ or "Not yet run" for any
missing run.  Preserves the file's original line endings (LF or CRLF).
Idempotent: a second run with the same data produces no diff.

### `make_results_summary.py`
Writes `outputs/RESULTS_SUMMARY.md` from metrics JSONs and predictions
CSVs.  For each run it verifies the manifest file exists, its SHA-256
matches the JSON, and no `lesion_id` is shared between splits.
Exits 1 and writes nothing if no metrics exist.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Lesion-level split | Prevents the same lesion appearing in train and test, which inflates metrics non-trivially on HAM10000. |
| SHA-256 manifest hash | Makes every evaluation traceable; the hash is computed from the in-memory split so it is never null. |
| Fingerprint-locked checkpoints | A checkpoint trained on a different split or seed cannot be silently reused. |
| `train_model` has no test-loader parameter | API-level enforcement: the test split cannot be passed to training code. |
| `final_test_evaluation` is the only test-split consumer | One entry point makes it easy to audit what touches test data. |
| `train_time_s = null` when unknown | Never a placeholder of 0.0, which would be a false claim. |
| Cluster bootstrap CI | Resampling lesions (not images) respects the correlation structure of the test set. |
| `pytest.importorskip` for optional deps | Tests run without pytorch_grad_cam; Grad-CAM tests skip cleanly. |
