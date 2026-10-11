# HAM10000 Skin-Lesion Classification -- Leakage-Aware Evaluation

This repository implements a 7-class skin-lesion classifier on the
[HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161), with an
explicit focus on evaluation rigour: lesion-grouped splits, recorded
provenance, and a leakage comparison that quantifies the effect of a
naive image-level split.

---

> **Research use only.**
> This code and any results it produces are for academic and educational
> purposes. They are not validated for clinical use and must not be used
> to inform medical decisions.

---

## Results

<!-- BEGIN RESULTS -->
Not yet run.
<!-- END RESULTS -->

---

## Evaluation Protocol

All reported results will use a lesion-grouped, class-stratified
70/15/15 split (split_mode="lesion", seed 42; the manifest and its
SHA-256 are recorded with every run). Checkpoint selection and early
stopping use the validation split only. The test split is used only to
evaluate each model's final selected checkpoint. Class weights are
computed from the training split only (w_c = N / (C -- n_c)). All
models share the data pipeline, loss, epoch budget and selection rule;
the learning rate differs between the from-scratch and the fine-tuned
models. Primary metric: macro-F1; secondary: recall on mel, bcc and
akiec. Confidence intervals come from a bootstrap that resamples
lesions. An image-level split is run only to quantify leakage.

---

## Dataset

<!-- BEGIN DATASET -->
Not yet run.
<!-- END DATASET -->

**Citation:** Tschandl, P., Rosendahl, C. & Kittler, H.
The HAM10000 dataset, a large collection of multi-source dermatoscopic
images of common pigmented skin lesions.
*Scientific Data* **5**, 180161 (2018).
<https://doi.org/10.1038/sdata.2018.161>

Dataset available via the
[ISIC Archive](https://www.isic-archive.com) and the
[Kaggle mirror](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
(CC BY-NC 4.0).

---

## Stage Status

<!-- BEGIN STAGES -->
| Notebook | Status |
|---|---|
| 01 | Code ready, not yet run |
| 02 | Complete |
| 03 | Complete |
| 04 | Code ready, not yet run |
| 05 | Complete |
<!-- END STAGES -->

---

## How to Run

### Tests (local, CPU only, no dataset needed)

```bash
pytest tests/ -W ignore
```

### Colab notebooks

Run notebooks in order on a free Colab T4 GPU:

| Notebook | Colab badge |
|---|---|
| 01 Data & Splits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/ham10000-lesion-eval/blob/main/notebooks/01_data_and_splits.ipynb) |
| 02 Baseline CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/ham10000-lesion-eval/blob/main/notebooks/02_baseline_cnn.ipynb) |
| 03 Transfer Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/ham10000-lesion-eval/blob/main/notebooks/03_transfer_learning.ipynb) |
| 04 Grad-CAM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/ham10000-lesion-eval/blob/main/notebooks/04_gradcam.ipynb) |
| 05 Comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/ham10000-lesion-eval/blob/main/notebooks/05_comparison.ipynb) |

**Notes:**
- You will be prompted to upload `kaggle.json` (your Kaggle API key).
  The file is never written into the repository.
- Run notebooks in order (01 -> 02 -> 03 -> 04 -> 05).
- To test a branch before merging, change `REPO_BRANCH` in the
  parameter cell of each notebook.
- After each notebook, download the zip, unzip at the root of your
  local clone, and commit.
- Then run:

```bash
python scripts/update_readme.py
python scripts/make_results_summary.py
```

---

## Project Structure

```
src/skinlesion/
    __init__.py
    data.py          load_metadata, HAMDataset, dataset_stats
    splits.py        make_split (lesion/image), manifest, SHA-256, audit
    transforms.py    train/eval pipelines, ImageNet constants
    models.py        BaselineCNN, build_model, count_params, compute_class_weights
    config.py        TrainConfig dataclass, fingerprint
    train.py         train_model, load_or_train, seed_everything
    evaluate.py      predict, final_test_evaluation, cluster_bootstrap_ci
    gradcam.py       select_examples, make_gradcam_figure
    export.py        make_outputs_zip

tests/               133+ tests; no network, no dataset, synthetic fixtures
notebooks/           01-05 Colab notebooks
scripts/
    update_readme.py          fill README marker blocks from outputs/
    make_results_summary.py   write outputs/RESULTS_SUMMARY.md
docs/
    MANUAL_STEPS.md     checklist for running the project end-to-end
    CODE_TOUR.md        one-liner per module + design decisions
    INTERVIEW_QUESTIONS.md  questions an admissions committee might ask
outputs/
    metrics/            JSON metrics files (one per run)
    predictions/        CSV prediction files (one per run)
    figures/            PNG figures
    splits/             manifest CSVs and audit JSONs
```

---

## Grad-CAM Note

Grad-CAM heatmaps highlight image regions with a high class-specific
gradient signal at the chosen layer. They are a qualitative
visualisation and cannot establish what the model actually relies on.

---

## Limitations

<!-- Q1: HAM10000's metadata has no patient ID, so lesion_id is the best grouping available. What leakage could remain between different lesions of the same patient? -->

<!-- Q2: HAM10000 was collected at two sites (see the Methods section of the HAM10000 paper). What would you expect to happen to these metrics on images from a different clinic, device or patient population? -->

<!-- Q3: The rarest classes have few lesions in val and test (see the split audit). How does that affect how much you can trust their recall numbers and confidence intervals? -->

<!-- Q4: Class-weighted cross-entropy was chosen over focal loss or oversampling. What are the trade-offs, and what would you try next if recall on the malignant classes stayed low? -->

<!-- Q5: Test images from the same lesion are correlated. What does that mean for the metrics, and what does the lesion-level bootstrap address or leave open? -->

---

## What I Learned

<!-- Q1: What surprised you most about the results? -->

<!-- Q2: What would you do differently if you started again? -->

<!-- Q3: What is the one thing about the evaluation methodology you want an admissions committee to know? -->

---

## Acknowledgements

- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset.
  *Scientific Data* **5**, 180161 (2018).
  <https://doi.org/10.1038/sdata.2018.161>
- Kaggle mirror:
  [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ISIC Archive](https://www.isic-archive.com)
- [torchvision](https://github.com/pytorch/vision)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

<!-- EDIT: state truthfully which parts of this project had AI assistance and which are my own work -->

---

## License

Code: MIT. See [LICENSE](LICENSE).

Dataset: CC BY-NC 4.0.
See <https://creativecommons.org/licenses/by-nc/4.0/>.

---

## Disclaimer

This repository is for research and educational use only. Results
reported here have not been validated for clinical use and must not
be used to inform medical decisions.
