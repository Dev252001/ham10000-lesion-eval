# Manual Steps Checklist

Work through these in order after cloning the repository locally.

---

## 1. Create the GitHub repository

1. Go to <https://github.com/new>.
2. Create a public repository named **ham10000-lesion-eval**.
   If you choose a different name, replace it in:
   - `REPO_URL` in the parameter cell of each of the five notebooks
   - The Colab badge URLs in `README.md`
3. Add the remote and push:

```bash
git remote add origin https://github.com/Dev252001/ham10000-lesion-eval.git
git push -u origin main
```

---

## 2. Run notebooks on Colab (T4)

Open each notebook from the Colab badges in README.md and run all
cells in order.  Save checkpoints to Drive between sessions.

Notebook order:

1. `01_data_and_splits.ipynb` — downloads data, saves `dataset_stats.json`,
   manifests and audits for both split modes.
2. `02_baseline_cnn.ipynb` — trains BaselineCNN, saves metrics + predictions.
3. `03_transfer_learning.ipynb` — trains ResNet-18 and EfficientNet-B0,
   saves metrics + predictions.
4. `04_gradcam.ipynb` — loads transfer checkpoints, saves Grad-CAM figures.
5. `05_comparison.ipynb` — reads JSON/predictions only (no GPU needed),
   saves comparison bar chart.

You will be prompted to upload `kaggle.json` (your Kaggle API key).  The
file is never written into the repository.

For the leakage comparison, re-run notebooks 02 and 03 with
`SPLIT_MODE = "image"` in their parameter cell.

To test a branch before merging, change `REPO_BRANCH` in the parameter
cell of each notebook.

---

## 3. Download outputs and commit

After each notebook (or after a group of notebooks):

1. The last cell downloads a zip.
2. Unzip at the root of your local clone:
   ```bash
   unzip outputs_01.zip   # (or 02, 03, 04, 05)
   ```
3. Commit:
   ```bash
   git add outputs/
   git commit -m "add outputs from notebook 01"
   ```

---

## 4. Update README and summary

```bash
python scripts/update_readme.py
python scripts/make_results_summary.py
python scripts/make_leakage_figure.py
python scripts/compute_paired_bootstrap.py   # re-run after committing predictions
git add README.md outputs/
git commit -m "update README and results summary"
```

---

## 5. Stage Status

| Notebook | Status |
|---|---|
| 01 | Complete |
| 02 | Complete |
| 03 | Complete |
| 04 | Complete |
| 05 | Complete |

---

## 6. Final editorial steps

- Fill in the `<!-- AUTHOR TO CONFIRM -->` block in `README.md`: state
  truthfully which parts had AI assistance and which are your own work.
- If you add more seeds or models, re-run the scripts in step 4.

---

## 7. GitHub About text and topics

Set the repository description (160 characters or fewer, no results):

> HAM10000 skin-lesion classification: lesion-grouped leakage-aware evaluation,
> cluster bootstrap CIs, Grad-CAM. Three models on free Colab T4.

Suggested topics: `deep-learning`, `skin-lesion`, `ham10000`,
`pytorch`, `medical-imaging`, `grad-cam`, `reproducibility`.

Pin the repository on your profile.

---

## 8. Tag v1.0

After merging `polish/review-ready` into `main`:

```bash
git checkout main
git merge --no-ff polish/review-ready -m "merge polish/review-ready"
git tag -a v1.0 -m "v1.0: review-ready snapshot"
# git push origin main v1.0   # when ready to publish
```

---

## 9. Old repository

Decide what to do with any previous repository:
- Archive it on GitHub (Settings → Archive this repository).
- Add a README note pointing to this repository.
- Delete it if it was never pushed.
