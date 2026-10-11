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

1. `01_data_and_splits.ipynb` " downloads data, saves `dataset_stats.json`,
   manifests and audits for both split modes.
2. `02_baseline_cnn.ipynb` " trains BaselineCNN, saves metrics + predictions.
3. `03_transfer_learning.ipynb` " trains ResNet-18 and EfficientNet-B0,
   saves metrics + predictions.
4. `04_gradcam.ipynb` " loads transfer checkpoints, saves Grad-CAM figures.
5. `05_comparison.ipynb` " reads JSON/predictions only (no GPU needed),
   saves comparison bar chart.

For the leakage comparison, re-run notebooks 02 and 03 with
`SPLIT_MODE = "image"` in their parameter cell.

For additional seeds, change `SEED` in the parameter cell.

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
git add README.md outputs/RESULTS_SUMMARY.md
git commit -m "update README and results summary"
```

---

## 5. Final editorial steps

- In `README.md`, change "will use" to the present tense in the
  Evaluation Protocol section once results exist.
- Fill in the **Limitations** section (the HTML comment questions are
  prompts; write your own answers in prose below each comment or
  replace the comments).
- Fill in the **What I Learned** section similarly.
- Edit the AI assistance line:
  `<!-- EDIT: state truthfully which parts of this project had AI assistance and which are my own work -->`

---

## 6. GitHub About text and topics

Set the repository description (350 characters or fewer, no results):

> HAM10000 skin-lesion classification with lesion-grouped, leakage-aware evaluation. Lesion-level stratified splits, manifest SHA-256 provenance, cluster bootstrap CIs, and a leakage comparison. Three models: baseline CNN, ResNet-18, EfficientNet-B0. Runs on free Colab T4.

Suggested topics: `deep-learning`, `skin-lesion`, `ham10000`,
`pytorch`, `medical-imaging`, `grad-cam`, `reproducibility`.

Pin the repository on your profile.

---

## 7. Old repository

Decide what to do with the old repository at
`d:\HAM\ham10000-classifier`.  Options:
- Archive it on GitHub (Settings -> Archive this repository).
- Add a README note pointing to this repository.
- Delete it if it was never pushed.
