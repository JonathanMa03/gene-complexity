# Shared real-data inputs and classical preprocessing

From the repository root with `.venv` activated:

```bash
python code/pipeline/prepare_classical.py
```

Requires local `data/processed/baron_raw.h5ad` and the finalized `data/processed/baron_splits_final/` bundle. No new splits are made. The loader verifies raw-file and manifest hashes, final/scGPT-confirmed status, ID uniqueness, cohort partitioning, class coverage, split sizes, and matching labels/donors. It aligns rows by cell ID, not file position. The shared counts and manifest remain unchanged.

## Initial baseline choices

These defaults are an initial classical baseline to coordinate with the classifier owner, not tuned settings or a finalized biological recipe:

1. Normalize each cell to 10,000 total counts using **all original genes**.
2. Apply `log1p`.
3. Select up to 2,000 genes with highest variance **among training cells only**. This is simple log-variance ranking, not dispersion-adjusted HVG selection. Constant training genes are excluded; ties use source gene order.
4. Center and scale each selected gene with its training mean and population standard deviation. Apply the frozen parameters to validation/test. No clipping is applied.

Per-cell library normalization uses only that cell's counts and a fixed target, so it does not learn across held-out cells. Only the selected gene matrix is made dense, one split at a time. The resulting float32 features are signed and appropriate for the signed dictionary baseline. **Do not pass these features to scGPT**, which has its own raw-count/panel processing.

Use `--n-genes` and `--target-sum` to change settings and `--output` to select a fresh output directory. Existing runs are never overwritten. Changing settings constitutes a new experiment; do not choose them using final test performance. Cross-validation requires refitting gene selection and scaling in each training fold.

## Outputs

Default directory: `data/processed/baron_classical_v1/` (ignored by Git).

- `train.npz`, `validation.npz`, `test.npz`: `X`, `cell_ids`, `labels`, `donors`, and identically ordered `gene_ids`.
- `preprocessing.npz`: selected source-column indices, ordered original/selected gene IDs, training means/scales/variances, target sum, and training cell IDs.
- `split_manifest.csv`: the shared cell assignments.
- `preprocessing.json`: methods, settings, source/manifest/artifact hashes, and versions.

Use `np.load(path, allow_pickle=False)`. For new cells, require identical original gene order before using the saved column indices and `transform`; the lower-level transform accepts a matrix and cannot check gene names itself. The validated loader establishes this order for the current cohort.

Dictionary learning should fit on `train.npz` only and encode the other splits using its fixed dictionary. The expression classifier should use these same `X` arrays directly. Dictionary codes and scGPT embeddings must retain cell IDs so the classifier can align them with the shared manifest. Full model fitting and classifiers are not implemented here.

```bash
python -m unittest discover -s code/pipeline -p 'test_*.py'
```

## Train/validation dictionary pilot while waiting for scGPT

```bash
python code/pipeline/dictionary_pilot.py
```

Alternatively open `code/dict-learn/baron_pilot.ipynb` in the project kernel and Run All. The notebook invokes the same implementation and plots diagnostics.

The pilot uses all 5,954 training cells, 16 atoms, penalty 1.0, mini-batches of 128, and at most 10 epochs. These are provisional engineering defaults, not selected hyperparameters. Mini-batch learning avoids the cost of the synthetic demo's batch solver on real data. Both training and transformation use Lasso coordinate descent with the same penalty. See the [scikit-learn API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html).

It verifies feature hashes, labels, donors, cell/gene ordering, and training provenance; fits only training features; and encodes training/validation with a frozen dictionary. It never opens `test.npz`. It exports:

- `dictionary.npz`: atoms, ordered genes, training cell IDs, and encoding penalty.
- `train_codes.npz`, `validation_codes.npz`: `vectors`, `cell_ids`, `labels`, `donors`.
- `pilot_report.json`: reconstruction MSE versus zero, sparsity, unused atoms, iterations/steps, captured solver warnings, versions, and input/output hashes.

Default output is `results/dict-learning/baron_pilot_v2/`, ignored by Git. Existing outputs are preserved; use `--output` for another run. Options include `--atoms`, `--alpha`, `--epochs`, and `--seed`. Warnings and epoch-budget exhaustion are reported rather than silently treated as convergence. The script verifies encoding from the saved dictionary on eight validation rows. No classifier, scGPT comparison, automatic tuning, or final test evaluation is performed. Reconstruction improvements alone do not establish classification utility.

The solver runs in float64 to avoid float32 coordinate-descent precision warnings; no new scaling is fitted. The earlier local `baron_pilot_v1/` run had numerical warnings and is superseded by `baron_pilot_float64/`. The corrected pilot had zero warnings, validation relative reconstruction MSE about 0.802, and about 72% active coefficients. This modest pilot is not yet particularly sparse and is not a tuned final model.

## Configurable dictionary candidate grid

```bash
python code/pipeline/dictionary_grid.py
```

The starter config is `configs/dictionary_grid_v1.json`: atoms `[16, 32]`, penalties `[1.0, 5.0]`, seed `[42]`, and at most 10 epochs. This is a small provisional grid, not a final search budget. Each candidate fits on all training cells and exports training/validation codes through the same checked pilot implementation. Test features are never opened.

Outputs are under `results/dict-learning/baron_grid_v1/`. Each candidate has its own dictionary, codes, and report; `grid_summary.csv` aggregates reconstruction, sparsity, and warnings and gives artifact directories for the classifier teammate. `grid_config.json` and `grid_report.json` record settings and provenance. No best model is automatically selected.

Rerunning verifies and skips completed matching candidates. Source features, model script, library versions, parameters, and output hashes must match; incomplete or changed artifacts are rejected. Preserve failed run directories for diagnosis and use a new output root. The grid config cannot change within an existing root.

**After the classifier teammate supplies initial logistic train/validation results, review them together and revise the candidate grid as needed.** Create `dictionary_grid_v2.json` and a new output root rather than editing historical runs:

```bash
python code/pipeline/dictionary_grid.py \
  --config code/pipeline/configs/dictionary_grid_v2.json \
  --output results/dict-learning/baron_grid_v2
```

Create that v2 config first; it is not supplied yet. Choose a modest agreed search budget using validation classification results, sparsity, and program interpretation. More validation searches can overfit validation, so document revisions. Do not select from reconstruction alone or use test performance to revise the grid. After choosing settings, freeze the pipeline before final test encoding/evaluation.

### The three classifier inputs

1. **Processed-expression baseline:** `baron_classical_v1/train.npz` and `validation.npz`, field `X`. Already available; it does not need dictionary fitting or its own dictionary grid.
2. **Dictionary representation:** each candidate's `train_codes.npz` and `validation_codes.npz`, field `vectors`.
3. **scGPT representation:** the full embedding NPZ, field `vectors`, joined by cell ID to the same shared manifest and filtered to its retained cohort.

Each representation feeds a separately fitted logistic classifier under the same agreed classifier tuning procedure. These are alternative feature sets, not concatenated features. Additional code/embedding scaling, if chosen, must be training-fitted. The classifier owner supplies validation classification metrics and uses the same class/probability mapping across comparisons.
