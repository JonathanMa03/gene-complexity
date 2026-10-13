# Preprocessing steps

This guide covers the Baron human pancreas data download, shared splits, and feature preparation for dictionary learning and logistic classification. Run commands from the repository root. The shared splits are finalized; the classical feature recipe below is an initial baseline to confirm with the classifier teammate before model tuning.

## 1. Start from integrated main and set up the environment

After the pipeline integration PR is merged into `main`, the classifier teammate should start a new branch from it. Commit or stash existing work before switching branches.

```bash
git switch main
git pull --ff-only origin main
git switch -c classifier-analysis
```

The `code/pipeline/` code must be present on that version of `main`. Until integration is merged, it is available on `pipeline-integration`; an older `main` will not contain the classical preparation script.

Create the virtual environment only if it does not already exist:

```bash
python3 -m venv .venv
```

Activate it and install dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `python` and activate it in PowerShell with `.venv\Scripts\Activate.ps1`. R and `Rscript` must also be installed for the initial dataset export. Generated data is ignored by Git and does not arrive when pulling a branch: obtain the team's existing bundle or generate it using the steps below.

## 2. Download and export Baron counts

```bash
Rscript code/preprocessing/export_baron.R
```

The script installs missing R dependencies, creates `data/raw/` and `data/processed/`, and downloads `scRNAseq::BaronPancreasData(which = "human")` if `data/raw/baron_human_counts.rds` is absent. It reuses the cached RDS on subsequent runs. First use requires internet access and a writable R package library. On macOS, the script detects existing Homebrew OpenSSL 3 libraries for package compilation.

Source: Baron et al., [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133). This replaces the earlier combined Pancreas tutorial dataset; do not use `data/raw/pancreas.h5ad` for this pipeline.

Exports:

- `data/processed/baron_counts.mtx`: sparse genes × cells counts.
- `data/processed/baron_genes.csv`: ordered gene symbols.
- `data/processed/baron_cells.csv`: cell IDs, donor IDs, and cell-type labels.

## 3. Build AnnData and the panel mapping

```bash
python code/preprocessing/build_baron_h5ad.py
python code/preprocessing/map_baron_panel.py
```

These create:

- `data/processed/baron_raw.h5ad`: **8,569 cells × 20,125 genes**, four donors, 14 cell types; raw integer counts in `X` and `layers["counts"]`.
- `data/processed/baron_scgpt_gene_mapping.csv`: supplied-panel Ensembl IDs, symbols, presence flags, and source-column indices.

Metadata fields are `obs_names` for cell IDs, `obs["cell_type"]` for labels, `obs["donor"]` for donor IDs, and `var_names` for gene symbols. Preserve these IDs throughout the pipeline. This mapping CSV does not itself change the H5AD gene identifiers.

The scGPT owner validated **1,879/2,000 panel genes**, no zero-panel cells among all 8,569 cells, and successful eight-cell frozen extraction. Thus the agreed cohort has **no additional scGPT cell exclusions**. This establishes execution compatibility, not absence of pretraining overlap.

The conversion/mapping scripts overwrite their generated outputs when rerun. If the team has already supplied verified files, reuse them rather than regenerating unnecessarily.

## 4. Use the finalized shared splits

If `data/processed/baron_splits_final/` already exists, reuse it. Otherwise generate it:

```bash
python code/preprocessing/create_splits.py \
  --output-dir data/processed/baron_splits_final \
  --scgpt-confirmed
```

The confirmation flag refers to the completed scGPT check on this Baron input. A changed dataset or exclusion list requires a new compatibility check and a new versioned bundle.

Policy: seed **42**, random **70/15/15 splits stratified by cell type**, excluding classes with **fewer than 30 cells** after any scGPT exclusions. This is a same-donor-pool evaluation: donors can occur in multiple splits. Per-class integer rounding means total fractions are approximate.

| Partition | Cells |
|---|---:|
| Training | 5,954 |
| Validation | 1,277 |
| Test | 1,275 |
| Excluded | 63 |

Ten classes remain; the excluded classes are T cell, Schwann, epsilon, and mast. Every retained class appears in every split.

The bundle contains `baron_split_manifest.csv` (`cell_id, cell_type, donor, split`), exclusions with reasons, class/split counts, and `split_settings.json`. Do not reshuffle or create separate splits inside any model branch. The script refuses to overwrite an existing output directory.

Expected SHA256 values for the agreed bundle:

- Raw H5AD: `452177f01994c2f81ad79cb31b1aaf562b12ff4d56d52299dce1f5ed18c08176`
- Manifest CSV: `bb1e6f0810e20d02400892743c49077323d30bb67e57e043387da84115affca5`

A fresh R/package release can produce a different serialized file. Investigate hash differences and coordinate the canonical bundle rather than changing recorded hashes just to bypass validation.

## 5. Prepare shared classical features

If the agreed `data/processed/baron_classical_v1/` bundle is already present, reuse it. To generate it for the first time:

```bash
python code/pipeline/prepare_classical.py
```

The loader validates final split status, source/manifest hashes, unique IDs, labels/donors, cohort coverage, and class presence. It aligns raw data to the manifest by cell ID.

The initial normalization and feature recipe is:

1. **Library-size normalization:** divide each cell's counts by its total across all original genes, then multiply by **10,000**. The fixed target and the cell's own counts require no fitting across cells. Zero-total cells are rejected.
2. **Log transformation:** apply `log1p`, i.e. `log(1 + normalized_count)`.
3. **Training-only gene selection:** rank genes by variance of log-normalized expression in training cells and retain up to **2,000** nonconstant genes. This is simple variance ranking, not dispersion-adjusted highly variable gene selection. Selected genes are exported in original gene order.
4. **Training-only standardization:** subtract each selected gene's training mean and divide by its training population standard deviation (`ddof=0`). Reuse those exact parameters for validation and test. No clipping is applied.

Only the selected feature matrix is made dense, one split at a time, and exported as float32. These are signed, centered features. Dictionary learning and the processed-expression classifier baseline should use the **same classical feature bundle**. Do not apply this transformation to scGPT inputs.

Default outputs under `data/processed/baron_classical_v1/`:

| File | Contents |
|---|---|
| `train.npz` | `X` (5,954 × 2,000), `cell_ids`, `labels`, `donors`, `gene_ids` |
| `validation.npz` | Same fields, 1,277 × 2,000 features |
| `test.npz` | Same fields, 1,275 × 2,000 features |
| `preprocessing.npz` | Selected columns, original/selected gene IDs, training mean/scale/variance, target sum, training cell IDs |
| `split_manifest.csv` | Shared cell assignments |
| `preprocessing.json` | Recipe, versions, settings, and source/artifact hashes |

Changing `--n-genes` or `--target-sum` requires a fresh `--output` directory and agreement with the other classical pipeline owner. Existing outputs are never overwritten. These preprocessing defaults remain provisional; choose changes using training/validation information, not final test results.

## 6. Validate before classifier development

```bash
python -m unittest discover -s code/preprocessing -p 'test_*.py'
python -m unittest discover -s code/pipeline -p 'test_*.py'
```

Inspect the saved settings and verify the expected shapes and ID alignment. The tests exercise splitting and transformation behavior; they do not replace inspection of real artifact provenance.

Example loading code, run from the repository root:

```python
from pathlib import Path
import numpy as np

root = Path("data/processed/baron_classical_v1")
with np.load(root / "train.npz", allow_pickle=False) as data:
    X_train = data["X"]
    y_train = data["labels"]
    train_cell_ids = data["cell_ids"]
    gene_ids = data["gene_ids"]
```

Load validation and test using the same keys. Confirm identical gene ordering across all three files. The expression baseline uses these `X` matrices directly: do not normalize or z-score them again by default.

## 7. Logistic-classifier integration contract

Once the classifier teammate branches from integrated `main`:

- **Expression baseline:** fit on classical training `X` and `labels`; use the matching validation/test files.
- **Dictionary representation:** fit the dictionary on training features only; encode other splits with the dictionary fixed. Consume exported codes with their cell IDs. Real-data dictionary artifacts are a separate implementation step and are not yet produced by this guide.
- **scGPT representation:** obtain full frozen embeddings and cell IDs for the agreed checkpoint from its owner. The existing eight-cell smoke output is not sufficient. scGPT may export all 8,569 cells; explicitly select the 8,506 retained manifest IDs and record the 63 excluded cells rather than silently taking an arbitrary intersection.
- Join every representation by **cell ID**, reject duplicates or missing retained cells, and align labels/splits from the same manifest. Never rely on row position across independently generated files.
- Use the same L2-regularized multinomial logistic model family and agreed regularization-tuning procedure across representations. If additional representation-specific scaling is needed for codes or embeddings, fit it on training features only and document it.
- Keep one class-to-probability-column mapping when computing log loss. Select settings on validation data; reserve final test evaluation until decisions are frozen. Report Macro-F1, balanced accuracy, and multiclass log loss on the same held-out cells.
- If replacing the fixed validation procedure with cross-validation, refit learned preprocessing and dictionary learning within each training fold. The already-transformed full training bundle is not a substitute for fold-specific preprocessing.

scGPT uses its separate environment and raw-count/panel preparation described in [the Baron handoff](../../code/scgpt/pretraining_handoff/baron/README.md). Primary checkpoint choice and pretraining-overlap limitations must be documented before interpreting the final comparison.

Track code, documentation, and tests in Git. Share generated data/feature bundles separately, or reproduce them with the recorded versions and hashes. Do not commit raw datasets, model weights, or local virtual environments.

## 8. Dictionary grid and follow-up after logistic validation

Run `python code/pipeline/dictionary_grid.py` for the provisional grid in
`code/pipeline/configs/dictionary_grid_v1.json`. It fits 16/32 atoms with penalties
1.0/5.0, seed 42, and a 10-epoch ceiling on training cells only. It exports each
candidate's training/validation codes and diagnostics under
`results/dict-learning/baron_grid_v1/`, with a `grid_summary.csv` index. It does
not select a final dictionary or open the test features.

**When the classifier teammate returns initial logistic validation results,
revisit the grid together and change/add candidates in a new versioned config
and output directory as needed.** Record the revision rationale and keep a
modest search budget. Compare validation classification, sparsity, and
interpretability; reconstruction alone is not the selection criterion. Keep
final test evaluation out of this loop.

The third method requires no additional representation learner: the
processed-expression baseline uses the existing classical `X` matrices directly.
The dictionary and scGPT methods use their code/embedding matrices. Fit separate
logistic models, align all three by cell ID, and use the same manifest and
classifier tuning procedure. Do not concatenate the representations by default.
