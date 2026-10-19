# Logistic Regression Downstream Classification

## Overview

This module evaluates how well different feature representations support downstream **cell-type classification** on the Baron single-cell dataset. It compares an expression baseline, four dictionary-learning representations, and scGPT embeddings using the same logistic-regression training, validation, and hyperparameter-selection procedure.

The main experiment is implemented in **`run_all_validation_coarse_to_fine.py`**. The follow-up validation analysis is implemented in **`analyze_validation.py`**. The other Python files are supporting baseline and data-inspection utilities.

## Scripts

| File | Purpose |
| --- | --- |
| `run_all_validation_coarse_to_fine.py` | **Main experiment.** Align all six representations, run two-stage C tuning, and save validation metrics. |
| `analyze_validation.py` | **Follow-up analysis.** Use selected C values to calculate per-class metrics and generate validation confusion matrices. |
| `expression_baseline.py` | Standalone expression-feature logistic-regression baseline. |
| `dictionary_baseline.py` | Standalone dictionary-code logistic-regression baseline. |
| `scgpt_baseline.py` | Standalone scGPT-embedding logistic-regression baseline. |
| `inspect_data.py` | Inspect the expression/dictionary input files and their structure. |
| `inspect_scgpt.py` | Inspect scGPT embedding files and cell IDs. |

The supporting baseline and inspection scripts were used during development and troubleshooting. Run the **main experiment first**, then run `analyze_validation.py` for the follow-up per-class evaluation. Earlier intermediate comparison scripts are not required to reproduce these results.

## Feature Representations

| Representation | Number of features | Feature used in classification | NPZ key |
| --- | ---: | --- | --- |
| Expression | 2,000 | Gene expression matrix $X$ | `X` |
| Dictionary K16, alpha=1 | 16 | Dictionary codes $Z$ | `vectors` |
| Dictionary K16, alpha=5 | 16 | Dictionary codes $Z$ | `vectors` |
| Dictionary K32, alpha=1 | 32 | Dictionary codes $Z$ | `vectors` |
| Dictionary K32, alpha=5 | 32 | Dictionary codes $Z$ | `vectors` |
| scGPT | 128 | scGPT cell embeddings | `vectors` |

For dictionary learning, the expression matrix is approximated as $X \approx ZD$, where $Z$ contains **cell-level dictionary codes** and $D$ contains the learned dictionary (gene programs). The downstream classifier uses **$Z$, not $D$**. In the supplied `train_codes.npz` and `validation_codes.npz` files, these codes are stored under the NPZ key `vectors`. This key is a storage-field name, not the mathematical name of the matrix. scGPT also uses a `vectors` key, but its contents are scGPT embeddings rather than dictionary codes.

## Required Inputs and Paths

Paths are resolved relative to the repository root in the main script.

**Expression features** — `data/processed/baron_classical_v1/`
- `train.npz`: `X`, `cell_ids`, `labels`
- `validation.npz`: `X`, `cell_ids`, `labels`

**Dictionary features** — `results/dict-learning/baron_grid_v1/`
- `k16_a1.0_s42_e10/`
- `k16_a5.0_s42_e10/`
- `k32_a1.0_s42_e10/`
- `k32_a5.0_s42_e10/`

Each dictionary run needs `train_codes.npz` and `validation_codes.npz`, with `vectors` (the cell-level codes $Z$) and `cell_ids` fields.

**scGPT embeddings** — `code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz`
- Required fields: `vectors`, `cell_ids`.

**Finalized split manifest** — A CSV with `cell_id` and `split` columns. The script checks common paths such as `data/processed/baron_classical_v1/split_manifest.csv` and `data/processed/baron_split_manifest.csv`. If the correct finalized manifest is elsewhere or multiple candidates exist, update `find_final_manifest()` to select the correct file explicitly.

Input feature bundles are obtained from the team handoff; this script does not generate dictionary codes or scGPT embeddings.

## Data Alignment and Preprocessing

1. Load the expression training and validation files as the reference splits and labels.
2. Verify that their cell-ID membership agrees with the finalized split manifest.
3. Reorder each dictionary-code matrix $Z$ (saved as `vectors`) by `cell_ids` to match the reference train/validation row order.
4. Read scGPT `vectors` and `cell_ids` from the full embedding file, exclude cells not in the finalized manifest, and align the remaining rows to the reference splits. The expected number of excluded cells in this handoff is **63**; the script prints the actual number and warns if it differs.
5. Check feature dimensions, duplicate IDs, missing reference cells, and non-finite feature values.

The reference data in the current run contain **5,954 training cells** and **1,277 validation cells**.

**No additional `StandardScaler` or other downstream feature scaling is applied.** The experiment uses the upstream feature representations as provided. Their upstream preprocessing may differ, so comparisons reflect both the learned representation and its preprocessing.

## Classifier and Hyperparameter Tuning

All six representations use scikit-learn `LogisticRegression` with:

- Solver: `lbfgs`
- Default L2 regularization
- `max_iter=5000`
- `random_state=42`
- Hyperparameter: `C` (inverse regularization strength)
- Selection metric: **validation Macro-F1**
- Tie-breaking rule: prefer the smaller `C` when scores are equal

The procedure is a **two-stage, validation-based grid search**, run independently for each representation:

**Stage 1 — Coarse search**

Test `C = [0.01, 0.1, 1, 10, 100, 1000]` and select the converged model with the highest validation Macro-F1.

**Stage 2 — Fine search**

Generate 21 log-spaced candidates between `coarse_best_C / 10` and `coarse_best_C * 10`, skipping values already tested in Stage 1. With the current grid, this adds 18 new fits per representation.

Finally, select the highest-scoring **converged** candidate across both stages. Each candidate is fit on the training split and scored on the validation split.

**Total: 36 coarse fits + 108 fine fits = 144 fits.** This is *not* K-fold cross-validation. No new data splits are generated, and the test set is not loaded or used.

## Run Instructions

Install the Python dependencies in your project environment:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

Place the required NPZ files and finalized manifest in the locations described above. From the repository root, run:

```bash
python code/logit/run_all_validation_coarse_to_fine.py
```

The main script resolves input paths relative to its own location, prints the coarse and fine results, and saves CSV files to `results/logit/`.

After the main experiment has generated `results/logit/validation_best_C.csv`, run the follow-up analysis:

```bash
python code/logit/analyze_validation.py
```

This script loads the same train/validation representations and reads each feature set's selected `C` from `validation_best_C.csv`. It fits one logistic-regression model per feature set with the same settings, calculates per-class metrics, and creates confusion matrices. **It does not repeat the C search or load the test set.**

## Environment and Reproducibility

The experiments were run using the project's Python
virtual environment (`.venv`).

| Package | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |


## Validation Results

Results from the completed two-stage search (Macro-F1 is the tuning criterion):

| Feature set | Best C | Accuracy | Macro-F1 | Balanced accuracy | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expression | 0.316228 | 0.990603 | **0.984365** | 0.984698 | 0.034559 |
| Dictionary K32, alpha=1 | 0.063096 | 0.989037 | 0.982813 | 0.981463 | 0.037899 |
| scGPT | 630.957344 | 0.972592 | 0.964146 | 0.961069 | 0.087737 |
| Dictionary K16, alpha=1 | 5.011872 | 0.979640 | 0.954980 | 0.955633 | 0.086279 |
| Dictionary K16, alpha=5 | 0.316228 | 0.950666 | 0.900204 | 0.888690 | 0.154693 |
| Dictionary K32, alpha=5 | 158.489319 | 0.949099 | 0.878918 | 0.874847 | 0.226668 |

Expression has the highest validation Macro-F1. Dictionary K32, alpha=1 is close (0.982813 vs. 0.984365) while using 32 rather than 2,000 features. This is promising for low-dimensional prediction, but does not by itself establish statistical equivalence or biological interpretability.

These scores were used for hyperparameter selection and should **not** be presented as independent test performance.

## Generated Outputs

The main script creates these files under `results/logit/`:

| Output | Contents |
| --- | --- |
| `validation_coarse_runs.csv` | Metrics for all coarse-grid candidates. |
| `validation_fine_runs.csv` | Metrics for all new fine-grid candidates. |
| `validation_all_runs.csv` | Both stages combined, including `C`, performance metrics, and convergence status. |
| `validation_best_C.csv` | Selected `C` and validation metrics for each of the six representations. |

The follow-up script writes the following files under `results/logit/validation_analysis/`:

| Output | Contents |
| --- | --- |
| `validation_per_class_metrics.csv` | Per-class Precision, Recall, F1-score, and Support for all six feature representations. |
| `confusion_matrices/confusion_matrix_<feature_set>.csv` | One confusion matrix with true classes as rows and predicted classes as columns for each representation. |
| `confusion_matrices/confusion_matrix_<feature_set>.png` | A visualization of each confusion matrix. |

The analysis also prints the validation Macro-F1 and compares it with the corresponding result in `validation_best_C.csv`. Six confusion matrix CSVs and six PNGs are produced.

Generated outputs are **local files and are excluded from Git** using `.gitignore`. They can be reproduced with the same input features, split manifest, packages, and environment.
