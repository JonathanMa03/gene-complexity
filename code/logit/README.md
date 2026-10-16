# Logistic Regression Downstream Classification

## 1. Overview

This module compares six feature representations for downstream cell-type classification on the Baron dataset:

- Expression features (2,000 dimensions)
- Dictionary Learning K16, alpha=1
- Dictionary Learning K16, alpha=5
- Dictionary Learning K32, alpha=1
- Dictionary Learning K32, alpha=5
- scGPT embeddings (128 dimensions)

All representations are evaluated using Logistic Regression with the same training and validation splits.

## 2. Data Preparation

Features are aligned by cell ID to ensure consistent sample ordering and labels across representations.

The scGPT embeddings are loaded from `baron_scgpt_full.npz`, using the `vectors` and `cell_ids` fields. Only cells included in the reference train/validation splits are used.

No additional feature scaling is applied in the downstream classifier.

## 3. Hyperparameter Tuning

A two-stage coarse-to-fine grid search is used to select the regularization parameter C independently for each feature representation.

**Stage 1: Coarse Search**

Candidate values: 0.01, 0.1, 1, 10, 100, 1000.

**Stage 2: Fine Search**

The search is refined around each representation's best coarse C, using 21 logarithmically spaced candidates between C/10 and 10C, excluding previously evaluated values.

A total of 144 Logistic Regression fits are performed across six representations.

The best C is selected by maximizing validation Macro-F1.

## 4. Validation Results

| Representation | Best C | Macro-F1 |
|---|---:|---:|
| Expression | 0.316228 | 0.984365 |
| Dictionary K32 alpha1 | 0.063096 | 0.982813 |
| scGPT | 630.957344 | 0.964146 |
| Dictionary K16 alpha1 | 5.011872 | 0.954980 |
| Dictionary K16 alpha5 | 0.316228 | 0.900204 |
| Dictionary K32 alpha5 | 158.489319 | 0.878918 |

The Dictionary K32 alpha1 representation achieves performance close to the expression baseline while using substantially fewer features.

These are validation results, not independent test results.

## 5. How to Run

Prepare the required expression, dictionary, scGPT, and finalized manifest files in their expected project locations.

From the project root, run:

```bash
python code/logit/run_all_validation_coarse_to_fine.py
```

Generated outputs are saved under `results/logit/`:

- `validation_coarse_runs.csv`
- `validation_fine_runs.csv`
- `validation_all_runs.csv`
- `validation_best_C.csv`

Generated outputs are excluded from Git.

## 6. Current Status

Completed:
- Training/validation feature alignment
- Logistic Regression classification
- Coarse-to-fine hyperparameter tuning
- Overall validation performance comparison

Pending:
- Per-class F1 analysis
- Confusion matrix analysis

The test set has not been used. Final test evaluation will be performed after the experimental procedure is finalized.