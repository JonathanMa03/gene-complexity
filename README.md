# Interpretable Gene Programs or Foundation Models?

**Comparing Sparse Dictionary Learning and scGPT for Single-Cell Classification**

**Course:** EN.553.740 Machine Learning 1

**Instructor:** Dr. Sichen Yang

**Group members:** [Jonathan Ma](https://jonathanma03.github.io/), [Sijia Zhu](https://chesleaz.github.io/sijiazhu.github.io/), Jiatong Gao, Yunwei Chai

## Project Overview

This project will compare pretrained scGPT representations with sparse dictionary learning for cell-type classification using public single-cell RNA-sequencing data. Dictionary learning represents each cell as a sparse combination of learned gene programs, providing an interpretable alternative to foundation-model embeddings. Both representations will be evaluated with the same regularized multinomial logistic regression classifier, alongside a baseline using processed gene-expression features directly.

We will compare Macro-F1, balanced accuracy, and predictive log loss, and examine the highest-weight genes and cell-type activation patterns of the learned dictionary atoms. The goal is to understand whether pretrained scGPT provides predictive information beyond a sparse representation learned from the target data, and to characterize the tradeoff between predictive performance and interpretability.

See the [project abstract](docs/references/group1_abstract.pdf) and [project proposal](docs/references/group1_proposal.pdf) for more detail.

## Data

The active pipeline uses **Baron human pancreas raw integer counts**, obtained through Bioconductor's `scRNAseq::BaronPancreasData`. Run `python scripts/download_data.py`, then follow the [runnable workflow](#runnable-workflow) below. Generated datasets are excluded from Git.

The earlier [Pancreas benchmark download](https://exampledata.scverse.org/scvi-tools/pancreas.h5ad) was used for initial exploration only; it is not the raw-count input for this pipeline. Do not substitute it for `baron_raw.h5ad`.

## Methods

TBD — Representation-learning pipelines, classifier setup, and evaluation procedures will be added here.

## Repository Structure

```text
gene-complexity/
├── .github/
│   └── CODEOWNERS
├── code/
│   ├── dict-learn/              # Sparse dictionary learning
│   ├── logit/                   # Logistic classifiers and analysis
│   ├── pipeline/                # Shared representation preparation
│   ├── preprocessing/           # Data preparation
│   └── scgpt/                   # scGPT representations
├── data/                        # Project datasets
├── docs/
│   ├── logging/
│   │   └── CHANGELOG.md
│   ├── notes/
│   │   └── main.tex             # LaTeX reference notes
│   ├── references/
│   │   ├── denAdel.pdf
│   │   ├── group1_abstract.pdf
│   │   └── group1_proposal.pdf
│   ├── Python_git.md
│   └── R_git.md
├── scripts/                     # Runnable workflow entry points
├── results/                     # Experiment outputs and figures
├── .gitignore
├── LICENSE
├── README.md
├── environment_test.ipynb       # Environment checks and data preview
└── requirements.txt            # Python dependencies
```

Empty project directories contain `.gitkeep` placeholders, omitted above for readability.

## Getting Started

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/JonathanMa03/gene-complexity.git
cd gene-complexity
python3 -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies (PyTorch, NumPy, pandas, Matplotlib, Jupyter Notebook, and AnnData), register the project kernel, then launch the environment test:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gene-complexity --display-name "Python (gene-complexity)"
python -m notebook environment_test.ipynb
```

Select the **Python (gene-complexity)** kernel and run all cells. The notebook checks that the kernel uses this repository's `.venv`, imports every dependency in `requirements.txt`, and previews the downloaded dataset. The notebook dependencies include `ipykernel` transitively. On Windows, use `python` instead of `python3` when creating the environment.

Additional dependencies for the project pipelines will be documented as they are implemented.

### Server scGPT pretraining handoff

The [small scGPT pretraining handoff](code/scgpt/pretraining_handoff/README.md)
documents 51 completed server checkpoints, their fixed gene panel/vocabulary,
sampling provenance, dependency versions, and frozen 128-dimensional embedding
extraction. Its dependencies are isolated from the project environment. Large
weights and sampling tables remain outside Git, with locations and checksum-verified
retrieval instructions in the handoff. These are controlled small scGPT models,
not the original large published scGPT checkpoint.

## Team Workflow: Branches and Pull Requests

1. Start from an up-to-date `main` branch with no uncommitted changes:

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

2. Create your own branch for a task. Replace the example with your name and a short task description:

   ```bash
   git switch -c jonathan/data-preprocessing
   ```

3. Make and check your changes, then stage the relevant files and commit. Replace the example path with the files you changed:

   ```bash
   git status
   git diff
   git add code/preprocessing/example.py
   git commit -m "Add initial data preprocessing"
   ```

4. Push your branch to GitHub:

   ```bash
   git push -u origin your-name-here/data-preprocessing
   ```

5. Open the [repository on GitHub](https://github.com/JonathanMa03/gene-complexity) and select **Compare & pull request**, or open **Pull requests → New pull request**. Set the base branch to `main` and the compare branch to your task branch. Describe what changed and how you checked it, then request a teammate's review.

6. Address review feedback by committing and pushing to the same branch; the pull request updates automatically. After review and any required checks pass, merge the pull request into `main` and delete the merged branch on GitHub.

7. Update your local `main` before starting your next task:

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

Keep each pull request focused on one task and use a new branch for subsequent work.

## Runnable workflow

Run these commands from the repository root with `.venv` activated. Python scripts use the active environment. The download step also requires R; its exporter installs missing Bioconductor packages on first use and reuses the cached download afterward.

The active experiment uses **raw-count Baron human pancreas**, not the older normalized pancreas benchmark linked above. Existing shared splits and artifacts should normally be reused. New model/evaluation/summary directories are never overwritten; omitted output paths create timestamped directories under `results/workflow/` (ignored by Git). A failed run can leave a partial directory: fix the cause and choose a new output path.

The default input config is `code/pipeline/configs/validation_v1.json`. Its paths resolve relative to the repository root; absolute paths also work. Direct CLI paths resolve relative to your working directory. To use new artifacts, copy the config and edit:

| Config field | Default | Meaning |
|---|---|---|
| `expression_dir` | `data/processed/baron_classical_v1` | Processed expression and preprocessing provenance |
| `split_dir` | `data/processed/baron_splits_final` | Final shared manifest, settings, exclusions |
| `dictionary_dir` | `results/dict-learning/baron_grid_v1` | Parent of dictionary candidate folders |
| `dictionary_runs` | Four K16/K32, alpha1/alpha5 candidates | Maps CLI candidate names to subfolders |
| `scgpt_path` | `code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz` | Supplied full frozen embeddings |

Each method loads only its required representation, plus the shared expression metadata and final split bundle. A baseline run does not need dictionary/scGPT artifacts. Training and validation never open `test.npz`; the full scGPT archive contains test rows, but those rows are not used for fitting or validation scoring.

### 1. Download data — `scripts/download_data.py`

```bash
python scripts/download_data.py
```

Downloads/caches Baron through `scRNAseq::BaronPancreasData`, then exports counts, genes, and cells into `data/processed/`. Uses `data/raw/baron_human_counts.rds` as its cache. Preserves a complete existing export; rejects a partial export rather than overwriting it.

| Parameter | Default | Meaning |
|---|---|---|
| `--rscript` | `Rscript` | R executable name or path |

### 2. Preprocess and split — `scripts/preprocess_data.py`

For first-time preparation, after scGPT exclusions have been confirmed:

```bash
python scripts/preprocess_data.py --scgpt-confirmed
```

Converts exported counts to `data/processed/baron_raw.h5ad` if absent, creates or verifies final splits, and fits normalization/gene selection/scaling on training cells. Applies the fitted transformation to each split. Existing raw data and split bundles are reused and verified; existing feature outputs are preserved by refusing overwrite.

| Parameter | Default | Meaning |
|---|---|---|
| `--splits` | `data/processed/baron_splits_final` | Existing final bundle or new split output directory |
| `--output` | `data/processed/baron_classical_v1` | New classical feature directory |
| `--seed` | `42` | New split RNG seed; must match a reused bundle |
| `--scgpt-confirmed` | Off | Explicit confirmation required to create final splits |
| `--scgpt-exclusions` | None | Confirmed exclusion CSV with `cell_id`; none means no additional scGPT exclusions |
| `--n-genes` | `2000` | Maximum training-variable genes retained |
| `--target-sum` | `10000` | Per-cell library-size normalization target before log1p |

The agreed **70/15/15 within-cell-type split and exclusion of classes with fewer than 30 cells are fixed**, not tuning parameters. Donors overlap across splits. Classical preprocessing uses all original genes for library totals, training log-variance ranking, and training z-scoring. It is not scGPT input processing.

### 3. Load/check scGPT — `scripts/load_scgpt.py`

```bash
python scripts/load_scgpt.py
# Or import the downloaded handoff into the configured destination:
python scripts/load_scgpt.py --source-dir /path/to/downloaded/handoff
```

Requires all four teammate-supplied files: `baron_scgpt_full.npz`, `baron_scgpt_full.metadata.json`, `checkpoint_selection.json`, and `preparation_report.json`. Validates hashes, finite embeddings, cohort coverage, raw-data provenance, and consistency of checkpoint/preparation metadata. Missing files fail with their paths. It does **not** download private weights, pretrain scGPT, or extract new embeddings. Existing embeddings suffice for downstream classification; weights and sampling archives are not required for that use.

| Parameter | Default | Meaning |
|---|---|---|
| `--config` | Default input config above | Artifact locations |
| `--source-dir` | Configured embedding directory | Optional external directory to validate and copy from; existing destination files are preserved |
| `--checkpoint` | None | Optional extracted checkpoint file, checked against the recorded SHA-256; not a `.tar.gz` archive |
| `--output` | Timestamped directory | Check report destination |

Classical preprocessing and final splits must be available first. The optional weight check verifies file identity, not model execution. Re-extraction remains in `code/scgpt/pretraining_handoff/` and needs its separate dependencies.

### 4. Dictionary learning — `scripts/run_dictionary.py`

```bash
python scripts/run_dictionary.py --atoms 32 --alpha 1 \
  --output results/dict-learning/custom_grid/k32_a1
```

Fits only training expression, then exports the dictionary and train/validation codes. Register the new output under `dictionary_dir`/`dictionary_runs` in a copied input config before classifier use.

| Parameter | Default | Meaning |
|---|---|---|
| `--features` | `data/processed/baron_classical_v1` | Verified classical feature bundle |
| `--output` | Required | New candidate directory |
| `--atoms` | `16` | Number of dictionary atoms/code dimensions |
| `--alpha` | `1.0` | L1 sparsity penalty for learning and encoding |
| `--epochs` | `10` | Epoch ceiling; solver can stop early |
| `--seed` | `42` | Dictionary initialization/shuffling seed |

Uses the existing float64 solver: batch size 128, one thread, Lasso coordinate-descent encoding, 3,000 encoding iterations. These implementation settings are fixed in this wrapper. For the existing configurable multi-candidate grid, use `code/pipeline/dictionary_grid.py`.

### 5. Expression baseline — `scripts/run_baseline.py`

```bash
python scripts/run_baseline.py --C 1 --output results/workflow/baseline_model
```

Fits a logistic classifier directly on processed expression. This is a convenience wrapper for `train_logistic.py --method baseline`; it does not refit expression preprocessing or evaluate any held-out split.

| Parameter | Default | Meaning |
|---|---|---|
| `--config` | Default input config | Classical features and final splits |
| `--C` | `1.0` | Inverse L2 penalty; larger means weaker regularization |
| `--max-iter` | `5000` | Maximum classifier optimization iterations |
| `--tol` | `0.0001` | Classifier stopping tolerance |
| `--seed` | `42` | Recorded estimator random state; lbfgs generally does not use it |
| `--threads` | `1` | Numerical-library thread limit during fitting |
| `--output` | Timestamped directory | New saved-model directory |

### 6. Train logistic classifier — `scripts/train_logistic.py`

```bash
python scripts/train_logistic.py --method dictionary \
  --candidate Dictionary_K32_alpha1 --C 0.0630957344480193 \
  --output results/workflow/dictionary_model
python scripts/train_logistic.py --method scgpt --C 100 \
  --output results/workflow/scgpt_model
```

These C values illustrate usage, not frozen final choices. Fit one C per invocation or use the existing integrated coarse-to-fine search to guide validation selection. Saves `model.npz` (coefficients, intercepts, class order) and `training.json` (configuration, hyperparameters, versions and input/model hashes). No pickle is needed. Nonconvergence fails the run. No extra feature scaling is applied.

| Parameter | Default | Meaning |
|---|---|---|
| `--method` | Required | `baseline`, `dictionary` (sparse codes), or `scgpt` |
| `--candidate` | None | Required for dictionary: key in config `dictionary_runs`; ignored for other methods |
| `--config` | Default input config | Input locations |
| `--C` | `1.0` | Inverse L2 regularization strength |
| `--max-iter` | `5000` | Optimization iteration ceiling |
| `--tol` | `0.0001` | Optimization stopping tolerance |
| `--seed` | `42` | Estimator random state |
| `--threads` | `1` | Fit thread limit |
| `--output` | Timestamped directory | New model output directory |

### 7. Validation evaluation — `scripts/evaluate_validation.py`

```bash
python scripts/evaluate_validation.py --model results/workflow/dictionary_model \
  --output results/workflow/dictionary_validation
```

Loads saved coefficients, verifies unchanged training artifacts, and evaluates validation cells without refitting. Produces `metrics.json`, `predictions.npz` (IDs, labels, probabilities and class order), `per_class.csv`, and `confusion_matrix.csv`.

| Parameter | Default | Meaning |
|---|---|---|
| `--model` | Required | Saved model directory from step 5 or 6 |
| `--output` | Timestamped directory | New evaluation directory |
| `--freeze` | Off | Explicitly declare selection complete and write `selection.json`, binding model, training report and validation metrics hashes |

Use validation to select representation settings and C. **Only add `--freeze` once the team has agreed on final choices.** This is an explicit record of that decision, not an automatic best-model selection. Validation selection scores are not independent test results.

### 8. Final test evaluation — `scripts/run_test.py`

After model selection is complete, rerun validation with `--freeze` into a fresh directory, then:

```bash
python scripts/run_test.py --model results/workflow/dictionary_model \
  --selection results/workflow/dictionary_validation_frozen/selection.json \
  --output results/workflow/dictionary_test
```

| Parameter | Default | Meaning |
|---|---|---|
| `--model` | Required | Frozen saved model directory |
| `--selection` | Required | Matching `selection.json` from explicit validation freezing |
| `--output` | Timestamped directory | New test evaluation directory |

Checks frozen-selection and input hashes. Baseline uses existing test expression; dictionary uses the saved atoms and penalty to encode test expression; scGPT aligns frozen full embeddings by cell ID. **Nothing is refitted on test or train+validation.** Outputs have the same schema as validation. Do not use test scores to revise hyperparameters. The script exists for later use; project test evaluation has not been run while building this framework.

### 9. Summary and optional plots — `scripts/summarize_results.py`

```bash
python scripts/summarize_results.py \
  --evaluations results/workflow/baseline_validation results/workflow/dictionary_validation \
  --plots --output results/workflow/validation_summary
```

| Parameter | Default | Meaning |
|---|---|---|
| `--evaluations` | Required, one or more | Evaluation directories from step 7 or 8 |
| `--plots` | Off | Generate a Macro-F1 comparison chart and confusion-matrix plots |
| `--output` | Timestamped directory | New summary directory |

Writes `summary.csv` and source metrics hashes. Requires the same cell IDs/labels and split across runs; rejects mixing validation and test. Summaries do not choose or freeze models automatically. This script consumes the new saved-model evaluation schema, not older coarse-to-fine search CSVs.

### Existing all-method validation search

The integrated search remains available:

```bash
python code/pipeline/run_validation.py --config code/pipeline/configs/validation_v1.json
```

It searches C for all configured representations and creates a fresh validation report. Its C choices can be supplied to step 6; it does not save the coefficient bundles used by steps 7–8. Test the framework with synthetic data using:

```bash
python -m unittest discover -s scripts -p 'test_*.py'
python -m unittest discover -s code/pipeline -p 'test_*.py'
```

## TA quick start: one-command end-to-end dry run

Here, **dry run means a small real train/validation smoke run**, not a simulated execution. It rebuilds classical features and a dictionary, checks the supplied frozen scGPT embeddings, trains all three classifiers, evaluates validation, and writes a comparison summary. It does not tune hyperparameters, freeze selections, or evaluate the test set. Raw-count preprocessing generates transformed test features as part of its normal bundle, but no classifier scores them.

After installing `requirements.txt` and activating `.venv`, run from the repository root:

```bash
# Check the retained files and provenance hashes without fitting or writing outputs:
python scripts/run_end_to_end.py --check-only

# Rebuild everything downstream in a fresh directory:
python scripts/run_end_to_end.py --plots
```

The default smoke profile uses **all shared train/validation cells**, 256 expression genes, eight dictionary atoms, at most two dictionary epochs, and fixed classifier C=1. These are execution checks, not final experimental settings. scGPT uses the supplied 128-dimensional embeddings unchanged. No R installation, download, GPU, checkpoint weights, or scGPT pretraining is needed for this route.

For a larger validation run using 2,000 genes, 32 atoms and ten dictionary epochs:

```bash
python scripts/run_end_to_end.py --profile standard --plots
```

This still uses fixed C=1 and does not replace model selection. Both profiles reuse the exact shared final split, rather than making new assignments. Existing grid results, models and classical features are not inputs to this runner.

### Files to keep before deleting generated artifacts

**Keep the repository source and configuration**: `scripts/`, `code/`, `requirements.txt`, README/docs, and `.git/`. Many workflow changes may still be uncommitted: commit or back them up before cleaning. Keep `.venv/` to avoid reinstalling dependencies; it is optional if you recreate it.

For an offline end-to-end dry run, these are the **eight required local input files** in addition to source code:

```text
data/processed/
├── baron_raw.h5ad                         # Exact raw-count file used for shared splits
└── baron_splits_final/
    ├── baron_split_manifest.csv           # Frozen train/validation/test assignments
    ├── split_settings.json               # Final status, raw/manifest hashes and settings
    └── baron_exclusions.csv               # Excluded cells; required even if empty

code/scgpt/pretraining_handoff/outputs/baron/
├── baron_scgpt_full.npz                    # Full frozen embeddings and cell IDs
├── baron_scgpt_full.metadata.json          # Embedding hash and extraction provenance
├── checkpoint_selection.json              # Selected checkpoint identity
└── preparation_report.json                # Raw-count/panel preparation provenance
```

Preserve the original files byte-for-byte: rewriting `baron_raw.h5ad` or the manifest can change their hashes. The scGPT reports must refer to this same raw data. **The final split bundle is small but essential; do not delete it with other preprocessing outputs.** `baron_class_counts.csv` and `baron_split_counts.csv` are useful to retain but not required for the runner.

After backing up any results needed for the report, the following generated artifacts are **not needed as inputs** and may be removed for a clean rebuild:

- `data/processed/baron_classical_v1/` and any other generated classical feature bundles.
- `data/processed/baron_splits_draft/`.
- `results/dict-learning/`, `results/logit/`, and `results/workflow/` (previous results/models/figures).
- The R exports `data/processed/baron_counts.mtx`, `baron_genes.csv`, `baron_cells.csv`, and R cache `data/raw/baron_human_counts.rds`, **only if keeping `baron_raw.h5ad`**.

The teammate's weight/sampling archives, gene-mapping CSV, and mapped scGPT `.h5ad` are not required to *consume the supplied embeddings*. Keep/back them up separately if you want to reproduce embedding extraction later. They should not be confused with the four required embedding handoff files above. Do not delete the entire `data/processed/` or `code/scgpt/` directories, and do not use `git clean -fdx`: ignored input files and environments could be deleted.

The runner never deletes anything. It fails before rebuilding if a required retained file is missing or its recorded hash disagrees. `--check-only` checks file presence and provenance hashes; full array, cell alignment, and cohort checks occur during execution.

### End-to-end runner parameters and outputs

| Parameter | Default | Meaning |
|---|---|---|
| `--profile` | `smoke` | `smoke`: 256 genes/8 atoms/2 epochs; `standard`: 2,000 genes/32 atoms/10 epochs |
| `--raw` | `data/processed/baron_raw.h5ad` | Retained raw-count AnnData file |
| `--splits` | `data/processed/baron_splits_final` | Retained final split bundle |
| `--scgpt-dir` | `code/scgpt/pretraining_handoff/outputs/baron` | Directory containing the four handoff files |
| `--output` | Timestamped `results/workflow/end_to_end_<profile>_*` | New run directory; existing paths rejected |
| `--check-only` | Off | Check inputs without writing outputs or fitting |
| `--n-genes` | Profile default | Override training-variable gene count |
| `--atoms` | Profile default | Override dictionary dimensions |
| `--epochs` | Profile default | Override dictionary epoch ceiling |
| `--alpha` | `1.0` | Dictionary L1 penalty |
| `--C` | `1.0` | Fixed inverse L2 classifier penalty, shared across methods; no C search |
| `--threads` | `1` | Classifier fitting thread limit; dictionary remains one thread |
| `--plots` | Off | Add comparison and confusion-matrix PNGs |

The raw/split/handoff defaults are repository-relative even when invoked elsewhere; explicit relative CLI paths use the current directory. Fixed settings are seed 42, normalization target 10,000, classifier `max_iter=5000`, and classifier `tol=0.0001`. The retained split is never regenerated.

Every run contains its own `config.json`, `features/`, `dictionary/`, `scgpt_check/`, `models/`, `validation/`, and `summary/`. Open **`summary/summary.csv`** for the comparison and **`end_to_end_report.json`** for status, settings, source/input hashes, versions and stage timings. Numbered stage logs identify failures. Partial failed runs are preserved; fix the problem and choose a fresh output directory.

For a named run:

```bash
python scripts/run_end_to_end.py --plots --output results/workflow/ta_demo
```

Choose a different `--output` next time, or omit it for a new timestamped directory. The smoke run rebuilds preprocessing and downstream models from the retained files; it verifies and reuses scGPT embeddings rather than claiming to reproduce their pretraining/extraction.

## Model selection: one dictionary-grid refinement

After v1 validation, run the agreed refinement with a single command:

```bash
python scripts/tune_models.py --output results/workflow/model_selection_v2
```

This keeps preprocessing, shared splits, seed 42, and the existing dictionary solver fixed. `code/pipeline/configs/dictionary_grid_v2.json` contains **atoms `[32, 64]`, alpha `[0.5, 1, 2]`, and ten maximum epochs**. It reuses the v1 32-atom/alpha-1 candidate only if its input hashes, source hash, parameters, output hashes and NumPy/scikit-learn versions match; five other candidates are fitted. If a cached candidate exists but fails verification, the command fails instead of silently mixing experiments. Use `--reuse-grid` pointing to an empty directory to refit all candidates.

Each of the six dictionary candidates, expression, and scGPT receives the same existing coarse-to-fine logistic C search: coarse `[0.01, 0.1, 1, 10, 100, 1000]`, then 21 log-spaced values from the coarse winner divided by ten to multiplied by ten, excluding already tested values. Only converged fits are eligible. This normally gives 192 total classifier fits; endpoint coarse winners can produce one additional fine value. C is selected separately for each representation. All numerical fitting uses one thread, and installed package versions are recorded in `environment.txt` and the validation run report. This records the actual environment rather than guaranteeing bitwise agreement across platforms.

Selection rules, fixed before this run:

- Highest validation Macro-F1; classifier score ties prefer smaller C.
- Across dictionary candidates, exact Macro-F1 ties prefer fewer atoms, then fewer mean active atoms on validation. Remaining exact ties use candidate ID for deterministic ordering.
- Sparsity, solver warnings and gene-program interpretation remain review criteria. A ranking is a recommendation, not an automatically frozen final model.

| Parameter | Default | Meaning |
|---|---|---|
| `--inputs` | `code/pipeline/configs/validation_v1.json` | Existing expression, final split and scGPT paths; its dictionary mapping is replaced by the new grid |
| `--grid` | `code/pipeline/configs/dictionary_grid_v2.json` | Fixed dictionary candidate grid for this run |
| `--reuse-grid` | `results/dict-learning/baron_grid_v1` | Verified matching dictionaries are copied into the new run; missing ones are fitted |
| `--output` | Timestamped `results/workflow/model_selection_v2_*` | New output root; existing directories rejected |

Review `dictionary_ranking.csv`, `recommended_models.csv`, and `selection_report.json`. Full C-search results and per-class analysis are under `validation/classification/`; reconstruction and sparsity diagnostics are under `dictionary_grid/`. Each run preserves its config snapshots and existing v1 artifacts. Review the results before using the saved-model scripts to train selected models and explicitly freeze them. This command never opens classical test features or evaluates test cells.

### Reviewed and frozen v2 models

The review and fixed choices are recorded in [the model-selection note](docs/notes/model_selection_v2.md). Saved classifiers and explicit selection files are under `results/workflow/frozen_models_v2/`; these local files and the source artifacts they reference must be backed up before any cleanup. Final test evaluation is complete; see [the final held-out results](docs/notes/final_test_v2.md).

To reproduce the descriptive gene-weight/activity review:

```bash
python scripts/review_dictionary.py \
  --config results/workflow/model_selection_v2/validation_config.json \
  --candidate Dictionary_k64_a2.0_s42_e10 --plots
```

| Parameter | Default | Meaning |
|---|---|---|
| `--config` | `code/pipeline/configs/validation_v1.json` | Input-artifact config; use the v2 run config for the selected v2 dictionary |
| `--candidate` | Required | Dictionary name in config |
| `--top-genes` | `10` | Genes exported per sign per atom |
| `--display-atoms` | `8` | Heatmap atoms, ranked by training-only cell-type contrast |
| `--plots` | Off | Write the train/validation activity heatmap |
| `--output` | Timestamped `results/workflow/dictionary_review_*` | New output directory |

This exports signed gene weights, per-class activities, atom diagnostics and a review report. It does not perform pathway enrichment, infer causality, or measure stability across random seeds.

### Final held-out test results

The frozen models were evaluated without refitting on 1,275 shared test cells:

| Representation | Macro-F1 | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|---:|
| Processed expression | 0.993913 | 0.994510 | 0.993687 | 0.019454 |
| Dictionary (64 atoms, alpha 2) | 0.980962 | 0.989020 | 0.977933 | 0.036989 |
| Frozen scGPT | 0.962358 | 0.968627 | 0.955637 | 0.096535 |

See [the evaluation record](docs/notes/final_test_v2.md) for limitations and verification. Predictions, per-class metrics and plots are under `results/workflow/frozen_models_v2/test/`. Model settings remain frozen; these test scores must not be used for further tuning.
