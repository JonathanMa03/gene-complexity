# Shared preprocessing and validation orchestration

Run from the repository root with `.venv` activated:

```bash
python -m gene_complexity prepare-classical
python -m gene_complexity run-validation --config gene_complexity/classification/configs/validation_v1.json
```

`classical.py` requires raw Baron AnnData and the final shared split bundle. It verifies hashes, IDs, labels, donors, class coverage, and split membership. Each cell is normalized to 10,000 total counts across all original genes and log1p transformed. The default selects 2,000 genes by training-only variance and applies training-fitted gene means and population standard deviations to each split. This is classical preprocessing, not scGPT input processing.

The output under `data/processed/baron_classical_v1/` contains train/validation/test NPZ features, ordered IDs and gene names, fitted preprocessing arrays, a manifest copy, and a provenance report. Use fresh `--output` directories when changing `--n-genes` or `--target-sum`. Fold-based evaluation must refit transforms within each training fold.

`validation.py` coordinates the logistic coarse-to-fine C search and per-class analysis using verified features. The input config names expression, split, dictionary, and scGPT paths; paths resolve relative to the repository root. It does not regenerate representations, freeze classifiers, or score test cells. The default config describes the historical starter grid; use the generated tuning config for a new refined run.

Each invocation writes a fresh `results/logit/` directory with config, log, source/input/output hashes, search metrics, and per-class analysis. Use `--output` to select a new directory and `--threads` to change the default of one. Failed/partial runs are preserved.

Dictionary implementations and grid configs live in the [dictionary-learning package](dictionary_learning.md). For the complete reusable workflow, use the [root README](../../README.md) and `gene_complexity` package API.
