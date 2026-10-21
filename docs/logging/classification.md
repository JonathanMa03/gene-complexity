# Logistic classification

These modules support the runnable entry points in `gene_complexity.api`:

| Module | Responsibility |
|---|---|
| `inputs.py` | Load verified expression, dictionary, and scGPT arrays; align by cell ID and enforce the final shared cohort and provenance. |
| `search.py` | Tune C separately for every configured representation on training/validation cells. |
| `analysis.py` | Refit selected validation settings and export per-class metrics/confusion matrices, checking convergence and score agreement. |

Use `python -m gene_complexity run-validation --config gene_complexity/classification/configs/validation_v1.json` for the integrated validation search, or `python -m gene_complexity tune-models` to generate a new dictionary grid and tune all methods together. The number of candidates follows the supplied config; it is not fixed at six feature sets.

For saved classifiers and explicit frozen evaluation, use `gene-complexity train-logistic`, `gene-complexity evaluate-validation`, `gene-complexity run-test`, and `gene-complexity summarize-results` as documented in the [root README](../../README.md).

Each representation feeds its own L2-regularized multinomial logistic classifier. Expression uses `X`; dictionary and scGPT archives use `vectors`. All joins use unique cell IDs and the shared manifest. No representation concatenation is performed. The original cell-split workflow applies no extra classifier-side scaling; the separate donor workflow explicitly controls scaling within folds.

The C search uses coarse `[0.01, 0.1, 1, 10, 100, 1000]`, followed by 21 log-spaced values within a factor of ten of the coarse winner, excluding already tested values. Only converged fits qualify; ties prefer smaller C. Selection uses validation Macro-F1, never final test scores.

Generated datasets/models/results remain local and ignored by Git. Use fresh output directories and retain their provenance and referenced artifacts.
