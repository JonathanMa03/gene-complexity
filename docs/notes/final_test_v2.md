# Final held-out evaluation — 19 October 2026

The three previously frozen classifiers were evaluated on the same **1,275 test cells**, using the final shared manifest. No preprocessing, dictionary, classifier or checkpoint setting was changed; no model was refitted. The frozen model files, selection files and freeze manifest have unchanged SHA-256 hashes.

| Representation | Features | Logistic C | Macro-F1 | Accuracy | Balanced accuracy | Log loss | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Processed expression | 2,000 | 0.3162277660168379 | 0.993913 | 0.994510 | 0.993687 | 0.019454 | 7 |
| Dictionary, 64 atoms, alpha 2 | 64 | 0.0630957344480193 | 0.980962 | 0.989020 | 0.977933 | 0.036989 | 14 |
| Frozen scGPT | 128 | 316.22776601683796 | 0.962358 | 0.968627 | 0.955637 | 0.096535 | 40 |

The expression baseline has the strongest observed test performance on all four metrics. Dictionary codes compress the classifier input from 2,000 to 64 dimensions while preserving high classification accuracy, but do not outperform expression on this test split. The supplied small custom scGPT representation also does not improve this evaluation over the expression baseline. These results do not establish a general comparison against all dictionary methods or the original large published scGPT model.

The dictionary's tiny validation Macro-F1 lead did not persist on test. Do not retune based on this outcome. These are single-split observations without statistical significance testing. Donors overlap across splits, rare classes were excluded, and scGPT pretraining overlap remains unaudited. Preserve those qualifications in the main report.

## Verification and artifacts

- Each prediction archive contains exactly the manifest's 1,275 unique test cell IDs with matching labels.
- Probability arrays are finite and sum to one within their floating-point precision. Macro-F1 and log loss were independently recomputed from saved predictions/probabilities.
- Dictionary test codes use the saved atoms and alpha with Lasso coordinate descent; no dictionary update occurs.
- The original frozen manifest's `test_evaluated: false` describes its pre-test state and remains unchanged for provenance. The new evaluation report records this completed test run separately.

Outputs are under `results/workflow/frozen_models_v2/test/`:

- `evaluation_report.json`: run status, timing, original freeze-manifest hash, all three metrics records and output hashes.
- `{baseline,dictionary,scgpt}/`: `metrics.json`, `predictions.npz`, `per_class.csv`, `confusion_matrix.csv`.
- `summary/summary.csv`: three-method comparison.
- `summary/macro_f1.png` and `confusion_1.png` through `confusion_3.png`: test plots (baseline, dictionary, scGPT order).
- Per-method logs.

The pre-test backup `backups/frozen_models_v2.zip` remains unchanged. Test outputs are preserved separately in `backups/final_test_v2.zip` with a SHA-256 checksum. Copy both archives off-device. The next work is reporting and interpretation, not additional tuning on this test set.
