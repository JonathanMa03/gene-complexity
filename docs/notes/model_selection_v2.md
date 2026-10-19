# Model-selection review and freeze — 19 October 2026

The single planned v2 refinement is complete. Select the 64-atom dictionary with alpha 2 using the agreed primary criterion, validation Macro-F1. Freeze all three representation-specific classifiers below. No test evaluation was performed; no additional grid expansion is planned before test evaluation.

| Representation | Logistic C | Validation Macro-F1 | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|---:|---:|
| Processed expression | 0.3162277660168379 | 0.984365 | 0.990603 | 0.984698 | 0.034559 |
| Dictionary: 64 atoms, alpha 2 | 0.0630957344480193 | 0.984580 | 0.989037 | 0.982884 | 0.043370 |
| Frozen scGPT | 316.22776601683796 | 0.965181 | 0.973375 | 0.961333 | 0.092422 |

The dictionary's Macro-F1 advantage over expression is only 0.000216. Expression has fewer errors (12 versus 14) and better balanced accuracy and log loss. These validation-selected scores do not establish superiority, statistical equivalence, or independent generalization performance. scGPT makes 34 errors on the same 1,277 validation cells.

## Per-class review

The dictionary improves F1 over expression for acinar (0.9895 versus 0.9825) and activated stellate (0.9655 versus 0.9545), but is lower for alpha, beta, delta and ductal. Delta-to-beta errors are four for dictionary, one for expression, and eleven for scGPT. These tradeoffs explain why Macro-F1 and total errors favor different representations.

All ten retained classes appear in validation. Macrophage has only eight validation cells, quiescent stellate 26, endothelial and gamma 38 each, and activated stellate 43. Perfect macrophage F1 for all three classifiers is weak evidence because of the small support. Donors overlap across splits; this is not unseen-donor evaluation. The rare-class exclusion policy remains unchanged.

## Dictionary weight and activity review

The selected dictionary has 64 unit-norm atoms and 2,000 input genes. There are no unused atoms in training or validation and no recorded solver warnings. Training coefficients are 15.30% active; validation coefficients are 17.02% active (10.89 active atoms per cell on average, threshold absolute coefficient > 1e-6). The maximum absolute cosine between distinct atoms is 0.6881: no duplicate or sign-reversed duplicate atoms were found, but this is not a proof of stability or uniqueness.

Examples below are empirical associations from this fitted dictionary, **not pathway-enrichment results or biological validation**. Atom numbers are one-based and apply only to this run. The activity values are cell-type mean codes centered/scaled using the overall training-code mean and standard deviation.

| Atom | Largest positive-weight genes (first six) | Cell type | Train activity | Validation activity |
|---|---|---|---:|---:|
| 2 | CUZD1, CTRC, PNLIPRP1, CLPS, PNLIPRP2, GP2 | acinar | 2.442 | 2.265 |
| 5 | COL1A2, COL3A1, COL1A1, BGN, COL6A3, COL6A2 | activated stellate | 4.524 | 4.480 |
| 11 | PLVAP, STC1, ENG, HSPG2, EGFL7, ETS1 | endothelial | 5.522 | 5.425 |

The full exports contain both positive and negative gene weights and signed cell activities. Atom sign and ordering are arbitrary. The eight heatmap atoms were selected by maximum absolute cell-type mean contrast on **training cells only**, then displayed for both train and validation using the same normalization and color scale. The three illustrative rows above are a descriptive subset, not additional criteria used to choose hyperparameters. No seed-sensitivity or formal enrichment analysis has been performed, so do not claim reproducible biological programs yet.

## Frozen artifacts and evaluation policy

- Tuning evidence: `results/workflow/model_selection_v2/` (192 converged classifier fits).
- Gene/activity review: `results/workflow/model_selection_review_v2/dictionary/`.
- Saved classifiers: `results/workflow/frozen_models_v2/models/{baseline,dictionary,scgpt}/`.
- Validation evaluations and frozen `selection.json` files: `results/workflow/frozen_models_v2/validation/{baseline,dictionary,scgpt}/`.
- Combined record: `results/workflow/frozen_models_v2/freeze_manifest.json`.
- Validation summary: `results/workflow/frozen_models_v2/summary/`.

Each saved classifier was fitted using training cells only with one numerical thread, lbfgs, max_iter 5,000 and tol 1e-4. All four validation metrics from the saved coefficients were checked against the tuning run within absolute tolerance 1e-8 before creating any frozen selections. Preprocessing, the dictionary, and classifier training remain train-only; there is no train+validation refit.

Preserve **both** `frozen_models_v2/` and its referenced source artifacts: the v2 dictionary grid, classical features and preprocessing provenance, final splits, and scGPT embeddings/metadata. These local directories are ignored by Git and must be backed up separately. The saved models include source paths and hashes; deleting or replacing referenced files will invalidate evaluation checks. The README's clean smoke-run procedure does not preserve this experimental record automatically.

Next step is one final test evaluation for each frozen model using `scripts/run_test.py`, followed by a test-only summary. Do not use those results to revise preprocessing, dictionary settings, classifier C, or the checkpoint choice. Final test scores belong in the report's main results table; the figures above currently describe validation diagnostics only.

## Preservation and tuning decision

The frozen models and their evaluation dependencies are preserved in `backups/frozen_models_v2.zip`, with a SHA-256 inventory and restore instructions alongside it. This includes the original frozen selection records, all referenced training/validation artifacts, the unchanged test feature archive (copied and hashed only, not scored), v2 selection/review evidence, the four scGPT handoff files, and evaluation source code. Original working artifacts remain in place. The raw Baron `.h5ad` is not duplicated in this model backup; retain it separately as described in the README.

No further tuning is warranted by the completed review. The agreed one-round refinement is complete, the selected fits converged, all atoms are active, and saved-model validation scores reproduce the search. Further searching on the same validation set would add selection bias without a specific identified problem. Leave the frozen models unchanged and proceed to final evaluation as a separate step.
