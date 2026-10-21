# Integration and experimental workflow

This note summarizes the integrated workflow introduced in `82e0503`
(2026-10-19) and reorganized in `1f12cd9` (2026-10-21). For installation and
download locations, use the [root README](../../README.md). For chronology and
contributor attribution, use the [changelog](CHANGELOG.md).

## Shared inputs and representations

The [preprocessing handoff](preprocessing.md) defines raw Baron counts, the
shared cohort, and final cell splits. The [classical pipeline](pipeline.md)
fits normalization-related gene selection and scaling on training cells.
[Dictionary learning](dictionary_learning.md) fits atoms on training features
and freezes them for held-out encoding. The [scGPT handoff](scgpt_handoff.md)
and [Baron extraction record](scgpt_baron.md) document the supplied frozen
embeddings. Every classifier uses matching cell IDs and labels.

## Selection and evaluation

[Logistic classification](classification.md) tunes regularization separately
for each representation. The integrated search fits configurable dictionary
candidates and selects settings using validation metrics. Explicit freezing
records the selected model and supporting hashes before final test evaluation.
Test scoring uses saved models and does not refit them.

The original stratified cell split shares donors across partitions. A separate,
retrospective donor study uses outer held-out donors and inner donor-based
selection, refitting classical transforms and dictionaries within training
folds. It compares multiple dictionary initializations and controlled
classifier-side scaling across representations. Atom matching and optimization
diagnostics supplement predictive metrics; these do not prove global optimality
or biological recovery. The scGPT representation remains frozen.

The donor study is a robustness analysis added after the original test had been
examined, not a new prospectively untouched study. It also does not establish
independence from scGPT pretraining data. These limits and the mathematical
formulation are documented in the [report](../notes/main.tex) and its appendix.

## Reusable execution and cleanup

`workflow.ipynb` calls `gene_complexity` functions directly. Its default smoke
run regenerates features, fits the three classifier approaches, and exports
validation metrics and figures. Full tuning, freezing, test evaluation, and
donor experiments require explicit opt-in. Run records capture settings,
source/input/output hashes, and stage dependencies. Partial or changed runs
require a new output name; unchanged completed stages are verified before reuse.

The downloadable `external/` bundle supplies final splits, the scGPT handoff,
and saved CSV/PNG sources for the original report snapshots. It does not replace
fresh training with saved classifiers. The notebook was verified in a minimal
checkout without `code/` or previous results. Historical frozen artifacts and
provenance were preserved separately during the package migration.

Component READMEs now live in `docs/logging/`. The theory/design and rigor
material, experimental notes, and results were consolidated into the sectioned
report under `docs/notes/`. Synthetic demos, one-time inspection/testing
utilities, redundant wrappers, and superseded standalone notes were removed;
the reusable methods remain in the package. The root README intentionally
focuses on the TA's download-and-run workflow.
