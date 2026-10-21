Dates below use Git author dates for the commits introducing each change; merge
commits are not repeated. Commit IDs identify the historical implementation.
Linked notes describe the current layout and may include later updates.

Initials: **SZ** — Sijia Zhu (scGPT); **YC** — Yunwei Chai (logistic
classification); **JG** — Jiatong Gao (preprocessing); **JM** — Jonathan Ma
(integration and dictionary learning). Entries credit the recorded contributor,
including changes outside their primary role.

# Added

- JM 2026-10-21: Add the installable `gene_complexity` package and root
  `workflow.ipynb`, with individually runnable stages, input/provenance checks,
  saved run records, metrics, and figures. See [workflow notes](experimental_workflow.md).
  (`1f12cd9`)

- JM 2026-10-21: Assemble the external download bundle with the scGPT handoff,
  final shared splits, file-hash inventory, and original report snapshot inputs.
  Verify notebook execution in a clean checkout without previous results.
  See [TA setup](../../README.md) and [scGPT handoff](scgpt_handoff.md). (`1f12cd9`)

- JM 2026-10-19: Add integrated validation, configurable dictionary/C selection,
  saved classifiers, explicit validation freezing and test evaluation, and
  summary/interpretation outputs. Add donor-held-out evaluation with dictionary
  initialization and representation-scaling controls. See
  [workflow notes](experimental_workflow.md), [pipeline](pipeline.md), and
  [dictionary learning](dictionary_learning.md). (`82e0503`)

- YC 2026-10-19: Add validation analysis, per-class metrics and confusion matrices,
  and revise classifier documentation. See [classification](classification.md).
  (`fad57b6`)

- YC 2026-10-15: Add logistic baselines and coarse-to-fine regularization search
  for expression, four dictionary candidates, and scGPT. See
  [classification](classification.md). (`2e6c17f`, `c0bae7f`)

- JM 2026-10-13: Add training-fitted classical normalization, gene selection and
  scaling, the real-data dictionary pilot, configurable candidate grid, and
  classifier integration handoff. See [pipeline](pipeline.md) and
  [dictionary learning](dictionary_learning.md). (`94abbd0`, `0024bee`, `f196262`)

- SZ 2026-10-13: Deliver frozen Baron features for all 8,569 cells using the
  prespecified 1.6M-cell seed0 checkpoint, with input hashes, ID/shape checks,
  checkpoint rationale, and extraction metadata. Verify finite 8,569 x 128
  vectors without changing weights or excluding cells. See
  [Baron extraction](scgpt_baron.md). (`3b8664e`)

- SZ 2026-10-12: Add Baron/scGPT Ensembl mapping preparation and validation:
  1,879/2,000 panel genes, no empty-panel cells, and finite 8 x 128 CPU smoke
  embeddings with preserved IDs. See [Baron extraction](scgpt_baron.md).
  (`7bd8095`)

- JM 2026-10-11: Add shared stratified 70/15/15 splitting, the fewer-than-30
  rare-class rule, exclusion records, and split validation. Add first-time R
  dependency setup to the exporter. See [preprocessing](preprocessing.md).
  (`d276ea1`)

- JG 2026-10-08: Add raw Baron download/export, AnnData conversion, gene-panel
  mapping, and the cell-type by donor count table. Preserve raw integer counts,
  labels, and donor IDs. See [preprocessing](preprocessing.md).
  (`b126236`, `ef14f37`)

- JM 2026-10-08: Add the initial dictionary theory/design note and synthetic
  Python/notebook demonstration, plus preprocessing handoff guidance. These were
  later consolidated into the reusable implementation and report. See
  [dictionary learning](dictionary_learning.md). (`f97fe0c`)

- SZ 2026-10-06: Add the verified small-scGPT pretraining handoff: 51 checkpoint
  configurations, fixed panel/vocabulary, model and preprocessing sources,
  sampling provenance, artifact manifests, and frozen extraction. See
  [handoff](scgpt_handoff.md), [888k extension](scgpt_extension_888k.md), and
  [1.6M extension](scgpt_extension_1600k.md). (`c63ec75`)

- JM 2026-10-01: Initialize the repository, license, source/data/result layout,
  course/team README, abstract/proposal/reference documents, dependencies, environment
  notebook, Git guidance, and CODEOWNERS. See [project overview](../../README.md).
  (`a47bb01`, `695360c`, `d66ecae`)

# Changed

- JM 2026-10-21: Move reusable implementations from `code/` and `scripts/` into
  `gene_complexity`; make the notebook call Python APIs directly. Move the scGPT
  handoff into the ignored `external/` bundle and update active resource paths;
  preserve historical provenance and frozen results. See
  [workflow notes](experimental_workflow.md) and [handoff](scgpt_handoff.md).
  (`1f12cd9`)

- JM 2026-10-21: Move eight component READMEs into `docs/logging/`, repair links,
  and shorten the root README to the TA reproduction workflow. Consolidate
  theory, methods, results, and rigor notes into the sectioned LaTeX report with
  supplementary theory, tables, figures, and references. See
  [workflow notes](experimental_workflow.md) and [TA setup](../../README.md).
  (`1f12cd9`)

- JM 2026-10-19: Strengthen integrated classifier input alignment, cohort/hash
  checks, convergence handling, and reproducible run records; refine the
  dictionary grid and document model selection and frozen test evaluation.
  Update the report draft. See [classification](classification.md) and
  [workflow notes](experimental_workflow.md). (`82e0503`, `db33371`)

- JM 2026-10-12: Finalize/document shared Baron splits after scGPT compatibility
  confirmation, tighten raw integer-count validation, and reconcile generated
  data ignore rules during branch integration. See
  [preprocessing](preprocessing.md). (`2828ba7`, `fa7091e`, `30f1616`)

- SZ 2026-10-12: Use portable LF line endings in zero-cell reports. See
  [Baron extraction](scgpt_baron.md). (`ff020fd`)

# Deleted

- JM 2026-10-21: Remove the synthetic dictionary demos, standalone environment
  and pilot notebooks, one-time inspection scripts, redundant baseline wrappers,
  and tracked `test_*` utilities. Retain the reusable functionality in the package
  and root notebook. See [dictionary learning](dictionary_learning.md),
  [classification](classification.md), and [workflow notes](experimental_workflow.md).
  (`1f12cd9`)

- JM 2026-10-21: Remove superseded standalone theory/handoff/experiment notes,
  the unused report style file, placeholder files, and the compiled `jm.pdf` from
  the tracked source tree. Consolidated documentation and report sources remain;
  component README and scGPT handoff moves are recorded under Changed above.
  See [workflow notes](experimental_workflow.md). (`1f12cd9`)

- YC 2026-10-19: Remove the duplicate logistic README after updating the primary
  classifier documentation. See [classification](classification.md). (`d47aa8a`)
