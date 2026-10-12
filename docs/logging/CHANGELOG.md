# Added

- SZ 2026-10-12: Add Baron/scGPT Ensembl mapping preparation and validation.
  Verified 1,879/2,000 panel genes, zero empty-panel cells across all 8,569 cells,
  and finite 8 x 128 CPU embeddings with preserved cell IDs. Original input
  expression/metadata are retained; no training or test splits changed.

- SZ 2026-10-06: Add a verified small-scGPT pretraining handoff: 51 checkpoint
  configurations, model/tokenizer/preprocessing sources, fixed 2,000-gene panel,
  data sampling provenance, external artifact manifests, and portable frozen
  embedding extraction. CPU smoke validation on eight real cells produced finite
  8 × 128 L2-normalized embeddings. No pretraining or downstream jobs were changed.

# Changed

# Deleted
