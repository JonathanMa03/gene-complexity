# Baron Human Pancreas Preprocessing

## Data source

Baron et al. human pancreas single-cell RNA-seq dataset
(GSE84133), accessed through scRNAseq::BaronPancreasData(which = "human").

Dataset:
- 8,569 cells
- 20,125 genes
- 4 donors
- 14 annotated cell types

## Raw count validation

The counts assay contains nonnegative, finite, integer-valued counts.
Minimum: 0; maximum: 4,318.
All 20,125 gene symbols are unique.

The shared raw dataset is not normalized or gene-filtered.

## Preprocessing scripts

1. export_baron.R
   Installs missing R dependencies, downloads/caches the original RDS on first use, and exports the sparse count matrix,
   gene symbols, cell IDs, donor IDs, and cell-type labels.

2. build_baron_h5ad.py
   Converts the exported matrix into AnnData (cells x genes).
   Preserves integer counts in X and layers["counts"].
   Writes data/processed/baron_raw.h5ad and validates the result.

3. map_baron_panel.py
   Matches Baron gene symbols to Sijia's fixed scGPT gene panel.
   Preserves the supplied Ensembl identifiers and gene symbols.
   Writes data/processed/baron_scgpt_gene_mapping.csv.

## scGPT panel compatibility

Total panel genes: 2,000
Matched genes: 1,879
Missing genes: 121
Gene-symbol coverage: 93.95%

Mapping source:
code/scgpt/pretraining_handoff/assets/sctab_hvg_2000.tsv

The supplied Ensembl-to-symbol pairs have not been independently
validated against an external annotation database.

Missing panel genes are not zero-filled in the shared raw dataset.
Model-specific handling remains to be confirmed.

## Reproducing the preprocessing

Required R packages: scRNAseq, SingleCellExperiment, Matrix.
Required Python packages: anndata, scipy, pandas, numpy.

Run these commands from the repository root. `export_baron.R` installs missing
R dependencies without updating existing packages, creates the data directories,
and downloads `BaronPancreasData(which = "human")` if
`data/raw/baron_human_counts.rds` is absent. Subsequent runs reuse that cache.
First-time setup requires internet access and a writable R package library.
On macOS, the script detects an existing Homebrew OpenSSL 3 installation and
temporarily exposes its libraries during package installation. This addresses
the `ld: library 'ssl' not found` error observed while building `alabaster.base`.
It does not change global R configuration or install Homebrew packages.

Rscript code/preprocessing/export_baron.R
python code/preprocessing/build_baron_h5ad.py
python code/preprocessing/map_baron_panel.py

Generated data files are excluded from Git.

## Shared splits and rare classes

The agreed primary analysis uses a seeded **70/15/15 cell-type-stratified random split** and excludes classes with **fewer than 30 cells after confirmed scGPT exclusions**. Donors may appear in multiple splits; this evaluates new cells from the same donor pool, not unseen-donor generalization. Raw counts are never modified.

Run a provisional split from the repository root:

```bash
python code/preprocessing/create_splits.py
```

This reads `data/processed/baron_raw.h5ad` and writes a draft under `data/processed/baron_splits_draft/`. Once scGPT confirms compatibility, create the final bundle:

```bash
python code/preprocessing/create_splits.py \
  --output-dir data/processed/baron_splits_final \
  --scgpt-confirmed
```

If scGPT reports excluded cells, also pass `--scgpt-exclusions path/to/exclusions.csv` (required column: `cell_id`). The flag `--scgpt-confirmed` is a human attestation, not an automated compatibility test. Set it only after the scGPT owner confirms the supplied list, including an empty list. Unknown/duplicate exclusion IDs are rejected.

Outputs:

- `baron_split_manifest.csv`: `cell_id, cell_type, donor, split` for retained cells.
- `baron_exclusions.csv`: excluded metadata and reasons (scGPT exclusions take precedence).
- `baron_split_counts.csv`: retained class counts by split.
- `baron_class_counts.csv`: counts before and after scGPT exclusions and rare-class filtering.
- `split_settings.json`: draft/final status, seed, policy, source/manifest/script hashes, versions, and totals.

Default seed is 42. Allocation is performed independently within each class: take the floor of 70/15/15 percent, then allocate remaining cells to the largest fractional remainders, breaking ties in train/validation/test order. Fractions are therefore approximate for small classes (30 cells become 21/5/4). Input cells and class names are sorted before seeded shuffling, making results independent of input row order. All retained classes must occur in each split; retained and excluded IDs must partition the input exactly. Existing output directories are never overwritten; supply a fresh directory when regenerating.

The current count table implies exclusions of T cells, Schwann, epsilon, and mast (63 cells total) before any additional scGPT exclusions. Recompute the policy on the actual input rather than hardcoding these names. Share one finalized manifest with every model branch. Its generated files remain outside Git under the existing `data/processed/` ignore rule; share the bundle or regenerate with the recorded inputs/settings. Any feature selection or fitted scaling must use training cells only.

Run the synthetic metadata checks with:

```bash
python -m unittest discover -s code/preprocessing -p 'test_*.py'
```

## Remaining decisions

- Confirm scGPT handling of unmatched panel genes.
- Independently verify Ensembl mappings if required.
