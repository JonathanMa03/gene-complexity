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
   Reads the original RDS and exports the sparse count matrix,
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

First, download BaronPancreasData(which = "human") in R and save
the SingleCellExperiment object to:
data/raw/baron_human_counts.rds

Then run these commands from the repository root:

Rscript code/preprocessing/export_baron.R
python code/preprocessing/build_baron_h5ad.py
python code/preprocessing/map_baron_panel.py

Generated data files are excluded from Git.

## Remaining decisions

- Agree on shared train/validation/test splits.
- Decide how to handle extremely rare cell types.
- Confirm scGPT handling of unmatched panel genes.
- Independently verify Ensembl mappings if required.
