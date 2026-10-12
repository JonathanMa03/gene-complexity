# Baron input compatibility

This validates the teammate-supplied `baron_raw(1).h5ad` and
`baron_scgpt_gene_mapping(1).csv` against the fixed scGPT panel. The mapping
generation logic was inspected on `baron-preprocessing` commit
`d276ea1452bf3b2d7e4039ee239f749b200de0e5`, at
`code/preprocessing/map_baron_panel.py`. No changes to that branch are needed.

The input has 8,569 cells and 20,125 unique gene symbols. Both `X` and `counts`
contain identical finite, nonnegative, integer-valued expression. Mapping
presence flags and zero-based gene indices are checked against the actual data;
all 2,000 mapping rows must match the fixed panel IDs, symbols and order.

The prepared file keeps the original gene-symbol index, expression, cell IDs,
row order, donor and cell-type metadata. It adds
`var['scgpt_panel_ensembl_id']`. Unmapped non-panel genes receive explicit
`__UNMAPPED_SYMBOL__:` sentinels, which are not real Ensembl IDs and are ignored
by the extractor. Missing panel genes are zero-filled only inside the extractor.

## Reproduce

Use the parent handoff's dependencies. This input uses nullable-string metadata;
the preparation script enables its AnnData >=0.11 serialization rather than
converting or discarding the metadata. Run from `code/scgpt/pretraining_handoff`:

```bash
python baron/prepare_baron.py \
  --input '/path/to/baron_raw(1).h5ad' \
  --mapping '/path/to/baron_scgpt_gene_mapping(1).csv' \
  --output-dir /path/to/baron_compatibility

python extract_embeddings.py \
  --checkpoint-dir artifacts/weights/sctab_seed0_n2220_scgpt \
  --input /path/to/baron_compatibility/baron_scgpt_mapped.h5ad \
  --counts-layer counts \
  --gene-id-column scgpt_panel_ensembl_id \
  --allow-missing-panel --limit 8 --device cpu \
  --output /path/to/baron_compatibility/baron_smoke8_embeddings.npz
```

Use a new output directory/path on every run. The script hashes the inputs,
checks all 8,569 cells for zero expression across the matched panel, and writes
`preparation_report.json`, `missing_panel_genes.csv`, and
`zero_panel_cell_ids.csv`. If zero-panel cells exist, it reports their IDs and
stops without silently dropping them. Otherwise it saves the mapped file,
reloads it, verifies expression and metadata, and rechecks source hashes.

## Verified results (2026-10-12)

- Panel coverage: **1,879 / 2,000 (93.95%)**; 121 missing genes.
- Zero-expression cells across matched panel: **0** (all 8,569 cells checked).
- Duplicate gene symbols / cell IDs: **0**.
- Preparation used the unchanged handoff `panel_counts(..., allow_missing=True)`.
- CPU extraction of the first 8 cells: **8 × 128**, all finite float32 values,
  L2 norm range 0.99999994–1.0; output cell IDs match the input order.
- The small embedding result and model/output hashes are recorded in
  `validation/smoke_test.json` after actual extraction.

`validation/preparation_report.json` contains input and output checksums.
Large H5AD files, embeddings and checkpoints remain outside Git.

This is an execution compatibility check, not downstream classification or
scaling-law evaluation. Integer values do not by themselves prove upstream raw
processing provenance. The supplied file has no provenance metadata in `uns`;
study/donor/cell overlap with pretraining has not been audited here. No training,
test split, model weights, or original input files are changed.
