# Baron input compatibility

Source location: `external/scgpt/pretraining_handoff/baron`. Commands for the scGPT handoff are run from that source directory unless stated otherwise.

This validates the teammate-supplied `baron_raw(1).h5ad` and
`baron_scgpt_gene_mapping(1).csv` against the fixed scGPT panel. The mapping
generation logic was inspected on `baron-preprocessing` commit
`d276ea1452bf3b2d7e4039ee239f749b200de0e5`, at
`gene-complexity map-panel`. No changes to that branch are needed.

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
converting or discarding the metadata. Run from `external/scgpt/pretraining_handoff`:

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
- The development smoke report is archived. The reusable full-extraction record is
  `validation/full_extraction_metadata.json`.

`validation/preparation_report.json` contains input and output checksums.
Large H5AD files, embeddings and checkpoints remain outside Git.

This is an execution compatibility check, not downstream classification or
scaling-law evaluation. Integer values do not by themselves prove upstream raw
processing provenance. The supplied file has no provenance metadata in `uns`;
study/donor/cell overlap with pretraining has not been audited here. No training,
test split, model weights, or original input files are changed.

## Full comparison extraction

The prespecified comparison checkpoint is **sctab_seed0_n1600000_scgpt**:
1.6 million pretraining cells, sampling seed 0, initialization seed 20260821,
and final epoch 10. Its SHA256 is
`33d30d45051f858ad358a883ae9c39f9fbacc6e14e6fa037d3684320f12aa356`.
We choose the largest completed cell-number condition with seed 0 fixed before
observing Baron classification scores. This represents the largest available
pretraining scale; it does not establish that the checkpoint is best-performing.
The model has 1,166,466 parameters and is a project-trained small scGPT, not the
large official scGPT model. Full rationale is in `checkpoint_selection.json`.

After the preparation command above, retrieve this checkpoint and extract all
cells. Run from `external/scgpt/pretraining_handoff`; use new output paths:

```bash
python fetch_artifacts.py --checkpoint sctab_seed0_n1600000_scgpt
# Without lab access, download and unpack the seed0 weight archive instead;
# see the parent README and artifact_manifest.json.

python extract_embeddings.py \
  --checkpoint-dir artifacts/weights/sctab_seed0_n1600000_scgpt \
  --input /path/to/baron_compatibility/baron_scgpt_mapped.h5ad \
  --counts-layer counts --gene-id-column scgpt_panel_ensembl_id \
  --allow-missing-panel --batch-size 8 --seed 2026092500 --device cpu \
  --output /path/to/delivery/baron_scgpt_full.npz

python baron/finalize_full_extraction.py \
  --npz /path/to/delivery/baron_scgpt_full.npz \
  --input /path/to/baron_compatibility/baron_scgpt_mapped.h5ad \
  --raw-input '/path/to/baron_raw(1).h5ad' \
  --mapping '/path/to/baron_scgpt_gene_mapping(1).csv' \
  --preparation-report /path/to/baron_compatibility/preparation_report.json \
  --output /path/to/delivery/baron_scgpt_full.metadata.json
```

There is deliberately **no --limit** in the full extraction command. All model
parameters stay frozen. Use the validated CPU/float32, batch size 8 and seed for
reproduction; changing device, precision or batching can change numerical output.
The extractor hashes the complete input before and after inference. The finalizer
checks the raw, mapping, prepared-input and output hashes, all IDs, shape,
finiteness and normalization. It creates metadata linking all these artifacts.

The NPZ stores `vectors` (float32, 8569 x 128) and `cell_ids` (Unicode, length
8569). Load it safely without pickling and join downstream labels/splits by ID:

```python
import numpy as np
with np.load("baron_scgpt_full.npz", allow_pickle=False) as z:
    vectors, cell_ids = z["vectors"], z["cell_ids"]
assert vectors.shape == (8569, 128)
assert len(cell_ids) == 8569
```

The metadata records checkpoint ID/hash, prepared and original input hashes,
mapping/panel hashes, coverage, seed, batch size, threads, device, precision,
software/source versions and output checksum. The preparation JSON and checkpoint
selection JSON accompany the full extraction metadata. Full feature extraction
uses no target labels or downstream fitted statistics and does not create or
change the existing classification splits. Pretraining overlap remains unaudited.

### Download the completed full extraction

[Drive delivery folder](https://drive.google.com/drive/folders/1Llp26DZmhvBfMjf0Q-JVy_RR1oe9LIhI)
contains `baron_scgpt_full.npz`, `baron_scgpt_full.metadata.json`,
`preparation_report.json` and `checkpoint_selection.json`. Sizes, SHA256 and
individual file links are in `full_artifacts.json`. Verify downloaded hashes
before loading. The actual full extraction metadata is also checked into
`validation/full_extraction_metadata.json`. All 8,569 IDs were validated in
original order, with zero exclusions. The first 8 vectors exactly match the
separate smoke extraction using the chosen checkpoint and same CPU settings.
