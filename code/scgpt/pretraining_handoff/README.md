# Small scGPT pretraining handoff

This is a handoff of **completed server checkpoints**, not a new training run.
It contains the actual model implementation, frozen panel/vocabulary, checkpoint
configurations, training records, data provenance, and portable loading/extraction
commands. The downstream cell-to-text adapters and language models are outside this
handoff. The repository's existing classification project remains unchanged.

## Model and checkpoint scope

- Fixed small scGPT architecture: **1,166,466 parameters**, 4 Transformer layers,
  4 attention heads, 128-dimensional CLS/token embeddings, hidden size 512.
  These are controlled scGPT-architecture models, **not the original published
  large pretrained scGPT weights**.
- Fixed 2,000-gene Ensembl panel; vocabulary contains 2,003 entries including
  `<pad>`, `<cls>`, and `<eoc>`. Do not substitute another gene panel/vocabulary.
- Nonzero raw expression, per-cell 51-bin transformation, 1,200-token cap,
  40% masked values. Equal-weight masked gene-expression and MVC/GEPC MSE losses.
- 10 epochs, batch 64, AdamW lr 0.0001, weight decay 0.01, warmup 5%, cosine decay,
  gradient clipping 1. Historical training used BF16. Fixed initialization seed
  20260821; sampling seeds 0/1/2 are **cell sampling**, not three model initialization
  seeds. Different n with fixed epochs implies different optimizer-step counts.
- **33 cell-number checkpoints**: 11 sizes × 3 seeds: 2,220; 4,440; 8,880; 17,760;
  35,520; 71,040; 142,080; 222,000; 444,000; 888,000; 1,600,000.
- **18 depth checkpoints**: see the exact retention rates and used-cell counts in
  `checkpoint_manifest.tsv`. Some depth conditions remove zero-panel cells.
- All 51 weights were checked against config SHA256 and `_SUCCESS`. Final-epoch
  weights are used; there was no validation-based pretraining checkpoint selection.

## Files

| Path | Purpose |
|---|---|
| `vendor/scgpt/` | Actual server model, tokenizer, collator, binning and loss code |
| `vendor/LICENSE.scGPT` | Upstream MIT notice |
| `assets/` | Fixed ordered 2,000-gene panel and common vocabulary |
| `configs/<model_id>/config.json` | Exact configuration for each checkpoint |
| `training_records/` | Small per-epoch training histories, where present |
| `checkpoint_manifest.json`, `.tsv` | n, seeds, epochs, steps, original paths, sizes, hashes |
| `artifact_manifest.json` | External archive locations, sizes, SHA256 |
| `provenance/` | Sampling recipe, source versions, extension records, source hashes |
| `original_scripts/` | Unmodified server training/preprocessing/sampling scripts |
| `model_utils.py`, `extract_embeddings.py` | Portable frozen-model inference |
| `fetch_artifacts.py` | Retrieve individual weights or complete archives with hash checks |
| `audit_overlap.py` | Check release-specific cell IDs and study/donor metadata |
| `train_scgpt.py` | Portable path wrapper around the archived CUDA training algorithm |
| `requirements.txt`, `runtime_environment.json` | Tested versions and live environment inventory |
| `validation/` | Smoke-test results and handoff audit; no raw cells or embeddings |

The numerical vendor files are byte-for-byte server copies, listed in
`provenance/source_files.json`. Only package import initializers are simplified to
avoid loading unrelated scGPT tasks and optional frameworks. The server scGPT Git
revision is `cebd6fae655b9c585a4807daa3ac31bb764f06b4` (version label 0.2.5).
Compatibility helpers used on the server are included rather than assuming a
fresh public pip installation is identical.

## Install

From the repository root, create a **separate environment**; this handoff does not
change the repository-wide `requirements.txt`:

```bash
python3.10 -m venv .venv-scgpt
source .venv-scgpt/bin/activate
python -m pip install --upgrade pip
# CPU inference; choose an appropriate CUDA wheel instead for GPU use.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/scgpt/pretraining_handoff/requirements.txt
cd code/scgpt/pretraining_handoff
```

Python 3.10.13/Linux and the versions in `runtime_environment.json` were used for
the recorded smoke test. A fresh installation and macOS/Windows have **not** been
validated. The inventory is a live handoff-time capture, not a historical lock of
every package present during every past training run. The original data-building
environment lock is retained separately in `provenance/data_environment.lock.txt`.

## Obtain weights and sampling records

Weights and large cell-ID records are **outside Git**. The group-readable durable
handoff is on the lab filesystem:

```text
/n/holylabs/rongma_lab/Lab/sijiazhu/scinfodesign/code/handoffs/scgpt_pretraining_20261006/artifacts/
```

It contains individual `weights/<model_id>/{model.pt,config.json,vocab.json}` plus:

- `scgpt_weights.tar.gz`: all 51 selected checkpoints, configs, vocabularies and
  completion hashes; approximately 211 MiB, without optimizer/resume caches.
- `scgpt_sampling.tar.gz`: three ordered maximum 1.6M-cell sampling tables;
  approximately 57 MiB, without expression matrices.

Exact byte sizes and SHA256 are in `artifact_manifest.json`. Never treat a matching
filename alone as proof of identity. On a server with access to `/n/holylabs`:

```bash
python fetch_artifacts.py --checkpoint sctab_seed0_n2220_scgpt
# Optional: all weights and/or all sampling tables
python fetch_artifacts.py --archive weights
python fetch_artifacts.py --archive sampling
tar -xzf artifacts/scgpt_weights.tar.gz -C artifacts
tar -xzf artifacts/scgpt_sampling.tar.gz -C artifacts
```

From another computer with **your own authorized FASRC account**:

```bash
python fetch_artifacts.py --checkpoint sctab_seed0_n2220_scgpt \
  --ssh YOUR_FASRC_USER@boslogin06.rc.fas.harvard.edu
python fetch_artifacts.py --archive sampling \
  --ssh YOUR_FASRC_USER@boslogin06.rc.fas.harvard.edu
```

The login hostname above was observed on this server; use the current login entry
provided for your account if different. Credentials are never included. Directory
access remains subject to FASRC/rongma_lab permissions.

### Download without a FASRC account

[Google Drive handoff folder](https://drive.google.com/drive/folders/1YRIyI1ouMp_Qrsl9l8JiAe12Qk8yH0jk)
contains three independently extractable weight archives (17 checkpoints each)
and one sampling archive. The three seed packages together contain all 51 models.
The owner must set **General access → Anyone with the link → Viewer** in the Drive
web interface. Public access is pending and has not been verified; the connector
supports email/domain sharing only. No server account is needed after this setting.

Download all four files into `artifacts/`, then verify their contents before use:

```bash
for seed in 0 1 2; do
  python fetch_artifacts.py --verify-file "artifacts/scgpt_weights_seed${seed}.tar.gz"
  tar -xzf "artifacts/scgpt_weights_seed${seed}.tar.gz" -C artifacts
done
python fetch_artifacts.py --verify-file artifacts/scgpt_sampling_20261006.tar.gz
tar -xzf artifacts/scgpt_sampling_20261006.tar.gz -C artifacts
```

Archive hashes and individual download links are in `artifact_manifest.json`.
The browser sampling filename has a date suffix; verification checks content,
not filename. Uploaded metadata sizes match local archives; remote download hashes
have not been independently verified.

## Load a model

After fetching the example checkpoint, from this directory:

```python
from model_utils import load_model
model, vocab, config = load_model(
    "artifacts/weights/sctab_seed0_n2220_scgpt", device="cpu"
)
assert config["embsize"] == 128
assert not any(p.requires_grad for p in model.parameters())
```

The loader verifies the weight hash and uses strict state-dict loading. Original
metadata contains `TorchVersion`; the loader allows that one known class while
keeping PyTorch `weights_only=True`.

## Extract embeddings from your cells

Input must be raw nonnegative integer counts in AnnData, with gene identifiers
matching the panel (unversioned Ensembl IDs). Normalized/log values are rejected.
Use `--counts-layer counts` if raw counts are in `adata.layers['counts']`.
Use `--gene-id-column ensembl_id` if the Ensembl IDs are in that `adata.var` column.
Gene symbols are **not** silently converted to Ensembl IDs.

```bash
python extract_embeddings.py \
  --checkpoint-dir artifacts/weights/sctab_seed0_n2220_scgpt \
  --input /PATH/TO/raw_counts.h5ad \
  --output outputs/cell_embeddings.npz \
  --device cpu --batch-size 8
```

All panel genes must be present by default. If genuinely absent genes should be
zero-filled, opt in with `--allow-missing-panel`; coverage is recorded. This changes
input coverage and is not guaranteed to reproduce the server benchmark. Duplicate
matching gene columns are summed. Zero-panel cells are rejected so that cell rows
cannot silently disappear. Filter them explicitly while retaining cell IDs.

```python
import numpy as np
z = np.load("outputs/cell_embeddings.npz", allow_pickle=False)
print(z["vectors"].shape)  # (number_of_cells, 128)
print(z["cell_ids"])       # input row order, preserved
```

Extraction freezes every parameter, uses `eval()` / `inference_mode()`, disables
masking, keeps input order, extracts CLS and applies L2 normalization. It uses
float32, as in the cell-to-text feature pipeline; the historical DOGMA extractor
used BF16. Deterministic binning seed is 2026092500; record batch size/order because
random tie resolution during binning consumes RNG state. Cross-device bitwise
equivalence is not claimed. Output is the **native 128-d vector**, not the later
768-d alignment-adapter output.

## Pretraining data and downstream overlap

Source: human CELLxGENE Census LTS **2023-05-15**, using the official scTab
eligibility recipe reproduced in `original_scripts/01_build_sampling_manifests.py`.
The executed eligible universe has 22,190,622 cells, versus 22,189,056 reported in
the scTab paper; the 1,566-cell discrepancy and absence of the later duplicate
erratum correction are retained in the source record.

Sampling records retain `soma_joinid`, dataset/donor metadata and `sampling_rank`.
For each seed, smaller samples are ordered prefixes. The 888k extension appends
cells after the original 222k prefix using RNG seed 202609260+seed; the 1.6M
extension appends after 888k using 202609290+seed. Exact JSON records and verified
prefix identities are included. The panel was selected from scTab seed 0 / 222k,
not from DOGMA expression.

After retrieving/unpacking the sampling archive:

```bash
python audit_overlap.py --input /PATH/TO/test_cells.h5ad \
  --sampling-dir artifacts/sampling --n-cells 2220 --sampling-seed 0 \
  --census-version 2023-05-15 --id-column soma_joinid \
  --output outputs/overlap_seed0_n2220.json
```

Run for each checkpoint's seed/n if required. Census IDs are release-specific.
Missing IDs or ID values from another release do not establish disjointness.
The script also reports dataset/donor overlap where metadata exists; ID-disjoint
cells are not automatically study/donor-disjoint. Exact raw-expression-profile
deduplication is **not** performed by this script and needs the original raw
matrices plus a harmonized gene universe. No test-label information is used in
pretraining, but public-data source overlap still requires audit.

## Re-run the archived pretraining algorithm (optional, not run for this handoff)

Use a CUDA machine and a fresh output root; the wrapper refuses existing model
directories. It selects paths without changing the archived loss/optimizer recipe:

```bash
python train_scgpt.py --input /PATH/TO/sctab_seed0_n2220_raw.h5ad \
  --output-root outputs/retrained --model-id sctab_seed0_n2220_scgpt \
  --n-cells 2220 --sampling-seed 0
```

This wrapper was syntax/import checked; a complete retraining was not performed.
The original scripts retain their server paths as provenance, including separate
888k/1.6M download implementations. Raw expression matrices and CELLxGENE caches
are not distributed here. Current source snapshots alone cannot prove every
historical source file was unchanged at the time of each old training run.

## Validation and remaining limits

`validation/smoke_test.json` records loading `sctab_seed0_n2220_scgpt` and extracting
eight real scTab cells on CPU: **8 × 128**, finite float32 values, L2 norms near 1,
strict weight loading and zero trainable parameters.
`validation/native_compatibility.json` additionally confirms bitwise equality
with the original server model/dataset inference on the same cells. Complete
unique panel inputs preserve native CSR gene ordering, including sequence capping. These cells are an inference
smoke sample from training data, **not** a generalization benchmark.

Not supplied/confirmed: public Drive permissions (owner UI action pending);
cross-platform fresh-install validation; exhaustive historical package locks;
independently timestamped historical source revisions for every training job;
exact expression-profile overlap for arbitrary new downstream datasets. Raw
expression matrices, environments, caches, credentials and downstream chat outputs
are deliberately not committed.
