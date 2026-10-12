#!/usr/bin/env python
"""Generate frozen DOGMA CLS representations from one trained scGPT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab


LAB_ROOT = Path("/n/holylabs/rongma_lab/Lab/scaling_law")
ROOT = LAB_ROOT / "jingyuan" / "pretraining_cell_scaling"
PANEL_FILE = ROOT / "manifests" / "pca" / "sctab_hvg_2000.tsv"
CHECKPOINT_ROOT = ROOT / "outputs" / "cell_extension_1600k" / "scgpt_checkpoints"
OUTPUT_ROOT = ROOT / "outputs" / "cell_extension_1600k" / "dogma_scgpt_representations"
DOGMA_FILE = (
    LAB_ROOT
    / "jingyuan/dogma_seq/pca_scaling_pipeline/outputs/01_umi_matrices/01_umi_rho100_rep01.h5ad"
)
COMMON_CELLS_FILE = (
    LAB_ROOT
    / "jingyuan/dogma_seq/pca_scaling_pipeline/outputs/03_umi_scaling/03_common_cells.txt"
)
SIZES = (2_220, 4_440, 8_880, 17_760, 35_520, 71_040, 142_080, 222_000)


class SparseExpressionDataset(Dataset):
    def __init__(self, matrix: sp.csr_matrix, gene_token_ids: np.ndarray, cls_id: int, pad_value: int):
        self.matrix = matrix
        self.gene_token_ids = gene_token_ids
        self.cls_id = cls_id
        self.pad_value = pad_value

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end = self.matrix.indptr[index], self.matrix.indptr[index + 1]
        columns = self.matrix.indices[start:end]
        values = self.matrix.data[start:end]
        genes = np.empty(len(columns) + 1, dtype=np.int64)
        expressions = np.empty(len(columns) + 1, dtype=np.float32)
        genes[0] = self.cls_id
        expressions[0] = self.pad_value
        genes[1:] = self.gene_token_ids[columns]
        expressions[1:] = values
        return {
            "genes": torch.from_numpy(genes),
            "expressions": torch.from_numpy(expressions),
        }


def resolve(index: int) -> tuple[int, int]:
    if not 0 <= index < 33:
        raise ValueError("index must be in [0,32]")
    if index < 24:
        return index // 8, SIZES[index % 8]
    if index < 30:
        return (index - 24) // 2, (444000,888000)[(index - 24) % 2]
    return index - 30, 1600000




def make_model(vocab: GeneVocab, config: dict[str, object]) -> TransformerModel:
    return TransformerModel(
        ntoken=len(vocab),
        d_model=config["embsize"],
        nhead=config["nheads"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        nlayers_cls=config["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=config["dropout"],
        pad_token=config["pad_token"],
        pad_value=config["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        cell_emb_style="cls",
        explicit_zero_prob=False,
        use_fast_transformer=config["use_fast_transformer"],
        pre_norm=config["pre_norm"],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_binary(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    values.tofile(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for scGPT representation extraction")
    sampling_seed, n_cells = resolve(args.model_index)
    sampling_seed = int(os.environ.get("TRAINING_SEED", sampling_seed))
    n_cells = int(os.environ.get("TRAINING_N_CELLS", n_cells))
    model_id = os.environ.get(
        "PRETRAINING_UMI_MODEL_ID", f"sctab_seed{sampling_seed}_n{n_cells}_scgpt"
    )
    checkpoint_root = Path(os.environ.get("PRETRAINING_UMI_CHECKPOINT_ROOT", CHECKPOINT_ROOT))
    output_root = Path(os.environ.get("PRETRAINING_UMI_OUTPUT_ROOT", OUTPUT_ROOT))
    checkpoint_dir = checkpoint_root / model_id
    checkpoint_file = checkpoint_dir / "model.pt"
    destination = output_root / model_id
    success_file = destination / "_SUCCESS"
    if success_file.exists() and not args.overwrite:
        print(f"Reusing complete DOGMA representation {model_id}", flush=True)
        return
    if not (checkpoint_dir / "_SUCCESS").exists():
        raise FileNotFoundError(f"Incomplete scGPT checkpoint: {model_id}")
    destination.mkdir(parents=True, exist_ok=True)

    config = json.loads((checkpoint_dir / "config.json").read_text())
    vocab_mapping = json.loads((checkpoint_dir / "vocab.json").read_text())
    vocab = GeneVocab.from_dict(vocab_mapping)
    vocab.set_default_index(vocab[config["pad_token"]])
    panel = pd.read_csv(PANEL_FILE, sep="\t")
    panel_genes = panel["feature_id"].astype(str).to_numpy()
    if len(panel_genes) != 2_000 or any(gene not in vocab for gene in panel_genes):
        raise ValueError("Checkpoint vocabulary and fixed panel disagree")

    dogma = ad.read_h5ad(DOGMA_FILE)
    common_cells = pd.Index(COMMON_CELLS_FILE.read_text().splitlines(), dtype=str)
    cell_positions = dogma.obs_names.get_indexer(common_cells)
    if len(common_cells) != 21_524 or np.any(cell_positions < 0) or not sp.issparse(dogma.X):
        raise ValueError("Invalid DOGMA raw input or common-cell bank")
    counts = sp.csr_matrix(dogma.X[cell_positions, :], dtype=np.float32)
    dogma_ensembl = dogma.var["ensembl_id"].astype(str).to_numpy()
    panel_lookup = {gene: index for index, gene in enumerate(panel_genes)}
    source_columns: list[int] = []
    target_columns: list[int] = []
    for source, gene in enumerate(dogma_ensembl):
        target = panel_lookup.get(gene)
        if target is not None:
            source_columns.append(source)
            target_columns.append(target)
    mapping = sp.csr_matrix(
        (np.ones(len(source_columns), dtype=np.float32), (source_columns, target_columns)),
        shape=(dogma.n_vars, len(panel_genes)),
    )
    matrix = sp.csr_matrix(counts @ mapping, dtype=np.float32)
    matrix.eliminate_zeros()
    if np.any(np.diff(matrix.indptr) == 0):
        raise ValueError("DOGMA contains a zero-panel cell")
    del dogma, counts, mapping

    gene_token_ids = np.asarray([vocab[gene] for gene in panel_genes], dtype=np.int64)
    dataset = SparseExpressionDataset(
        matrix, gene_token_ids, vocab[config["cls_token"]], config["pad_value"]
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[config["pad_token"]],
        pad_value=config["pad_value"],
        do_mlm=False,
        do_binning=True,
        mlm_probability=config["mask_ratio"],
        mask_value=config["mask_value"],
        max_length=config["max_length"],
        sampling=False,
        keep_first_n_tokens=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    device = torch.device("cuda")
    model = make_model(vocab, config)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    embeddings = np.zeros((len(dataset), config["embsize"]), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            gene_batch = batch["gene"].to(device, non_blocking=True)
            expression_batch = batch["expr"].to(device, non_blocking=True)
            padding_mask = gene_batch.eq(vocab[config["pad_token"]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                encoded = model._encode(gene_batch, expression_batch, padding_mask)
                cell_embedding = F.normalize(encoded[:, 0, :], p=2, dim=1)
            values = cell_embedding.float().cpu().numpy()
            embeddings[offset : offset + len(values)] = values
            offset += len(values)
    norms = np.linalg.norm(embeddings, axis=1)
    if offset != len(dataset) or not np.isfinite(embeddings).all() or np.max(np.abs(norms - 1)) > 2e-3:
        raise ValueError("Invalid scGPT DOGMA embeddings")
    variances = embeddings.var(axis=0, ddof=1)
    if np.count_nonzero(variances > 1e-10) < 12:
        raise ValueError("scGPT representation has numerical rank below 12")

    score_file = destination / "embeddings.f32"
    atomic_binary(score_file, np.asarray(embeddings, dtype="<f4", order="C"))
    atomic_text(destination / "embeddings.dims", f"{embeddings.shape[0]}\t{embeddings.shape[1]}\n")
    atomic_text(destination / "cells.txt", "\n".join(common_cells) + "\n")
    matched_panel = set(dogma_ensembl).intersection(panel_genes)
    metadata = {
        "model_id": model_id,
        "model_index": args.model_index,
        "sampling_seed": sampling_seed,
        "pretraining_cells": n_cells,
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": sha256(checkpoint_file),
        "dogma_source": str(DOGMA_FILE),
        "dogma_cells": len(common_cells),
        "representation_dimension": int(embeddings.shape[1]),
        "pooling": "CLS followed by L2 normalization",
        "inference_tokenization": "nonzero raw counts; 51 within-cell bins; deterministic panel-rank truncation",
        "max_length": config["max_length"],
        "panel_genes": len(panel_genes),
        "dogma_matched_panel_genes": len(matched_panel),
        "dogma_missing_panel_genes": sorted(set(panel_genes).difference(matched_panel)),
        "embedding_sha256": sha256(score_file),
        "embedding_variance_min": float(variances.min()),
        "embedding_variance_max": float(variances.max()),
    }
    atomic_text(destination / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    atomic_text(success_file, metadata["embedding_sha256"] + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
