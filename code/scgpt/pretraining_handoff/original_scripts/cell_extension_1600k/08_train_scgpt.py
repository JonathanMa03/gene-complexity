#!/usr/bin/env python
"""Pretrain one small, fixed scGPT model on a nested scTab matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from scgpt.data_collator import DataCollator
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab


ROOT = Path("/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling")
PANEL_FILE = ROOT / "manifests" / "pca" / "sctab_hvg_2000.tsv"
OUTPUT_ROOT = ROOT / "outputs" / "cell_extension_1600k" / "scgpt_checkpoints"
SIZES = (2_220, 4_440, 8_880, 17_760, 35_520, 71_040, 142_080, 222_000)
SPECIAL_TOKENS = ("<pad>", "<cls>", "<eoc>")
CONFIG = {
    "model_seed": 20260821,
    "epochs": 10,
    "batch_size": 64,
    "max_length": 1200,
    "mask_ratio": 0.4,
    "n_bins": 51,
    "embsize": 128,
    "nheads": 4,
    "d_hid": 512,
    "nlayers": 4,
    "n_layers_cls": 3,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_fraction": 0.05,
    "gradient_clip": 1.0,
    "amp_dtype": "bfloat16",
    "pad_token": "<pad>",
    "cls_token": "<cls>",
    "pad_value": -2,
    "mask_value": -1,
    "objective": "masked gene-expression prediction plus MVC/GEPC cell-embedding prediction; equal-weight masked MSE",
    "input_style": "nonzero raw counts binned within each cell to 51 bins",
    "cell_embedding_style": "CLS",
    "use_fast_transformer": False,
    "pre_norm": False,
}


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
            "id": torch.tensor(index, dtype=torch.long),
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




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def make_vocab(genes: list[str]) -> tuple[GeneVocab, dict[str, int]]:
    ordered = list(SPECIAL_TOKENS) + genes
    mapping = {token: index for index, token in enumerate(ordered)}
    vocab = GeneVocab.from_dict(mapping)
    vocab.set_default_index(vocab[CONFIG["pad_token"]])
    return vocab, mapping


def make_model(vocab: GeneVocab) -> TransformerModel:
    return TransformerModel(
        ntoken=len(vocab),
        d_model=CONFIG["embsize"],
        nhead=CONFIG["nheads"],
        d_hid=CONFIG["d_hid"],
        nlayers=CONFIG["nlayers"],
        nlayers_cls=CONFIG["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=CONFIG["dropout"],
        pad_token=CONFIG["pad_token"],
        pad_value=CONFIG["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        cell_emb_style="cls",
        explicit_zero_prob=False,
        use_fast_transformer=CONFIG["use_fast_transformer"],
        pre_norm=CONFIG["pre_norm"],
    )


def lr_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for scGPT pretraining")

    sampling_seed, n_cells = resolve(args.model_index)
    sampling_seed = int(os.environ.get("TRAINING_SEED", sampling_seed))
    n_cells = int(os.environ.get("TRAINING_N_CELLS", n_cells))
    model_id = os.environ.get(
        "PRETRAINING_UMI_MODEL_ID", f"sctab_seed{sampling_seed}_n{n_cells}_scgpt"
    )
    input_file = Path(os.environ.get(
        "PRETRAINING_UMI_INPUT",
        ROOT / "data" / "nested_raw" / f"seed_{sampling_seed}"
        / f"sctab_seed{sampling_seed}_n{n_cells}_raw.h5ad",
    ))
    output_root = Path(os.environ.get("PRETRAINING_UMI_OUTPUT_ROOT", OUTPUT_ROOT))
    output_dir = output_root / model_id
    final_file = output_dir / "model.pt"
    resume_file = output_dir / "checkpoint_last.pt"
    success_file = output_dir / "_SUCCESS"
    if success_file.exists() and not args.overwrite:
        print(f"Reusing complete scGPT checkpoint {model_id}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(CONFIG["model_seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    panel = pd.read_csv(PANEL_FILE, sep="\t")
    genes = panel["feature_id"].astype(str).tolist()
    if len(genes) != 2_000 or len(set(genes)) != 2_000:
        raise ValueError("Invalid fixed gene panel")
    vocab, vocab_mapping = make_vocab(genes)
    atomic_json(output_dir / "vocab.json", vocab_mapping)

    adata = ad.read_h5ad(input_file)
    if adata.n_obs != n_cells or adata.n_vars not in (2_000, 19_331) or not sp.issparse(adata.X):
        raise ValueError(f"Unexpected source matrix: {adata.shape}")
    missing = pd.Index(genes).difference(adata.var_names)
    if len(missing):
        raise ValueError(f"Panel genes absent from scTab: {missing[:10].tolist()}")
    matrix = sp.csr_matrix(adata.X if adata.n_vars == 2_000 else adata[:, genes].X, dtype=np.float32)
    matrix.eliminate_zeros()
    nonzero_per_cell = np.diff(matrix.indptr)
    keep = nonzero_per_cell > 0
    n_zero_panel_cells = int((~keep).sum())
    if n_zero_panel_cells:
        matrix = matrix[keep]
    if np.any(matrix.data < 0) or matrix.shape[1] != 2_000:
        raise ValueError("Invalid panel count matrix")
    del adata

    gene_token_ids = np.asarray([vocab[gene] for gene in genes], dtype=np.int64)
    dataset = SparseExpressionDataset(
        matrix, gene_token_ids, vocab[CONFIG["cls_token"]], CONFIG["pad_value"]
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[CONFIG["pad_token"]],
        pad_value=CONFIG["pad_value"],
        do_mlm=True,
        do_binning=True,
        mlm_probability=CONFIG["mask_ratio"],
        mask_value=CONFIG["mask_value"],
        max_length=CONFIG["max_length"],
        sampling=True,
        keep_first_n_tokens=1,
    )

    device = torch.device("cuda")
    model = make_model(vocab).to(device)
    initial_checkpoint = os.environ.get("INITIAL_CHECKPOINT")
    if initial_checkpoint:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"], strict=True)
        print(f"Initialized from {initial_checkpoint}", flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    learning_rate = float(os.environ.get("TRAINING_LEARNING_RATE", CONFIG["learning_rate"]))
    epochs = int(os.environ.get("TRAINING_EPOCHS", CONFIG["epochs"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=CONFIG["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(dataset) / CONFIG["batch_size"])
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(CONFIG["warmup_fraction"] * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch = 0
    global_step = 0
    history: list[dict[str, object]] = []
    if resume_file.exists() and not args.overwrite:
        checkpoint = torch.load(resume_file, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
        print(f"Resuming {model_id} from epoch {start_epoch}", flush=True)

    metadata = {
        **CONFIG,
        "model_id": model_id,
        "model_index": args.model_index,
        "sampling_seed": sampling_seed,
        "requested_pretraining_cells": n_cells,
        "used_pretraining_cells": len(dataset),
        "zero_panel_cells_removed": n_zero_panel_cells,
        "panel_genes": len(genes),
        "vocab_size": len(vocab),
        "parameter_count": parameter_count,
        "input_file": str(input_file),
        "panel_file": str(PANEL_FILE),
        "scgpt_repository_commit": "cebd6fae655b9c585a4807daa3ac31bb764f06b4",
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "training_split": "all available nonzero-panel cells; no checkpoint selection",
        "initial_checkpoint": initial_checkpoint,
        "effective_learning_rate": learning_rate,
        "effective_epochs": epochs,
    }
    atomic_json(output_dir / "config.json", metadata)

    started = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(CONFIG["model_seed"] + epoch)
        loader = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            generator=epoch_generator,
            drop_last=False,
        )
        model.train()
        loss_sum = 0.0
        mlm_loss_sum = 0.0
        mvc_loss_sum = 0.0
        masked_values = 0
        completed_batches = 0
        for batch_index, batch in enumerate(loader, start=1):
            genes_batch = batch["gene"].to(device, non_blocking=True)
            target_values = batch["expr"].to(device, non_blocking=True)
            masked_values_input = batch["masked_expr"].to(device, non_blocking=True)
            padding_mask = genes_batch.eq(vocab[CONFIG["pad_token"]])
            masked_positions = masked_values_input.eq(CONFIG["mask_value"])
            if not masked_positions.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    genes_batch,
                    masked_values_input,
                    src_key_padding_mask=padding_mask,
                    MVC=True,
                )
                loss_mlm = masked_mse_loss(output["mlm_output"], target_values, masked_positions)
                loss_mvc = masked_mse_loss(output["mvc_output"], target_values, masked_positions)
                loss = loss_mlm + loss_mvc
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            count_masked = int(masked_positions.sum().item())
            loss_sum += float(loss.item()) * count_masked
            mlm_loss_sum += float(loss_mlm.item()) * count_masked
            mvc_loss_sum += float(loss_mvc.item()) * count_masked
            masked_values += count_masked
            completed_batches += 1
            if batch_index % 100 == 0 or batch_index == steps_per_epoch:
                print(
                    f"{model_id} epoch={epoch + 1}/{epochs} "
                    f"batch={batch_index}/{steps_per_epoch} loss={loss.item():.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3g}",
                    flush=True,
                )
        epoch_loss = loss_sum / max(masked_values, 1)
        epoch_mlm_loss = mlm_loss_sum / max(masked_values, 1)
        epoch_mvc_loss = mvc_loss_sum / max(masked_values, 1)
        epoch_row = {
            "epoch": epoch + 1,
            "total_masked_mse": epoch_loss,
            "mlm_masked_mse": epoch_mlm_loss,
            "mvc_masked_mse": epoch_mvc_loss,
            "masked_values": masked_values,
            "completed_batches": completed_batches,
            "global_step": global_step,
            "learning_rate_end": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
        atomic_torch_save(
            resume_file,
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "config": metadata,
            },
        )
        print(json.dumps(epoch_row), flush=True)

    atomic_torch_save(final_file, {"model": model.state_dict(), "config": metadata})
    metadata.update(
        {
            "elapsed_seconds": time.time() - started,
            "final_total_masked_mse": history[-1]["total_masked_mse"],
            "final_mlm_masked_mse": history[-1]["mlm_masked_mse"],
            "final_mvc_masked_mse": history[-1]["mvc_masked_mse"],
            "checkpoint_sha256": sha256(final_file),
        }
    )
    atomic_json(output_dir / "config.json", metadata)
    atomic_json(output_dir / "training_summary.json", {"metadata": metadata, "history": history})
    (output_dir / "_SUCCESS").write_text(metadata["checkpoint_sha256"] + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
