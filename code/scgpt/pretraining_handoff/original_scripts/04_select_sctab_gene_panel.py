#!/usr/bin/env python
"""Freeze a 2,000-gene HVG panel using scTab expression only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling")
SOURCE = ROOT / "outputs" / "sctab_seed0_n222000_umap.h5ad"
OUTPUT_DIR = ROOT / "manifests" / "pca"
PANEL_FILE = OUTPUT_DIR / "sctab_hvg_2000.tsv"
SUMMARY_FILE = OUTPUT_DIR / "gene_panel_summary.json"
N_GENES = 2_000


def atomic_tsv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, destination)


def atomic_json(value: dict[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, destination)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(SOURCE, backed="r")
    required = {"feature_id", "feature_name", "means", "dispersions", "dispersions_norm"}
    missing = required.difference(adata.var.columns)
    if missing:
        raise ValueError(f"Source is missing HVG statistics: {sorted(missing)}")

    table = adata.var.loc[:, sorted(required)].copy().reset_index(drop=True)
    table["feature_id"] = table["feature_id"].astype(str)
    if table["feature_id"].duplicated().any() or not table.index.is_unique:
        raise ValueError("scTab feature IDs must be unique")
    table = table.loc[np.isfinite(table["dispersions_norm"].to_numpy())].copy()
    table = table.sort_values(
        ["dispersions_norm", "feature_id"], ascending=[False, True], kind="mergesort"
    ).head(N_GENES)
    if len(table) != N_GENES:
        raise ValueError(f"Expected {N_GENES} finite-ranked genes, found {len(table)}")
    table.insert(0, "panel_rank", np.arange(1, N_GENES + 1, dtype=int))
    table = table[
        ["panel_rank", "feature_id", "feature_name", "means", "dispersions", "dispersions_norm"]
    ]
    atomic_tsv(table, PANEL_FILE)

    summary = {
        "source": str(SOURCE),
        "source_cells": int(adata.n_obs),
        "source_genes": int(adata.n_vars),
        "panel_genes": N_GENES,
        "selection": "top normalized dispersion from Scanpy seurat-flavor HVG statistics",
        "selection_uses_dogma_expression": False,
        "normalization_used_to_create_source_statistics": "total-count 1e4; log1p; gene centered only during PCA",
        "stable_identifier": "Ensembl gene ID without version",
        "panel_file": str(PANEL_FILE),
    }
    atomic_json(summary, SUMMARY_FILE)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
