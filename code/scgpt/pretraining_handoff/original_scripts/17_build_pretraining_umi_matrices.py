#!/usr/bin/env python
"""Build nested UMI-thinned 222k scTab panel matrices for three replicates."""

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


ROOT = Path("/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling")
PANEL_FILE = ROOT / "manifests/pca/sctab_hvg_2000.tsv"
OUTPUT_ROOT = ROOT / "data/pretraining_umi_scaling"
RATES_DESCENDING = (1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.025)
THINNING_SEED_BASE = 20260827
N_CELLS = 222_000
MAX_ZERO_PANEL_FRACTION = 0.05


def rate_label(rate: float) -> str:
    return f"rho{int(round(rate * 10_000)):04d}"


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


def atomic_h5ad(data: ad.AnnData, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.write_h5ad(temporary, compression="gzip", compression_opts=4)
    os.replace(temporary, path)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p05": float(quantiles[0]),
        f"{prefix}_p25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_p75": float(quantiles[3]),
        f"{prefix}_p95": float(quantiles[4]),
        f"{prefix}_sum": int(np.sum(values, dtype=np.int64)),
    }


def build_seed(seed: int, overwrite: bool) -> None:
    if seed not in (0, 1, 2):
        raise ValueError("seed must be 0, 1, or 2")
    destination = OUTPUT_ROOT / f"seed_{seed}"
    success = destination / "_SUCCESS"
    if success.exists() and not overwrite:
        print(f"Reusing completed thinning replicate {seed}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)

    source_file = ROOT / f"data/nested_raw/seed_{seed}/sctab_seed{seed}_n222000_raw.h5ad"
    source = ad.read_h5ad(source_file)
    if source.n_obs != N_CELLS or not sp.issparse(source.X) or not source.obs_names.is_unique:
        raise ValueError(f"Invalid source matrix: {source.shape}")
    full = sp.csr_matrix(source.X, dtype=np.float32)
    if np.any(full.data < 0) or not np.allclose(full.data, np.rint(full.data)):
        raise ValueError("Source matrix is not nonnegative integer counts")
    full.sort_indices()

    panel = pd.read_csv(PANEL_FILE, sep="\t")["feature_id"].astype(str).tolist()
    if len(panel) != 2_000 or len(set(panel)) != 2_000:
        raise ValueError("Invalid fixed scTab gene panel")
    panel_positions = source.var_names.get_indexer(panel)
    if np.any(panel_positions < 0):
        raise ValueError("Fixed panel is not contained in the source matrix")

    original_totals = np.asarray(full.sum(axis=1)).ravel().astype(np.int64)
    current = np.rint(full.data).astype(np.int64, copy=True)
    rng = np.random.default_rng(THINNING_SEED_BASE + seed)
    previous_rate = 1.0
    rows: list[dict[str, object]] = []

    for rate in RATES_DESCENDING:
        if rate < previous_rate:
            current = rng.binomial(current, rate / previous_rate).astype(np.int64, copy=False)
        thinned_full = sp.csr_matrix(
            (current.astype(np.float32, copy=False), full.indices, full.indptr),
            shape=full.shape,
        )
        retained_totals = np.asarray(thinned_full.sum(axis=1)).ravel().astype(np.int64)
        panel_matrix = sp.csr_matrix(thinned_full[:, panel_positions], dtype=np.float32)
        panel_matrix.eliminate_zeros()
        panel_totals = np.asarray(panel_matrix.sum(axis=1)).ravel().astype(np.int64)
        zero_panel_cells = int(np.count_nonzero(panel_totals == 0))
        zero_panel_fraction = zero_panel_cells / N_CELLS

        label = rate_label(rate)
        output_file = destination / f"sctab_seed{seed}_n222000_{label}_panel2000.h5ad"
        obs = pd.DataFrame(
            {
                "original_total_umi": original_totals,
                "retained_total_umi": retained_totals,
                "retained_panel_umi": panel_totals,
            },
            index=source.obs_names.copy(),
        )
        output = ad.AnnData(
            panel_matrix,
            obs=obs,
            var=pd.DataFrame(index=pd.Index(panel, dtype=str)),
        )
        output.uns["pretraining_umi_scaling"] = {
            "source_file": str(source_file),
            "sampling_seed": seed,
            "thinning_seed": THINNING_SEED_BASE + seed,
            "nominal_retention_rate": rate,
            "nested_thinning": True,
            "cells": N_CELLS,
            "panel_genes": len(panel),
        }
        atomic_h5ad(output, output_file)
        row = {
            "sampling_seed": seed,
            "thinning_seed": THINNING_SEED_BASE + seed,
            "retention_rate": rate,
            "rate_label": label,
            "cells": N_CELLS,
            "panel_genes": len(panel),
            "zero_panel_cells": zero_panel_cells,
            "zero_panel_fraction": zero_panel_fraction,
            **summarize(original_totals, "original_total_umi"),
            **summarize(retained_totals, "retained_total_umi"),
            **summarize(panel_totals, "retained_panel_umi"),
            "matrix_nnz": int(panel_matrix.nnz),
            "matrix_file": str(output_file),
            "matrix_sha256": sha256(output_file),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        previous_rate = rate

    summary = pd.DataFrame(rows).sort_values("retention_rate")
    summary_file = destination / "umi_thinning_summary.csv"
    temporary = summary_file.with_suffix(".csv.tmp")
    summary.to_csv(temporary, index=False)
    os.replace(temporary, summary_file)
    manifest = {
        "sampling_seed": seed,
        "source_file": str(source_file),
        "panel_file": str(PANEL_FILE),
        "rates": sorted(RATES_DESCENDING),
        "thinning_seed": THINNING_SEED_BASE + seed,
        "nested_thinning": True,
        "maximum_allowed_zero_panel_fraction": MAX_ZERO_PANEL_FRACTION,
        "observed_maximum_zero_panel_fraction": float(summary["zero_panel_fraction"].max()),
        "summary_file": str(summary_file),
    }
    if manifest["observed_maximum_zero_panel_fraction"] > MAX_ZERO_PANEL_FRACTION:
        atomic_text(destination / "_QC_FAILED", json.dumps(manifest, indent=2) + "\n")
        raise ValueError(f"Zero-panel fraction exceeds QC limit: {manifest}")
    atomic_text(destination / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_text(success, sha256(summary_file) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_seed(args.seed, args.overwrite)


if __name__ == "__main__":
    main()
