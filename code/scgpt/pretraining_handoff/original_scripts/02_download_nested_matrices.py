#!/usr/bin/env python
"""Download one 222k scTab seed and write all nested raw-count matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import cellxgene_census
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path("/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling")
CENSUS_VERSION = "2023-05-15"
MAX_CELLS = 222_000
SIZES = (2_220, 4_440, 8_880, 17_760, 35_520, 71_040, 142_080, 222_000)
OBS_COLUMNS = (
    "soma_joinid",
    "is_primary_data",
    "dataset_id",
    "donor_id",
    "assay",
    "cell_type",
    "development_stage",
    "disease",
    "tissue",
    "tissue_general",
    "sex",
    "self_reported_ethnicity",
    "suspension_type",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(seed: int) -> None:
    manifest_path = ROOT / "manifests" / f"sctab_seed{seed}_n{MAX_CELLS}_cells.parquet"
    feature_path = ROOT / "sources" / "scTab" / "notebooks" / "store_creation" / "features.parquet"
    seed_dir = ROOT / "data" / "nested_raw" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path).sort_values("sampling_rank")
    if len(manifest) != MAX_CELLS or manifest["soma_joinid"].duplicated().any():
        raise RuntimeError("Sampling manifest is not a unique 222,000-cell set.")

    protein_coding_genes = pd.read_parquet(feature_path)["gene_names"].tolist()
    selected_ids = manifest["soma_joinid"].astype(np.int64).tolist()

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            X_name="raw",
            obs_coords=selected_ids,
            var_value_filter=f"feature_name in {protein_coding_genes!r}",
            column_names={
                "obs": list(OBS_COLUMNS),
                "var": ["feature_id", "feature_name", "feature_length"],
            },
        )

    rank_lookup = manifest.set_index("soma_joinid")["sampling_rank"]
    adata.obs["sampling_rank"] = adata.obs["soma_joinid"].map(rank_lookup)
    if adata.obs["sampling_rank"].isna().any() or adata.n_obs != MAX_CELLS:
        raise RuntimeError("Downloaded cells do not exactly match the sampling manifest.")
    adata = adata[np.argsort(adata.obs["sampling_rank"].to_numpy())].copy()

    if not sparse.isspmatrix_csr(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32)
    adata.obs_names = [f"sctab_{int(joinid)}" for joinid in adata.obs["soma_joinid"]]
    adata.var_names = adata.var["feature_id"].astype(str)
    adata.var_names_make_unique()
    adata.uns["source"] = "CELLxGENE Census LTS 2023-05-15, official scTab eligibility rules"
    adata.uns["sampling_seed"] = seed
    adata.uns["values"] = "raw counts"
    adata.uns["nested_sizes"] = list(SIZES)

    records: list[dict[str, object]] = []
    for size in SIZES:
        output = seed_dir / f"sctab_seed{seed}_n{size}_raw.h5ad"
        subset = adata[:size].copy()
        subset.uns["n_pretraining_cells"] = size
        subset.write_h5ad(output, compression="gzip")
        records.append(
            {
                "seed": seed,
                "n_cells": size,
                "n_genes": subset.n_vars,
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
            }
        )
        print(f"WROTE {output} {subset.shape}", flush=True)

    inventory = pd.DataFrame.from_records(records)
    inventory.to_csv(ROOT / "manifests" / f"matrix_inventory_seed{seed}.tsv", sep="\t", index=False)
    with (ROOT / "manifests" / f"matrix_inventory_seed{seed}.json").open("w") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    main(args.seed)
