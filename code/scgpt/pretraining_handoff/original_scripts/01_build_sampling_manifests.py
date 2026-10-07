#!/usr/bin/env python
"""Reproduce the official scTab cell universe and create nested sample lists."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import cellxgene_census
import networkx as nx
import numpy as np
import obonet
import pandas as pd


ROOT = Path("/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling")
MANIFEST_DIR = ROOT / "manifests"
ONTOLOGY_DIR = ROOT / "data" / "ontology"
CENSUS_VERSION = "2023-05-15"
ONTOLOGY_URL = (
    "https://github.com/obophenotype/cell-ontology/releases/download/"
    "v2023-05-22/cl-simple.obo"
)
EXPECTED_ELIGIBLE_CELLS = 22_190_622
PAPER_REPORTED_CELLS = 22_189_056
MAX_CELLS = 222_000
SEEDS = (0, 1, 2)
SIZES = (2_220, 4_440, 8_880, 17_760, 35_520, 71_040, 142_080, 222_000)
PROTOCOLS = (
    "10x 5' v2",
    "10x 3' v3",
    "10x 3' v2",
    "10x 5' v1",
    "10x 3' v1",
    "10x 3' transcription profiling",
    "10x 5' transcription profiling",
)
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


def load_cell_ontology() -> nx.MultiDiGraph:
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    ontology_path = ONTOLOGY_DIR / "cl-simple-v2023-05-22.obo"
    if not ontology_path.exists():
        import urllib.request

        urllib.request.urlretrieve(ONTOLOGY_URL, ontology_path)

    graph = obonet.read_obo(ontology_path, ignore_obsolete=True)
    for source, target, key in list(graph.edges(keys=True)):
        if key != "is_a":
            graph.remove_edge(source, target, key=key)
    return graph


def eligible_sctab_obs() -> tuple[pd.DataFrame, dict[str, object]]:
    graph = load_cell_ontology()
    id_to_name = {node_id: data.get("name") for node_id, data in graph.nodes(data=True)}
    name_to_id = {name: node_id for node_id, name in id_to_name.items() if name}

    def child_names(cell_type: str) -> list[str]:
        return [id_to_name[node] for node in nx.ancestors(graph, name_to_id[cell_type])]

    def parent_names(cell_type: str) -> list[str]:
        return [id_to_name[node] for node in nx.descendants(graph, name_to_id[cell_type])]

    protocol_literal = repr(list(PROTOCOLS))
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        obs = (
            census["census_data"]["homo_sapiens"]
            .obs.read(
                column_names=list(OBS_COLUMNS),
                value_filter=f"is_primary_data == True and assay in {protocol_literal}",
            )
            .concat()
            .to_pandas()
        )

    obs["tech_sample"] = (obs["dataset_id"] + "_" + obs["donor_id"]).astype("category")
    for column in OBS_COLUMNS:
        if obs[column].dtype == object:
            obs[column] = obs[column].astype("category")

    native_children = set(child_names("native cell"))
    cell_types_to_remove = set(obs.loc[~obs["cell_type"].isin(native_children), "cell_type"])

    cell_counts = obs["cell_type"].value_counts()
    cell_types_to_remove.update(cell_counts[cell_counts < 5_000].index)

    tech_samples = obs.groupby("cell_type", observed=True)["tech_sample"].nunique()
    cell_types_to_remove.update(tech_samples[tech_samples <= 30].index)

    for cell_type in obs["cell_type"].dropna().unique():
        if cell_type not in name_to_id or len(parent_names(cell_type)) <= 7:
            cell_types_to_remove.add(cell_type)

    eligible = obs.loc[~obs["cell_type"].isin(cell_types_to_remove)].copy()
    eligible = eligible.drop(columns="tech_sample")
    for column in eligible.columns:
        if isinstance(eligible[column].dtype, pd.CategoricalDtype):
            eligible[column] = eligible[column].cat.remove_unused_categories()

    summary = {
        "census_version": CENSUS_VERSION,
        "ontology_url": ONTOLOGY_URL,
        "eligible_cells": int(len(eligible)),
        "eligible_cell_types": int(eligible["cell_type"].nunique()),
        "eligible_donors": int(eligible["donor_id"].nunique()),
        "eligible_tissues_general": int(eligible["tissue_general"].nunique()),
        "protocols": list(PROTOCOLS),
        "expected_eligible_cells": EXPECTED_ELIGIBLE_CELLS,
        "paper_reported_cells": PAPER_REPORTED_CELLS,
        "upstream_notebook_minus_paper_cells": EXPECTED_ELIGIBLE_CELLS - PAPER_REPORTED_CELLS,
        "duplicate_erratum_correction_applied": False,
        "python": sys.version,
        "platform": platform.platform(),
        "cellxgene_census": cellxgene_census.__version__,
    }
    return eligible, summary


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    eligible, summary = eligible_sctab_obs()
    if len(eligible) != EXPECTED_ELIGIBLE_CELLS:
        raise RuntimeError(
            f"Eligible universe has {len(eligible):,} cells; expected "
            f"{EXPECTED_ELIGIBLE_CELLS:,}. Refusing to sample from a mismatched universe."
        )

    counts = (
        eligible.groupby(["cell_type", "tissue_general"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    counts.to_csv(MANIFEST_DIR / "eligible_cell_type_tissue_counts.tsv", sep="\t", index=False)

    eligible_ids = eligible["soma_joinid"].to_numpy(dtype=np.int64)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        selected_positions = rng.choice(len(eligible_ids), size=MAX_CELLS, replace=False)
        selected = eligible.iloc[selected_positions].copy()
        selected.insert(0, "sampling_rank", np.arange(MAX_CELLS, dtype=np.int64))
        selected.insert(0, "seed", seed)
        output = MANIFEST_DIR / f"sctab_seed{seed}_n{MAX_CELLS}_cells.parquet"
        selected.to_parquet(output, index=False)
        summary[f"seed_{seed}_manifest"] = {
            "path": str(output),
            "sha256": sha256(output),
            "n_cells": MAX_CELLS,
        }

    summary["nested_sizes"] = list(SIZES)
    with (MANIFEST_DIR / "sampling_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
