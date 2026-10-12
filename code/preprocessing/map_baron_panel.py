from pathlib import Path
import pandas as pd
import anndata as ad

root = Path("data/processed")
panel_path = Path(
    "code/scgpt/pretraining_handoff/assets/sctab_hvg_2000.tsv"
)

adata = ad.read_h5ad(root / "baron_raw.h5ad")
panel = pd.read_csv(panel_path, sep="\t", dtype=str)

assert len(panel) == 2000
assert panel["feature_id"].is_unique
assert panel["feature_name"].is_unique
assert adata.var_names.is_unique

mapping = panel[["feature_id", "feature_name"]].copy()
mapping.columns = ["ensembl_id", "gene_symbol"]

mapping["present_in_baron"] = (
    mapping["gene_symbol"].isin(adata.var_names)
)

mapping["baron_gene_index"] = [
    adata.var_names.get_loc(symbol) if symbol in adata.var_names else -1
    for symbol in mapping["gene_symbol"]
]

mapping["mapping_source"] = "sctab_hvg_2000.tsv"

output = root / "baron_scgpt_gene_mapping.csv"
mapping.to_csv(output, index=False)

matched = int(mapping["present_in_baron"].sum())
missing = len(mapping) - matched

print("Panel genes:", len(mapping))
print("Matched genes:", matched)
print("Missing genes:", missing)
print("Coverage:", f"{100 * matched / len(mapping):.2f}%")
print("Mapping saved to:", output)
