from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

root = Path("data/processed")

# Load exported data
counts = mmread(root / "baron_counts.mtx").tocsr()
genes = pd.read_csv(root / "baron_genes.csv", keep_default_na=False)
cells = pd.read_csv(root / "baron_cells.csv", keep_default_na=False)

# R exports genes x cells; AnnData requires cells x genes
X = counts.T.tocsr()

assert X.shape == (len(cells), len(genes))
assert genes["gene_symbol"].is_unique
assert cells["cell_id"].is_unique
assert np.isfinite(X.data).all()
assert (X.data >= 0).all()
assert np.allclose(X.data, np.round(X.data), atol=1e-6)

# Preserve integer counts
X.data = np.rint(X.data).astype(np.int32)
X = X.astype(np.int32)

obs = cells.set_index("cell_id")
var = genes.set_index("gene_symbol")

adata = ad.AnnData(X=X, obs=obs, var=var)
adata.layers["counts"] = adata.X.copy()

output = root / "baron_raw.h5ad"
adata.write_h5ad(output, compression="gzip")

# Verify saved file
check = ad.read_h5ad(output)

assert check.shape == (8569, 20125)
assert sparse.issparse(check.X)
assert check.X.nnz == adata.X.nnz
assert np.array_equal(check.X.data, adata.X.data)
assert check.obs_names.equals(adata.obs_names)
assert check.var_names.equals(adata.var_names)

print("Conversion successful!")
print("Shape (cells x genes):", check.shape)
print("Matrix type:", type(check.X).__name__)
print("Count dtype:", check.X.dtype)
print("Donors:", check.obs["donor"].nunique())
print("Cell types:", check.obs["cell_type"].nunique())
print("Saved to:", output)
