
from pathlib import Path
import numpy as np

# Step 1: Set file paths
ROOT = Path(__file__).resolve().parents[2]

scgpt_path = ROOT / "code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz"
expression_dir = ROOT / "data/processed/baron_classical_v1"


# Step 2: Load scGPT embeddings
with np.load(scgpt_path, allow_pickle=False) as data:
    print("Available keys:", data.files)

    vectors = data["vectors"]
    cell_ids = data["cell_ids"].astype(str)

print("\nscGPT shape:", vectors.shape)
print("Number of cell IDs:", len(cell_ids))


# Step 3: Check scGPT data
assert vectors.shape[0] == len(cell_ids)
assert len(np.unique(cell_ids)) == len(cell_ids)
assert np.isfinite(vectors).all()

print("scGPT data: VALID")


# Step 4: Load Expression cell IDs
with np.load(expression_dir / "train.npz", allow_pickle=False) as data:
    train_ids = data["cell_ids"].astype(str)

with np.load(expression_dir / "validation.npz", allow_pickle=False) as data:
    val_ids = data["cell_ids"].astype(str)


# Step 5: Check matching cell IDs
scgpt_ids = set(cell_ids)

missing_train = set(train_ids) - scgpt_ids
missing_val = set(val_ids) - scgpt_ids

print("\nTraining cells:", len(train_ids))
print("Validation cells:", len(val_ids))

print("Missing training cells:", len(missing_train))
print("Missing validation cells:", len(missing_val))

assert len(missing_train) == 0
assert len(missing_val) == 0

print("\nscGPT cell ID matching: SUCCESS!")
