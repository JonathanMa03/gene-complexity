
from pathlib import Path
import numpy as np

# 1. Find the project root
ROOT = Path(__file__).resolve().parents[2]

# 2. Input folders
expression_dir = ROOT / "data/processed/baron_classical_v1"
dictionary_dir = (
    ROOT / "results/dict-learning/baron_grid_v1/"
    "k16_a1.0_s42_e10"
)


def load_features(path, feature_key):
    with np.load(path, allow_pickle=False) as data:
        print(f"\nFile: {path.name}")
        print("Available keys:", data.files)

        X = data[feature_key]
        cell_ids = data["cell_ids"].astype(str)
        labels = data["labels"].astype(str)

    print("Feature shape:", X.shape)
    print("Cell count:", len(cell_ids))
    print("Number of classes:", len(np.unique(labels)))

    assert X.shape[0] == len(cell_ids) == len(labels)
    assert len(np.unique(cell_ids)) == len(cell_ids)
    assert np.isfinite(X).all()

    return X, cell_ids, labels


# 3. Load train and validation data
for split in ["train", "validation"]:
    print(f"\n========== {split.upper()} ==========")

    X_expr, ids_expr, y_expr = load_features(
        expression_dir / f"{split}.npz", "X"
    )

    X_dict, ids_dict, y_dict = load_features(
        dictionary_dir / f"{split}_codes.npz", "vectors"
    )

    # 4. Verify sample identities
    assert set(ids_expr) == set(ids_dict), (
        "Expression and Dictionary cell IDs do not match."
    )

    # Match dictionary labels to expression cell order
    dictionary_labels = dict(zip(ids_dict, y_dict))
    aligned_labels = np.array(
        [dictionary_labels[cid] for cid in ids_expr]
    )

    assert np.array_equal(y_expr, aligned_labels), (
        "Labels do not match after cell ID alignment."
    )

    print("\nCell IDs: MATCH")
    print("Labels: MATCH")
    print("All feature values: FINITE")

print("\nData validation completed successfully!")
