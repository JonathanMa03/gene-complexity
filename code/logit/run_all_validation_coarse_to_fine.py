from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[2]

EXPRESSION_DIR = ROOT / "data/processed/baron_classical_v1"
DICTIONARY_DIR = ROOT / "results/dict-learning/baron_grid_v1"
SCGPT_PATH = ROOT / "code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz"
OUTPUT_DIR = ROOT / "results/logit"

COARSE_C_VALUES = np.array([0.01, 0.1, 1, 10, 100, 1000], dtype=float)
FINE_POINTS = 21

DICTIONARY_RUNS = {
    "Dictionary_K16_alpha1": "k16_a1.0_s42_e10",
    "Dictionary_K16_alpha5": "k16_a5.0_s42_e10",
    "Dictionary_K32_alpha1": "k32_a1.0_s42_e10",
    "Dictionary_K32_alpha5": "k32_a5.0_s42_e10",
}


def find_final_manifest():
    preferred_paths = [
        EXPRESSION_DIR / "split_manifest.csv",
        EXPRESSION_DIR / "baron_split_manifest.csv",
        ROOT / "data/processed/baron_split_manifest.csv",
        ROOT / "data/processed/split_manifest.csv",
    ]

    for path in preferred_paths:
        if path.is_file():
            return path

    found_paths = set()
    for path in ROOT.glob("data/**/baron_split_manifest.csv"):
        found_paths.add(path)
    for path in ROOT.glob("data/**/split_manifest.csv"):
        found_paths.add(path)
    found_paths = sorted(found_paths)

    if len(found_paths) == 1:
        return found_paths[0]
    if len(found_paths) == 0:
        raise FileNotFoundError(
            "Finalized split manifest not found. Set its path in "
            "find_final_manifest() after locating baron_split_manifest.csv "
            "or split_manifest.csv in your repository."
        )
    raise RuntimeError(
        "Several split manifests found; explicitly select the finalized one: "
        + ", ".join(str(path) for path in found_paths)
    )


def string_ids(values):
    """Convert byte/string cell IDs into an array of Python strings."""
    values = np.asarray(values).reshape(-1)
    converted_ids = []

    for cell_id in values:
        if isinstance(cell_id, (bytes, np.bytes_)):
            converted_ids.append(cell_id.decode("utf-8"))
        else:
            converted_ids.append(str(cell_id))

    return np.array(converted_ids, dtype=str)


def load_npz(path, feature_key):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    with np.load(path, allow_pickle=True) as data:
        required_keys = {feature_key, "cell_ids"}
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            raise KeyError(f"{path} is missing NPZ keys: {sorted(missing_keys)}")

        X = data[feature_key]
        cell_ids = string_ids(data["cell_ids"])
        labels = None
        if "labels" in data.files:
            labels = data["labels"]

    # Make sure that the rows and IDs match.
    if X.ndim != 2 or len(cell_ids) != X.shape[0]:
        raise ValueError(f"Bad X / cell_ids dimensions in {path}")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError(f"Duplicate cell_ids found in {path}")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError(f"Non-numeric or non-finite feature values in {path}")
    if labels is not None and len(labels) != len(cell_ids):
        raise ValueError(f"labels and cell_ids lengths differ in {path}")

    return X, cell_ids, labels



# Align cells to the reference train/validation splits
def align_rows(X, source_ids, target_ids, description):
    cell_id_to_row = {}
    for i, cell_id in enumerate(source_ids):
        cell_id_to_row[cell_id] = i

    missing_ids = []
    for cell_id in target_ids:
        if cell_id not in cell_id_to_row:
            missing_ids.append(cell_id)

    if missing_ids:
        raise ValueError(
            f"{description}: missing {len(missing_ids)} reference cells. "
            f"Example: {missing_ids[:5]}"
        )

    row_indices = []
    for cell_id in target_ids:
        row_indices.append(cell_id_to_row[cell_id])

    X_aligned = X[row_indices]
    return X_aligned


def check_reference_splits(manifest, train_ids, val_ids):
    """Check that Expression train/validation cells match the manifest."""
    required_columns = {"cell_id", "split"}
    if not required_columns.issubset(manifest.columns):
        raise KeyError(f"Manifest requires columns: {sorted(required_columns)}")

    if manifest["cell_id"].isna().any():
        raise ValueError("Manifest contains missing cell IDs")

    manifest = manifest.copy()
    manifest["cell_id"] = manifest["cell_id"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["split"] = manifest["split"].replace(
        {"val": "validation", "valid": "validation"}
    )

    if manifest["cell_id"].duplicated().any():
        raise ValueError("Manifest has duplicate cell IDs")
    if len(set(train_ids) & set(val_ids)):
        raise ValueError("Train and validation reference cell IDs overlap")

    split_by_id = {}
    for cell_id, split in zip(manifest["cell_id"], manifest["split"]):
        split_by_id[cell_id] = split

    for split_name, reference_ids in (("train", train_ids), ("validation", val_ids)):
        expected_ids = set()
        for cell_id, split in split_by_id.items():
            if split == split_name:
                expected_ids.add(cell_id)

        actual_ids = set(reference_ids)
        if actual_ids != expected_ids:
            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids
            raise ValueError(
                f"Expression {split_name} split disagrees with finalized manifest: "
                f"missing={len(missing_ids)}, extra={len(extra_ids)}; "
                f"examples missing={list(missing_ids)[:5]}, "
                f"extra={list(extra_ids)[:5]}"
            )

    return manifest


# Train Logistic Regression for each candidate C
def tune_and_score(name, X_train, X_val, y_train, y_val, C_values, stage):
    """Train on train, evaluate on validation, and save every C result."""
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(f"{name}: train/validation feature counts differ")

    results = []
    for C in C_values:
        model = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

        # Record optimizer warnings so we do not select an unconverged fit.
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)

        converged = True
        for warning_message in warning_list:
            if issubclass(warning_message.category, ConvergenceWarning):
                converged = False


        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        loss = log_loss(y_val, y_prob, labels=model.classes_)

        result = {
            "feature_set": name,
            "search_stage": stage,
            "C": float(C),
            "n_train": len(y_train),
            "n_validation": len(y_val),
            "n_features": X_train.shape[1],
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
            "val_balanced_accuracy": balanced_acc,
            "val_log_loss": loss,
            "converged": converged,
        }
        results.append(result)
        print(
            f"{name:23s} [{stage}] C={C:10.6g} "
            f"val_macro_f1={macro_f1:.4f}"
        )

    return results


def choose_best(rows):
    """
    Select the best C based on validation Macro-F1.
    Only converged models are considered.
    If multiple C values have the same Macro-F1, select the smallest C.
    """
    table = pd.DataFrame(rows)
    table = table[table["converged"]]
    if table.empty:
        raise RuntimeError("No converged Logistic Regression candidates")

    table = table.sort_values(
        by=["val_macro_f1", "C"],
        ascending=[False, True],
        kind="stable",
    )
    return table.iloc[0]


def make_fine_grid(coarse_best_C):
    """Search from best_C/10 to best_C*10, excluding coarse grid C's."""
    all_fine_values = np.geomspace(
        coarse_best_C / 10,
        coarse_best_C * 10,
        FINE_POINTS,
    )
    new_fine_values = []

    for C in all_fine_values:
        # Compare on the log scale, keeping the original tolerance.
        already_tested = np.any(
            np.isclose(
                np.log(C),
                np.log(COARSE_C_VALUES),
                atol=1e-10,
                rtol=0,
            )
        )
        if not already_tested:
            new_fine_values.append(C)

    return new_fine_values

# Run the full experiment
def main():
    manifest_path = find_final_manifest()
    manifest = pd.read_csv(manifest_path, dtype={"cell_id": str})

    exp_train, train_ids, y_train = load_npz(EXPRESSION_DIR / "train.npz", "X")
    exp_val, val_ids, y_val = load_npz(EXPRESSION_DIR / "validation.npz", "X")

    if y_train is None or y_val is None:
        raise KeyError("Expression train/validation NPZs need 'labels'")
    if not set(np.unique(y_val)).issubset(set(np.unique(y_train))):
        raise ValueError("Validation contains labels not present in train")

    manifest = check_reference_splits(manifest, train_ids, val_ids)
    manifest_ids = set(manifest["cell_id"])
    print(f"Manifest: {manifest_path}")
    print(f"Reference: train={len(train_ids)}, validation={len(val_ids)}")

    # Load the six feature representations.
    datasets = []
    datasets.append(("Expression", exp_train, exp_val))

    for name, folder_name in DICTIONARY_RUNS.items():
        folder = DICTIONARY_DIR / folder_name
        dict_train, dict_train_ids, _ = load_npz(folder / "train_codes.npz", "vectors")
        dict_val, dict_val_ids, _ = load_npz(folder / "validation_codes.npz", "vectors")

        if set(dict_train_ids) & set(dict_val_ids):
            raise ValueError(f"{name}: train/validation overlap")

        X_train = align_rows(dict_train, dict_train_ids, train_ids, f"{name} train")
        X_val = align_rows(dict_val, dict_val_ids, val_ids, f"{name} validation")
        datasets.append((name, X_train, X_val))

    sc_vectors, sc_ids, _ = load_npz(SCGPT_PATH, "vectors")
    keep = np.isin(sc_ids, list(manifest_ids))
    removed = int((~keep).sum())
    print(f"scGPT cells outside finalized manifest excluded: {removed}")
    if removed != 63:
        print("Warning: The number of excluded scGPT cells is not 63.")

    sc_vectors = sc_vectors[keep]
    sc_ids = sc_ids[keep]
    sc_train = align_rows(sc_vectors, sc_ids, train_ids, "scGPT train")
    sc_val = align_rows(sc_vectors, sc_ids, val_ids, "scGPT validation")
    datasets.append(("scGPT", sc_train, sc_val))

    # Run coarse search and then fine search for every feature set.
    coarse_rows = []
    fine_rows = []
    coarse_winners = {}

    for name, X_train, X_val in datasets:
        print(f"\n=== {name}: {X_train.shape[1]} features ===")

        group_coarse = tune_and_score(
            name, X_train, X_val, y_train, y_val,
            COARSE_C_VALUES, stage="coarse",
        )
        coarse_rows.extend(group_coarse)

        coarse_best = choose_best(group_coarse)
        coarse_winners[name] = {
            "coarse_best_C": float(coarse_best["C"]),
            "coarse_best_macro_f1": float(coarse_best["val_macro_f1"]),
        }

        fine_C_values = make_fine_grid(float(coarse_best["C"]))
        print(
            f"Refining around {name} coarse best C={coarse_best['C']:g} "
            f"with {len(fine_C_values)} new candidates"
        )
        group_fine = tune_and_score(
            name, X_train, X_val, y_train, y_val,
            fine_C_values, stage="fine",
        )
        fine_rows.extend(group_fine)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(coarse_rows).to_csv(
        OUTPUT_DIR / "validation_coarse_runs.csv", index=False
    )
    pd.DataFrame(fine_rows).to_csv(
        OUTPUT_DIR / "validation_fine_runs.csv", index=False
    )

    all_results = pd.DataFrame(coarse_rows + fine_rows)
    all_results.to_csv(OUTPUT_DIR / "validation_all_runs.csv", index=False)

    best_records = []
    for name, X_train, X_val in datasets:
        feature_rows = all_results[all_results["feature_set"] == name]
        best_row = choose_best(feature_rows)
        record = best_row.to_dict()
        record.update(coarse_winners[name])
        best_records.append(record)

    best_results = pd.DataFrame(best_records)
    best_results = best_results.sort_values(
        "val_macro_f1", ascending=False, kind="stable"
    ).reset_index(drop=True)
    best_results.to_csv(OUTPUT_DIR / "validation_best_C.csv", index=False)

    # Print a summary.
    print("\n=== BEST VALIDATION C PER FEATURE SET ===")
    columns_to_show = [
        "feature_set", "C", "search_stage", "coarse_best_C",
        "val_macro_f1", "val_accuracy", "val_balanced_accuracy", "val_log_loss",
    ]
    print(best_results[columns_to_show].to_string(index=False))
    print(f"\nCoarse fits: {len(coarse_rows)}")
    print(f"Fine fits: {len(fine_rows)}")
    print(f"Total fits: {len(all_results)}")

    output_files = [
        "validation_coarse_runs.csv",
        "validation_fine_runs.csv",
        "validation_all_runs.csv",
        "validation_best_C.csv",
    ]
    for filename in output_files:
        print(f"Saved: {OUTPUT_DIR / filename}")
    print("Test set was not loaded or used.")


if __name__ == "__main__":
    main()
