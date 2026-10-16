
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

# ------------------------------------------------------
# 1. Paths and experiment settings
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # code/logit/run_all_validation.py

EXPRESSION_DIR = ROOT / "data/processed/baron_classical_v1"
DICTIONARY_DIR = ROOT / "results/dict-learning/baron_grid_v1"
SCGPT_PATH = (
    ROOT
    / "code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz"
)
OUTPUT_DIR = ROOT / "results/logit"

# Two-stage search: same first-stage grid for all representations.
COARSE_C_VALUES = np.array([0.01, 0.1, 1, 10, 100, 1000], dtype=float)
FINE_POINTS = 21  # 21 log-spaced candidates between best C / 10 and best C * 10

DICTIONARY_RUNS = {
    "Dictionary_K16_alpha1": "k16_a1.0_s42_e10",
    "Dictionary_K16_alpha5": "k16_a5.0_s42_e10",
    "Dictionary_K32_alpha1": "k32_a1.0_s42_e10",
    "Dictionary_K32_alpha5": "k32_a5.0_s42_e10",
}


def find_final_manifest():
    """Locate the finalized manifest; fail instead of guessing if ambiguous."""
    preferred = [
        EXPRESSION_DIR / "split_manifest.csv",
        EXPRESSION_DIR / "baron_split_manifest.csv",
        ROOT / "data/processed/baron_split_manifest.csv",
        ROOT / "data/processed/split_manifest.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path

    matches = sorted(
        set(ROOT.glob("data/**/baron_split_manifest.csv"))
        | set(ROOT.glob("data/**/split_manifest.csv"))
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "Finalized split manifest not found. Set its path in "
            "find_final_manifest() after locating baron_split_manifest.csv "
            "or split_manifest.csv in your repository."
        )
    raise RuntimeError(
        "Several split manifests found; explicitly select the finalized one: "
        + ", ".join(str(p) for p in matches)
    )


def string_ids(values):
    """Decode cell IDs consistently (including NPZ byte strings)."""
    a = np.asarray(values).reshape(-1)
    return np.array([
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in a
    ], dtype=str)


def load_npz(path, feature_key):
    """Return (feature matrix, cell IDs, optional labels) with strict checks."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    with np.load(path, allow_pickle=True) as data:
        # Only load NPZ files from trusted project/team sources.
        required = {feature_key, "cell_ids"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing NPZ keys: {sorted(missing)}")
        X = data[feature_key]
        ids = string_ids(data["cell_ids"])
        labels = data["labels"] if "labels" in data.files else None

    if X.ndim != 2 or len(ids) != X.shape[0]:
        raise ValueError(f"Bad X / cell_ids dimensions in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate cell_ids found in {path}")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError(f"Non-numeric or non-finite feature values in {path}")
    if labels is not None and len(labels) != len(ids):
        raise ValueError(f"labels and cell_ids lengths differ in {path}")
    return X, ids, labels


def align_rows(X, source_ids, target_ids, description):
    """Reorder a feature matrix to match the exact reference cell-ID order."""
    lookup = {cell_id: i for i, cell_id in enumerate(source_ids)}
    missing = [cell_id for cell_id in target_ids if cell_id not in lookup]
    if missing:
        raise ValueError(
            f"{description}: missing {len(missing)} reference cells. "
            f"Example: {missing[:5]}"
        )
    indices = [lookup[cell_id] for cell_id in target_ids]
    return X[indices]


def check_reference_splits(manifest, train_ids, val_ids):
    """Verify that reference splits correspond to the finalized manifest."""
    required = {"cell_id", "split"}
    if not required.issubset(manifest.columns):
        raise KeyError(f"Manifest requires columns: {sorted(required)}")

    if manifest["cell_id"].isna().any():
        raise ValueError("Manifest contains missing cell IDs")
    manifest = manifest.copy()
    manifest["cell_id"] = manifest["cell_id"].astype(str)
    manifest["split"] = (
        manifest["split"].astype(str).str.strip().str.lower()
        .replace({"val": "validation", "valid": "validation"})
    )
    if manifest["cell_id"].duplicated().any():
        raise ValueError("Manifest has duplicate cell IDs")
    if len(set(train_ids) & set(val_ids)):
        raise ValueError("Train and validation reference cell IDs overlap")

    split_by_id = dict(zip(manifest["cell_id"], manifest["split"]))
    for split, ids in (("train", train_ids), ("validation", val_ids)):
        expected = {cid for cid, value in split_by_id.items() if value == split}
        actual = set(ids)
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(
                f"Expression {split} split disagrees with finalized manifest: "
                f"missing={len(missing)}, extra={len(extra)}; "
                f"examples missing={list(missing)[:5]}, extra={list(extra)[:5]}"
            )
    return manifest


def tune_and_score(name, X_train, X_val, y_train, y_val, C_values, stage):
    """Train on train; score each candidate on validation, with no scaling."""
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(f"{name}: train/validation feature counts differ")

    rows = []
    for C in C_values:
        model = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)

        converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        if not converged:
            print(f"WARNING: {name}, C={C:g} did not converge")

        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)
        result = {
            "feature_set": name,
            "search_stage": stage,
            "C": float(C),
            "n_train": len(y_train),
            "n_validation": len(y_val),
            "n_features": X_train.shape[1],
            "val_accuracy": accuracy_score(y_val, pred),
            "val_macro_f1": f1_score(
                y_val, pred, average="macro", zero_division=0
            ),
            "val_balanced_accuracy": balanced_accuracy_score(y_val, pred),
            "val_log_loss": log_loss(y_val, prob, labels=model.classes_),
            "converged": converged,
        }
        rows.append(result)
        print(
            f"{name:23s} [{stage}] C={C:10.6g} "
            f"val_macro_f1={result['val_macro_f1']:.4f}"
        )
    return rows


def choose_best(rows):
    """Best *converged* validation candidate; ties favor smaller C."""
    eligible = pd.DataFrame(rows)
    eligible = eligible.loc[eligible["converged"]]
    if eligible.empty:
        raise RuntimeError("No converged Logistic Regression candidates")
    return eligible.sort_values(
        ["val_macro_f1", "C"],
        ascending=[False, True],
        kind="stable"
    ).iloc[0]


def make_fine_grid(coarse_best_C):
    """One local refinement across +/- one log10 unit of the coarse best."""
    fine = np.geomspace(coarse_best_C / 10, coarse_best_C * 10, FINE_POINTS)
    # Don't waste fits on previously evaluated coarse candidates.
    fine = [c for c in fine if not np.any(
        np.isclose(np.log(c), np.log(COARSE_C_VALUES), atol=1e-10, rtol=0)
    )]
    return fine


def main():
    # ------------------------------------------------------
    # 2. Finalized manifest + reference expression splits
    # ------------------------------------------------------
    manifest_path = find_final_manifest()
    manifest = pd.read_csv(manifest_path, dtype={"cell_id": str})

    exp_train, train_ids, y_train = load_npz(
        EXPRESSION_DIR / "train.npz", "X"
    )
    exp_val, val_ids, y_val = load_npz(
        EXPRESSION_DIR / "validation.npz", "X"
    )
    if y_train is None or y_val is None:
        raise KeyError("Expression train/validation NPZs need 'labels'")
    if not set(np.unique(y_val)).issubset(set(np.unique(y_train))):
        raise ValueError("Validation contains labels not present in train")

    manifest = check_reference_splits(manifest, train_ids, val_ids)
    manifest_ids = set(manifest["cell_id"])
    print(f"Manifest: {manifest_path}")
    print(f"Reference: train={len(train_ids)}, validation={len(val_ids)}")

    # ------------------------------------------------------
    # 3. Load all representations, preserving reference order
    # ------------------------------------------------------
    datasets = [("Expression", exp_train, exp_val)]

    for name, run_folder in DICTIONARY_RUNS.items():
        folder = DICTIONARY_DIR / run_folder
        dict_train, dict_train_ids, _ = load_npz(
            folder / "train_codes.npz", "vectors"
        )
        dict_val, dict_val_ids, _ = load_npz(
            folder / "validation_codes.npz", "vectors"
        )
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
        warnings.warn(
            f"Expected 63 out-of-manifest scGPT cells, found {removed}. "
            "Check file and manifest versions.",
            stacklevel=1,
        )
    sc_vectors = sc_vectors[keep]
    sc_ids = sc_ids[keep]
    sc_train = align_rows(sc_vectors, sc_ids, train_ids, "scGPT train")
    sc_val = align_rows(sc_vectors, sc_ids, val_ids, "scGPT validation")
    datasets.append(("scGPT", sc_train, sc_val))

    # ------------------------------------------------------
    # 4. Coarse-to-fine C tuning, separately for each representation
    # ------------------------------------------------------
    coarse_rows = []
    fine_rows = []
    coarse_winners = {}

    for name, X_train, X_val in datasets:
        print(f"\n=== {name}: {X_train.shape[1]} features ===")
        group_coarse = tune_and_score(
            name, X_train, X_val, y_train, y_val,
            COARSE_C_VALUES, stage="coarse"
        )
        coarse_rows.extend(group_coarse)
        coarse_best = choose_best(group_coarse)
        coarse_winners[name] = {
            "coarse_best_C": float(coarse_best["C"]),
            "coarse_best_macro_f1": float(coarse_best["val_macro_f1"]),
        }

        grid_fine = make_fine_grid(float(coarse_best["C"]))
        print(
            f"Refining around {name} coarse best C={coarse_best['C']:g} "
            f"with {len(grid_fine)} new candidates"
        )
        fine_rows.extend(tune_and_score(
            name, X_train, X_val, y_train, y_val,
            grid_fine, stage="fine"
        ))

    # ------------------------------------------------------
    # 5. Save both search stages and one best C per feature set
    # ------------------------------------------------------
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
    for name in [name for name, _, _ in datasets]:
        feature_rows = all_results[all_results["feature_set"] == name]
        record = choose_best(feature_rows).to_dict()
        record.update(coarse_winners[name])
        best_records.append(record)

    best_results = pd.DataFrame(best_records).sort_values(
        "val_macro_f1", ascending=False, kind="stable"
    ).reset_index(drop=True)
    best_results.to_csv(OUTPUT_DIR / "validation_best_C.csv", index=False)

    print("\n=== BEST VALIDATION C PER FEATURE SET ===")
    print(best_results[
        ["feature_set", "C", "search_stage", "coarse_best_C",
         "val_macro_f1", "val_accuracy", "val_balanced_accuracy",
         "val_log_loss"]
    ].to_string(index=False))
    print(f"\nCoarse fits: {len(coarse_rows)}")
    print(f"Fine fits: {len(fine_rows)}")
    print(f"Total fits: {len(all_results)}")
    for fname in (
        "validation_coarse_runs.csv", "validation_fine_runs.csv",
        "validation_all_runs.csv", "validation_best_C.csv"
    ):
        print(f"Saved: {OUTPUT_DIR / fname}")
    print("Test set was not loaded or used.")


if __name__ == "__main__":
    main()
