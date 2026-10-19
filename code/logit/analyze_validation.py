from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from run_all_validation_coarse_to_fine import (
    EXPRESSION_DIR,
    DICTIONARY_DIR,
    DICTIONARY_RUNS,
    SCGPT_PATH,
    find_final_manifest,
    check_reference_splits,
    load_npz,
    align_rows
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results/logit"

OUTPUT_DIR = RESULTS_DIR / "validation_analysis"
CM_DIR = OUTPUT_DIR / "confusion_matrices"

def main():
    # Load training and validation features
    manifest = pd.read_csv(find_final_manifest(), dtype={"cell_id": str})
    # Expression
    exp_train, train_ids, y_train = load_npz(EXPRESSION_DIR / "train.npz", "X")
    exp_val, val_ids, y_val = load_npz(EXPRESSION_DIR / "validation.npz", "X")

    if y_train is None or y_val is None:
        raise ValueError("Expression labels are missing.")
    if not set(np.unique(y_val)).issubset(set(np.unique(y_train))):
        raise ValueError("Validation has unseen labels.")

    manifest = check_reference_splits(manifest, train_ids, val_ids)
    datasets = [("Expression", exp_train, exp_val)]

    # Dictionary Learning
    for name, folder_name in DICTIONARY_RUNS.items():
        folder = DICTIONARY_DIR / folder_name
        X_train, ids_train, _ = load_npz(folder / "train_codes.npz", "vectors")
        X_val, ids_val, _ = load_npz(folder / "validation_codes.npz", "vectors")

        if set(ids_train) & set(ids_val):
            raise ValueError(f"{name}: overlapping splits")

        X_train = align_rows(X_train, ids_train, train_ids, name)
        X_val = align_rows(X_val, ids_val, val_ids, name)
        datasets.append((name, X_train, X_val))

    # scGPT
    sc_vectors, sc_ids, _ = load_npz(SCGPT_PATH, "vectors")
    keep = np.isin(sc_ids, list(manifest["cell_id"]))
    print("Excluded scGPT cells:", int((~keep).sum()))

    sc_vectors = sc_vectors[keep]
    sc_ids = sc_ids[keep]
    sc_train = align_rows(sc_vectors, sc_ids, train_ids, "scGPT train")
    sc_val = align_rows(sc_vectors, sc_ids, val_ids, "scGPT validation")
    datasets.append(("scGPT", sc_train, sc_val))

    # Read the previously selected best C values
    best_results = pd.read_csv(RESULTS_DIR / "validation_best_C.csv")
    best_results = best_results.set_index("feature_set")
    CM_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    # Analyze each feature set
    for name, X_train, X_val in datasets:
        print(f"\nAnalyzing {name}")
        best_C = float(best_results.loc[name, "C"])
        # Train using the selected best C
        model = LogisticRegression(
            C=best_C, solver="lbfgs", max_iter=5000, random_state=42
        )
        model.fit(X_train, y_train)
        # Predict validation labels
        y_pred = model.predict(X_val)
        cell_types = model.classes_

        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, labels=cell_types, zero_division=0
        )
        for i in range(len(cell_types)):
            all_metrics.append({
                "feature_set": name,
                "cell_type": cell_types[i],
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": support[i],
            })

        # Create confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=cell_types)
        cm_df = pd.DataFrame(cm, index=cell_types, columns=cell_types)
        cm_df.index.name = "true_cell_type"
        cm_df.to_csv(CM_DIR / f"confusion_matrix_{name}.csv")

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cell_types)
        size = max(9, len(cell_types) * 0.7)
        fig, ax = plt.subplots(figsize=(size, size))
        display.plot(ax=ax, cmap="Blues", xticks_rotation=90,
                     values_format="d", colorbar=False)
        ax.set_title(f"{name} - Validation Confusion Matrix")
        fig.tight_layout()
        fig.savefig(CM_DIR / f"confusion_matrix_{name}.png", dpi=200)
        plt.close(fig)

        # Check consistency with previous results
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        previous_f1 = float(best_results.loc[name, "val_macro_f1"])
        print("Best C:", best_C)
        print("Validation Macro-F1:", round(macro_f1, 6))
        print("Matches previous results:",
              np.isclose(macro_f1, previous_f1, atol=1e-8, rtol=0))

    # Save all per-class metrics
    pd.DataFrame(all_metrics).to_csv(
        OUTPUT_DIR / "validation_per_class_metrics.csv", index=False
    )
    print("\nValidation analysis completed.")
    print("Results saved to:", OUTPUT_DIR)
    print("Test set was not loaded or used.")


if __name__ == "__main__":
    main()