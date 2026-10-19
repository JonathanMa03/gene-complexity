import warnings
from sklearn.exceptions import ConvergenceWarning
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

def main(bundle=None, results_dir=None):
    from validation_inputs import load_inputs, default_config
    bundle = load_inputs(default_config()) if bundle is None else bundle
    datasets, y_train, y_val = bundle["datasets"], bundle["y_train"], bundle["y_val"]
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    output_dir = results_dir / "validation_analysis"
    cm_dir = output_dir / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=False)

    # Read the previously selected best C values
    best_results = pd.read_csv(results_dir / "validation_best_C.csv")
    best_results = best_results.set_index("feature_set")
    cm_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    # Analyze each feature set
    for name, X_train, X_val in datasets:
        print(f"\nAnalyzing {name}")
        best_C = float(best_results.loc[name, "C"])
        # Train using the selected best C
        model = LogisticRegression(
            C=best_C, solver="lbfgs", max_iter=5000, random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, y_train)
        # Predict validation labels
        y_pred = model.predict(X_val)
        cell_types = model.classes_

        previous_f1 = float(best_results.loc[name, "val_macro_f1"])
        if not np.isclose(f1_score(y_val, y_pred, average="macro"), previous_f1, atol=1e-8, rtol=0):
            raise ValueError(f"{name}: analysis does not reproduce selected validation score")

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
        cm_df.to_csv(cm_dir / f"confusion_matrix_{name}.csv")

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cell_types)
        size = max(9, len(cell_types) * 0.7)
        fig, ax = plt.subplots(figsize=(size, size))
        display.plot(ax=ax, cmap="Blues", xticks_rotation=90,
                     values_format="d", colorbar=False)
        ax.set_title(f"{name} - Validation Confusion Matrix")
        fig.tight_layout()
        fig.savefig(cm_dir / f"confusion_matrix_{name}.png", dpi=200)
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
        output_dir / "validation_per_class_metrics.csv", index=False
    )
    print("\nValidation analysis completed.")
    print("Results saved to:", output_dir)
    print("Test cells were not used for fitting or scoring.")


if __name__ == "__main__":
    main()