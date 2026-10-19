
from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss


# Step 1: Set file paths
ROOT = Path(__file__).resolve().parents[2]

scgpt_path = ROOT / "code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz"
expression_dir = ROOT / "data/processed/baron_classical_v1"


# Step 2: Load scGPT embeddings
with np.load(scgpt_path, allow_pickle=False) as data:
    vectors = data["vectors"]
    cell_ids = data["cell_ids"].astype(str)


# Step 3: Load training and validation labels
with np.load(expression_dir / "train.npz", allow_pickle=False) as data:
    train_ids = data["cell_ids"].astype(str)
    y_train = data["labels"]

with np.load(expression_dir / "validation.npz", allow_pickle=False) as data:
    val_ids = data["cell_ids"].astype(str)
    y_val = data["labels"]


# Step 4: Align scGPT embeddings by cell ID
id_to_row = {cell_id: i for i, cell_id in enumerate(cell_ids)}

train_rows = [id_to_row[cell_id] for cell_id in train_ids]
val_rows = [id_to_row[cell_id] for cell_id in val_ids]

X_train = vectors[train_rows]
X_val = vectors[val_rows]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# Step 5: Try different C values
C_values = [0.01, 0.1, 1, 10, 100, 1000]

for C in C_values:

    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_prob, labels=model.classes_)

    print("\nC =", C)
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Log Loss:", round(loss, 4))
