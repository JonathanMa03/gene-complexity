from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss


ROOT = Path(__file__).resolve().parents[2]

train = np.load(ROOT / "data/processed/baron_classical_v1/train.npz")
val = np.load(ROOT / "data/processed/baron_classical_v1/validation.npz")

X_train = train["X"]
y_train = train["labels"]

X_val = val["X"]
y_val = val["labels"]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# Try different C values
C_values = [0.01, 0.1, 1, 10]

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

train.close()
val.close()
