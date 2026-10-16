from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss


# Step 1: Load Dictionary Learning data

ROOT = Path(__file__).resolve().parents[2]

data_dir = ROOT / "results/dict-learning/baron_grid_v1/k32_a5.0_s42_e10"

train = np.load(data_dir / "train_codes.npz")
val = np.load(data_dir / "validation_codes.npz")

X_train = train["vectors"]
y_train = train["labels"]

X_val = val["vectors"]
y_val = val["labels"]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# Step 2: Try different C values
C_values = [0.01, 0.1, 1, 10]

for C in C_values:

    # Train Logistic Regression
    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_prob, labels=model.classes_)

    # Print results
    print("\nC =", C)
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Log Loss:", round(loss, 4))


# Step 3: Close data files
train.close()
val.close()
