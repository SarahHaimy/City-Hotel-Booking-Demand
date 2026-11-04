import pandas as pd
from src.classification.bayes_preprocessing import prepare_bayes_features
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1. Load training and validation datasets
train = pd.read_csv("data/bayes/train.csv")
val = pd.read_csv("data/bayes/val.csv")

print("Training rows:", len(train))
print("Validation rows:", len(val))

# 2. Prepare features using our shared preprocessing function
X_train_ready, X_val_ready, y_train, y_val, encoder = prepare_bayes_features(train, val)

print("Final training shape:", X_train_ready.shape)
print("Final validation shape:", X_val_ready.shape)

# 3. Train baseline Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train_ready, y_train)

print("Baseline GaussianNB trained successfully!")

# 4. Predict cancellation on validation data
y_pred = model.predict(X_val_ready)

print("First 10 predictions:", y_pred[:10])
print("First 10 actual answers:", y_val.to_numpy()[:10])

# 5. Evaluate baseline
accuracy = accuracy_score(y_val, y_pred)
print("Validation accuracy:", round(accuracy, 4))

cancel_prob = model.predict_proba(X_val_ready)[:, 1]

print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.2%}")
print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.2%}")
print(f"F1-score:  {f1_score(y_val, y_pred, zero_division=0):.2%}")
print(f"ROC-AUC:   {roc_auc_score(y_val, cancel_prob):.4f}")

tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
