
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.classification.bayes_preprocessing import prepare_bayes_features

# 1. Load our existing datasets
train = pd.read_csv("data/bayes/train.csv")
val = pd.read_csv("data/bayes/val.csv")

# 2. Apply the same preprocessing
X_train_ready, X_val_ready, y_train, y_val, encoder = prepare_bayes_features(train, val)

# 3. Create GaussianNB with our selected smoothing value
base_model = GaussianNB(var_smoothing=1e-5)

# 4. Train and calibrate probabilities using training data
calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train_ready, y_train)

# 5. Predict probabilities on validation data
cancel_prob = calibrated_model.predict_proba(X_val_ready)[:, 1]

# 6. Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

best_threshold = None
best_f1 = -1

print("\nCalibration threshold results:")

for threshold in thresholds:
    predictions = (cancel_prob >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | "
        f"F1: {f1:.2%}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# 7. Evaluate the best tested threshold
final_predictions = (cancel_prob >= best_threshold).astype(int)

print("\nBest calibrated model results:")
print("Best threshold:", best_threshold)
print("Accuracy:", round(accuracy_score(y_val, final_predictions), 4))
print("Precision:", round(precision_score(y_val, final_predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, final_predictions, zero_division=0), 4))
print("F1-score:", round(f1_score(y_val, final_predictions, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, cancel_prob), 4))

print("Confusion matrix:")
print(confusion_matrix(y_val, final_predictions))

# 8. Check predictions specifically for Online TA
analysis = val.copy()
analysis["Predicted"] = final_predictions

online_ta = analysis[analysis["MarketSegment"] == "Online TA"]

print("\nOnline TA:")
print("Actual cancellation rate:", round(online_ta["IsCanceled"].mean(), 4))
print("Predicted cancellation rate:", round(online_ta["Predicted"].mean(), 4))
