
import pandas as pd

from src.classification.bayes_preprocessing import prepare_bayes_features
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1. Load the same training and validation datasets
train = pd.read_csv("data/bayes/train.csv")
val = pd.read_csv("data/bayes/val.csv")

# 2. Apply our shared preprocessing
X_train_ready, X_val_ready, y_train, y_val, encoder = prepare_bayes_features(train, val)

# 3. Train using our selected smoothing value
model = GaussianNB(var_smoothing=1e-5)
model.fit(X_train_ready, y_train)

# 4. Predict cancellation probabilities
cancel_prob = model.predict_proba(X_val_ready)[:, 1]

# 5. Apply our selected threshold
selected_threshold = 0.6
y_pred = (cancel_prob >= selected_threshold).astype(int)

# 6. Evaluate on validation data
print("Selected var_smoothing:", 1e-5)
print("Selected threshold:", selected_threshold)

print("Accuracy:", round(accuracy_score(y_val, y_pred), 4))
print("Precision:", round(precision_score(y_val, y_pred), 4))
print("Recall:", round(recall_score(y_val, y_pred), 4))
print("F1-score:", round(f1_score(y_val, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, cancel_prob), 4))

print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))


# Analyze false cancellation predictions
analysis = val.copy()
analysis["Predicted"] = y_pred

# Select bookings that were NOT canceled
not_canceled = analysis[analysis["IsCanceled"] == 0]

# Calculate false alarms for each market segment
results = not_canceled.groupby("MarketSegment")["Predicted"].agg(
    TotalBookings="count",
    FalseAlarms="sum"
)

results["FalseAlarmRate"] = (
    results["FalseAlarms"] / results["TotalBookings"]
)

print("\nFalse alarms by MarketSegment:")
print(results.sort_values("FalseAlarmRate", ascending=False))

print("\nActual cancellation rates by MarketSegment:")

print(
    val.groupby("MarketSegment")["IsCanceled"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
)
analysis["Predicted"] = y_pred

print("\nActual vs predicted cancellation rates:")
print(
    analysis.groupby("MarketSegment")
    .agg(
        ActualRate=("IsCanceled", "mean"),
        PredictedRate=("Predicted", "mean")
    )
)