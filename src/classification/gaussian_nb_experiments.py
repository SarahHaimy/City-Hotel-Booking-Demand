import pandas as pd

train = pd.read_csv("data/bayes/train.csv")
val = pd.read_csv("data/bayes/val.csv")

print("Training rows:", len(train))
print("Validation rows:", len(val))

from src.classification.bayes_preprocessing import prepare_bayes_features

X_train_ready, X_val_ready, y_train, y_val, encoder = prepare_bayes_features(train, val)

print("Training shape:", X_train_ready.shape)
print("Validation shape:", X_val_ready.shape)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4]

for value in smoothing_values:
    model = GaussianNB(var_smoothing=value)
    model.fit(X_train_ready, y_train)

    predictions = model.predict(X_val_ready)
    score = f1_score(y_val, predictions)

    print("Smoothing:", value, "F1-score:", round(score, 4))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

best_model = GaussianNB(var_smoothing=1e-5)
best_model.fit(X_train_ready, y_train)

y_pred_tuned = best_model.predict(X_val_ready)
cancel_prob_tuned = best_model.predict_proba(X_val_ready)[:, 1]

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    predictions = (cancel_prob_tuned >= threshold).astype(int)

    print(
        "Threshold:", threshold,
        "Precision:", round(precision_score(y_val, predictions, zero_division=0), 4),
        "Recall:", round(recall_score(y_val, predictions, zero_division=0), 4),
        "F1:", round(f1_score(y_val, predictions, zero_division=0), 4)
    )

predictions_60 = (cancel_prob_tuned >= 0.6).astype(int)

print("Confusion matrix at 60% threshold:")
print(confusion_matrix(y_val, predictions_60))    

print("\nTuned GaussianNB Results:")
print("Accuracy:", round(accuracy_score(y_val, y_pred_tuned), 4))
print("Precision:", round(precision_score(y_val, y_pred_tuned), 4))
print("Recall:", round(recall_score(y_val, y_pred_tuned), 4))
print("F1-score:", round(f1_score(y_val, y_pred_tuned), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, cancel_prob_tuned), 4))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred_tuned))
