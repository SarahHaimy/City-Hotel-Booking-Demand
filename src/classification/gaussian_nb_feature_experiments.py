
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from src.classification.bayes_preprocessing import (
    features,
    categorical_features,
    numerical_features
)

# 1. Load our existing datasets
train = pd.read_csv("data/bayes/train.csv")
val = pd.read_csv("data/bayes/val.csv")

# 2. Remove MarketSegment from our features
selected_features = [
    f for f in features if f != "MarketSegment"
]

experiment_categorical = [
    f for f in categorical_features if f != "MarketSegment"
]

print("Original features:", len(features))
print("Experiment features:", len(selected_features))
print("Removed feature: MarketSegment")

# 3. Separate inputs and answers
X_train = train[selected_features]
X_val = val[selected_features]

y_train = train["IsCanceled"]
y_val = val["IsCanceled"]

# 4. Encode the remaining categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_train = encoder.fit_transform(X_train[experiment_categorical])
encoded_val = encoder.transform(X_val[experiment_categorical])

# 5. Combine numerical and encoded features
X_train_ready = np.hstack([
    X_train[numerical_features].to_numpy(),
    encoded_train
])

X_val_ready = np.hstack([
    X_val[numerical_features].to_numpy(),
    encoded_val
])

print("Training shape:", X_train_ready.shape)
print("Validation shape:", X_val_ready.shape)

# 6. Train GaussianNB with our selected smoothing value
model = GaussianNB(var_smoothing=1e-5)
model.fit(X_train_ready, y_train)

# 7. Apply our selected 60% threshold
cancel_prob = model.predict_proba(X_val_ready)[:, 1]
predictions = (cancel_prob >= 0.6).astype(int)

# 8. Evaluate the experiment
print("\nResults WITHOUT MarketSegment:")
print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
print("Precision:", round(precision_score(y_val, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, predictions, zero_division=0), 4))
print("F1-score:", round(f1_score(y_val, predictions, zero_division=0), 4))

print("Confusion matrix:")
print(confusion_matrix(y_val, predictions))
