import pandas as pd

train = pd.read_csv("data/bayes/train.csv")
val = pd.read_csv("data/bayes/val.csv")

print("Training rows:", len(train))
print("Validation rows:", len(val))


features = [
    "LeadTime",
    "ArrivalDateMonth",
    "StaysInWeekNights",
    "StaysInWeekendNights",
    "Adults",
    "Children",
    "ReservedRoomType",
    "Meal",
    "MarketSegment",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "CustomerType",
]

X_train = train[features]
y_train = train["IsCanceled"]

X_val = val[features]
y_val = val["IsCanceled"]

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

categorical_features = [
    "ArrivalDateMonth",
    "ReservedRoomType",
    "Meal",
    "MarketSegment",
    "CustomerType"
]

numerical_features = [
    "LeadTime",
    "StaysInWeekNights",
    "StaysInWeekendNights",
    "Adults",
    "Children",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled"
]

print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))

print("Missing values in training:")
print(X_train.isnull().sum())
print("Missing values in validation:")
print(X_val.isnull().sum())


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_train = encoder.fit_transform(X_train[categorical_features])
encoded_val = encoder.transform(X_val[categorical_features])

print("Encoded training shape:", encoded_train.shape)
print("Encoded validation shape:", encoded_val.shape)


import numpy as np

X_train_ready = np.hstack([
    X_train[numerical_features].to_numpy(),
    encoded_train
])

X_val_ready = np.hstack([
    X_val[numerical_features].to_numpy(),
    encoded_val
])

print("Final training shape:", X_train_ready.shape)
print("Final validation shape:", X_val_ready.shape)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train_ready, y_train)

print("Baseline GaussianNB trained successfully!")

y_pred = model.predict(X_val_ready)

print("First 10 predictions:", y_pred[:10])
print("First 10 actual answers:", y_val.to_numpy()[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_pred)
print("Validation accuracy:", round(accuracy, 4))