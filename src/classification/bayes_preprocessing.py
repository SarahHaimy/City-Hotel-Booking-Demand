
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

categorical_features = [
    "ArrivalDateMonth",
    "ReservedRoomType",
    "Meal",
    "MarketSegment",
    "CustomerType",
]

numerical_features = [
    "LeadTime",
    "StaysInWeekNights",
    "StaysInWeekendNights",
    "Adults",
    "Children",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]


def prepare_bayes_features(train, val):
    X_train = train[features]
    X_val = val[features]

    if X_train.isnull().any().any() or X_val.isnull().any().any():
        raise ValueError("Missing values found in selected features")

    y_train = train["IsCanceled"]
    y_val = val["IsCanceled"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    encoded_train = encoder.fit_transform(X_train[categorical_features])
    encoded_val = encoder.transform(X_val[categorical_features])

    X_train_ready = np.hstack([
        X_train[numerical_features].to_numpy(),
        encoded_train
    ])

    X_val_ready = np.hstack([
        X_val[numerical_features].to_numpy(),
        encoded_val
    ])

    return X_train_ready, X_val_ready, y_train, y_val, encoder
