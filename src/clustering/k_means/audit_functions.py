import numpy as np
import pandas as pd

def audit_features(X: pd.DataFrame, fs: dict, corr_thresh: float = 0.8):
    numeric = X[fs["num"] + fs["bin"]]

    print("== Summary ==")
    print(numeric.describe().T[["mean", "std", "min", "max"]])

    print("\n== Skewness (|skew|>1 -> consider log1p) ==")
    print(numeric.skew().round(2))

    print("\n== Near-constant check (share of the most common value) ==")
    print(numeric.apply(lambda s: s.value_counts(normalize=True).iloc[0]).round(3))

    print(f"\n== Highly correlated pairs (|r| > {corr_thresh}) ==")
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    print(upper.stack()[lambda s: s > corr_thresh])

    for c in fs["cat"]:
        print(f"\n== {c}: category shares ==")
        print(X[c].value_counts(normalize=True).round(3))