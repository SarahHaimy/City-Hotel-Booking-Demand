from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from src.clustering.k_means.audit_functions import *
from src.clustering.k_means.features import *
import numpy as np

def build_preprocessor(fs: dict) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("log",   FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    transformers = [
        ("num", num_pipe, fs["num"]),
        ("bin", "passthrough", fs["bin"]),
    ]
    if fs["cat"]:   # Option A has no categoricals
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
            fs["cat"],
        ))
    return ColumnTransformer(transformers)

# Usage
X_raw, fs = get_feature_frame(df_clean, set_name="B")
audit_features(X_raw, fs)
prep = build_preprocessor(fs)
X = prep.fit_transform(X_raw)