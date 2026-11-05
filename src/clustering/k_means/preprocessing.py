from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from src.clustering.k_means.audit_functions import audit_features
from src.clustering.k_means.features import get_feature_frame
import numpy as np

def build_preprocessor(fs: dict) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("log",   FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    transformers = [
        ("num", num_pipe, fs["num"]),
        ("bin", "passthrough", fs["bin"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", 
         min_frequency=0.01, sparse_output=False), fs["cat"]),
    ]
    return ColumnTransformer(transformers)

