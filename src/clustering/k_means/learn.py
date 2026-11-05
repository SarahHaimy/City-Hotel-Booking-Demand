"""Baseline K-means (sklearn) for Task C: clustering.

Run from the project root:
    python -m src.clustering.k_means.cluster
    python -m src.clustering.k_means.cluster --feature-set B --k 3
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from src.data.loader import load_h2
from src.data.data_cleaning import clean_data
from src.clustering.k_means.features import get_feature_frame
from src.clustering.k_means.pipeline import build_cluster_pipeline
# from src.clustering.k_means.audit_functions import audit_features   # enable once the file has content

SEED = 42
ROOT = Path(__file__).resolve().parents[3]          # src/clustering/k_means/cluster.py -> project root
ARTIFACTS_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "outputs" / "clustering" / "figures"
TABLE_DIR = ROOT / "outputs" / "clustering" / "tables"


# ------------------------------ evaluation ---------------------------------
def evaluate(X, model, train_time):
    """Unsupervised metrics only. IsCanceled is never used here."""
    labels = model.labels_
    sizes = np.bincount(labels, minlength=model.n_clusters)
    return {
        "k": int(model.n_clusters),
        "silhouette": float(silhouette_score(X, labels, sample_size=10_000,
                                             random_state=SEED)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "min_cluster_pct": float(sizes.min() / len(labels)),
        "cluster_sizes": sizes.tolist(),
        "train_time_sec": round(train_time, 3),
    }


# ----------------------------- interpretation ------------------------------
def cluster_profile(X_raw, labels, fs):
    """Per-cluster means (original units), modes of categoricals, and sizes."""
    df = X_raw.copy()
    df["cluster"] = labels
    g = df.groupby("cluster")
    profile = g[fs["num"] + fs["bin"]].mean().round(2)
    for c in fs["cat"]:
        profile[f"{c}_mode"] = g[c].agg(lambda s: s.mode().iloc[0])
    profile["size"] = g.size()
    profile["size_pct"] = (profile["size"] / len(df) * 100).round(1)
    return profile


def posthoc_cancel_rate(df_clean, X_raw, labels):
    """Leakage sanity check ONLY. Never used to choose features or K."""
    tmp = pd.DataFrame({
        "cluster": labels,
        "IsCanceled": df_clean.loc[X_raw.index, "IsCanceled"].values,
    })
    return tmp.groupby("cluster")["IsCanceled"].agg(["mean", "size"]).round(3)


def plot_pca_2d(X, labels, path):
    """2D projection for VISUALIZATION only; does not affect the clustering."""
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    idx = np.random.default_rng(SEED).choice(len(Z), size=min(15_000, len(Z)),
                                             replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z[idx, 0], Z[idx, 1], c=labels[idx], s=4, alpha=0.5, cmap="tab10")
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("Baseline K-means clusters (PCA projection)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return ev.tolist()


# -------------------------------- training ---------------------------------
# def train_kmeans(X, params):
#     """Reused by the Milestone 2 tuning loop: call it with different params."""
#     t0 = time.perf_counter()
#     model = KMeans(**params).fit(X)
#     return model, time.perf_counter() - t0


def run_baseline(feature_set="B", k=3):
    for d in (ARTIFACTS_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data: shared cleaning, NO train/test split (unsupervised task)
    df_clean, report = clean_data(load_h2())
    print(report.to_string(index=False))

    # 2. features (the leakage guard lives inside get_feature_frame)
    X_raw, fs = get_feature_frame(df_clean, set_name=feature_set)
    # audit_features(X_raw, fs)

    # 3-4. pipeline: prep -> (PCA) -> KMeans
    pipe = build_cluster_pipeline(fs, k=k, n_components=0.9)   # 0.9 = 90% variance
    t0 = time.perf_counter()
    pipe.fit(X_raw)
    secs = time.perf_counter() - t0

    model = pipe.named_steps["kmeans"]
    X = pipe[:-1].transform(X_raw)       # the matrix KMeans actually clustered on
    params = model.get_params()
    print(f"\nFeature set {feature_set}: {X_raw.shape[1]} raw -> "
          f"{X.shape[1]} model columns, {X.shape[0]} rows")
    
    # 5. unsupervised metrics
    metrics = evaluate(X, model, secs)
    metrics.update({"feature_set": feature_set, "config": "baseline", "params": params})
    print(json.dumps(metrics, indent=2))

    # 6. interpretation
    profile = cluster_profile(X_raw, model.labels_, fs)
    print("\nCluster profile:\n", profile.to_string())
    profile.to_csv(TABLE_DIR / "kmeans_baseline_profile.csv")

    cancel = posthoc_cancel_rate(df_clean, X_raw, model.labels_)
    print("\nPost-hoc cancel rate per cluster (leakage check only):\n", cancel)
    cancel.to_csv(TABLE_DIR / "kmeans_baseline_cancel_rate.csv")

    metrics["pca_explained_variance_2d"] = plot_pca_2d(
        X, model.labels_, FIG_DIR / "kmeans_baseline_pca.png")
    
    metrics.update({"feature_set": feature_set, "config": "baseline", "params": params,
                "pca_n_components": 0.90,
                "pca_components_kept": int(pipe.named_steps["pca"].n_components_),
                "pca_variance_kept": float(pipe.named_steps["pca"].explained_variance_ratio_.sum())})

    # 7. save results and artifacts
    with open(TABLE_DIR / "kmeans_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.Series(model.labels_, index=X_raw.index, name="cluster").to_csv(
        TABLE_DIR / "kmeans_baseline_labels.csv")
    joblib.dump(pipe, ARTIFACTS_DIR / "kmeans_baseline_pipeline.joblib")
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")
    return model, prep, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="B", choices=["A", "B", "C"])
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    run_baseline(args.feature_set, args.k)