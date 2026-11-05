"""
Run the baseline K-means clustering with different PCA configurations.

    python -m src.clustering.k_means.cluster --pca none
    python -m src.clustering.k_means.cluster --pca 0.90 --k 3
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
from src.clustering.k_means.audit_functions import audit_features

SEED = 42
ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "outputs" / "clustering" / "figures"
TABLE_DIR = ROOT / "outputs" / "clustering" / "tables"

def evaluate(X, model, train_time):
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

def cluster_profile(X_raw, labels, fs):
    # statistical summary of each cluster (mean/mode) for numeric, binary, and categorical features.
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
    # second leakage check, basically checks if clusters are correlated with the outcome (IsCanceled) after the fact
    tmp = pd.DataFrame({
        "cluster": labels,
        "IsCanceled": df_clean.loc[X_raw.index, "IsCanceled"].values,
    })
    return tmp.groupby("cluster")["IsCanceled"].agg(["mean", "size"]).round(3)


def plot_pca_2d(X, labels, path):
    # 2D projection for visualization only; does not affect the clustering
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


# training (no longer useful, might be deleted or reused for tuned)
# def train_kmeans(X, params):
#     """Reused by the Milestone 2 tuning loop: call it with different params."""
#     t0 = time.perf_counter()
#     model = KMeans(**params).fit(X)
#     return model, time.perf_counter() - t0


def run_baseline(k=3, n_components = None):
    for d in (ARTIFACTS_DIR, FIG_DIR, TABLE_DIR):
         d.mkdir(parents=True, exist_ok=True)

    # file-name tag so the no-PCA and PCA runs don't overwrite each other
    tag = "nopca" if n_components is None else f"pca{int(round(n_components * 100))}"

    # data: shared cleaning, NO train/test split (unsupervised task)
    df_clean, report = clean_data(load_h2())
    print(report.to_string(index=False))

    # full training pipeline (features -> PCA -> KMeans)
    X_raw, fs = get_feature_frame(df_clean)
    pipe = build_cluster_pipeline(fs, k=k, n_components=n_components)
    t0 = time.perf_counter()
    pipe.fit(X_raw)
    secs = time.perf_counter() - t0

    model = pipe.named_steps["kmeans"]
    X = pipe[:-1].transform(X_raw)       # the matrix KMeans actually clustered on
    print(f"\nFeatures: {X_raw.shape[1]} raw -> "
          f"{X.shape[1]} model columns, {X.shape[0]} rows")

    # unsupervised metrics (computed in the space KMeans clustered in)
    metrics = evaluate(X, model, secs)
    pca_info = {"pca_n_components": n_components}
    if n_components is not None:
        pca = pipe.named_steps["pca"]
        pca_info["pca_components_kept"] = int(pca.n_components_)
        pca_info["pca_variance_kept"] = float(pca.explained_variance_ratio_.sum())
    metrics.update({"config": "baseline", "params": model.get_params(), **pca_info})

    # interpretation
    profile = cluster_profile(X_raw, model.labels_, fs)
    print("\nCluster profile:\n", profile.to_string())
    profile.to_csv(TABLE_DIR / f"kmeans_baseline_{tag}_profile.csv")

    cancel = posthoc_cancel_rate(df_clean, X_raw, model.labels_)
    print("\nPost-hoc cancel rate per cluster (leakage check only):\n", cancel)
    cancel.to_csv(TABLE_DIR / f"kmeans_baseline_{tag}_cancel_rate.csv")

    metrics["pca_explained_variance_2d"] = plot_pca_2d(
        X, model.labels_, FIG_DIR / f"kmeans_baseline_{tag}_pca.png")

    print(json.dumps(metrics, indent=2))

    # save results and artifacts
    with open(TABLE_DIR / f"kmeans_baseline_{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.Series(model.labels_, index=X_raw.index, name="cluster").to_csv(
        TABLE_DIR / f"kmeans_baseline_{tag}_labels.csv")
    joblib.dump(pipe, ARTIFACTS_DIR / f"kmeans_baseline_{tag}_pipeline.joblib")
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")
    return pipe, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pca", default="none", help="'none' or a float like 0.90")
    args = ap.parse_args()
    nc = None if args.pca == "none" else float(args.pca)
    run_baseline(k=args.k, n_components=nc)