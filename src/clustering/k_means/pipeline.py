# src/clustering/k_means/pipeline.py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.clustering.k_means.preprocessing import build_preprocessor

SEED = 42


def build_cluster_pipeline(fs, k=3, n_components=0.9, kmeans_params=None):
    """n_components=None -> no PCA (baseline); 0.90 -> keep 90% variance."""
    kmeans_params = kmeans_params or dict(init="k-means++", n_init=10, max_iter=300,
                                          tol=1e-4, algorithm="lloyd")
    steps = [("prep", build_preprocessor(fs))]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=SEED)))
    steps.append(("kmeans", KMeans(n_clusters=k, random_state=SEED, **kmeans_params)))
    return Pipeline(steps)