# clustering_analysis.py
from typing import Iterable, Optional
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

try:
    from kneed import KneeLocator  # optional
except Exception:  # kneed not installed
    KneeLocator = None


def elbow_with_kneedle(
    embeddings: np.ndarray,
    k_range: Iterable[int] = range(2, 21),
    title: str = "Elbow Curve"
) -> Optional[int]:
    """
    Compute inertia over k_range and (optionally) detect the elbow with KneeLocator.

    Returns:
        best_k (int or None): Detected elbow K if kneed is available and an elbow is found.
                              Otherwise returns None.
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=int(k), random_state=42, n_init="auto").fit(embeddings)
        inertias.append(km.inertia_)

    best_k = None
    if KneeLocator is not None:
        kn = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
        best_k = int(kn.elbow) if kn.elbow is not None else None

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, "bo-", label="Inertia")
    if best_k is not None:
        plt.axvline(x=best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.show()

    return best_k


def plot_clusters_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA Cluster Plot"
) -> None:
    """
    Reduce embeddings to 2D using PCA and scatter-plot points colored by labels.
    """
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def clustering_quality_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray
) -> dict:
    """
    Compute a small set of clustering quality metrics given data, labels, and centroids.

    Returns:
        dict with:
            - avg_intra_cluster_distance
            - intra_cluster_variance
            - silhouette_score
            - avg_inter_cluster_distance
            - min_inter_cluster_distance
            - dunn_index
            - db_index
    """
    metrics = {}

    dists = pairwise_distances(X, centroids, metric="euclidean")
    min_d = np.min(dists, axis=1)
    metrics["avg_intra_cluster_distance"] = float(np.mean(min_d))
    metrics["intra_cluster_variance"] = float(np.var(min_d))

    if len(np.unique(labels)) > 1:
        metrics["silhouette_score"] = float(silhouette_score(X, labels))
    else:
        metrics["silhouette_score"] = np.nan

    centroid_d = pairwise_distances(centroids, metric="euclidean")
    np.fill_diagonal(centroid_d, np.nan)
    metrics["avg_inter_cluster_distance"] = float(np.nanmean(centroid_d))
    metrics["min_inter_cluster_distance"] = float(np.nanmin(centroid_d))

    max_intra_diameter = 0.0
    for k in np.unique(labels):
        pts = X[labels == k]
        if len(pts) > 1:
            diam = np.max(pairwise_distances(pts))
            max_intra_diameter = max(max_intra_diameter, float(diam))
    metrics["dunn_index"] = (
        metrics["min_inter_cluster_distance"] / max_intra_diameter
        if max_intra_diameter > 0
        else np.nan
    )

    if len(np.unique(labels)) > 1:
        metrics["db_index"] = float(davies_bouldin_score(X, labels))
    else:
        metrics["db_index"] = np.nan

    return metrics
