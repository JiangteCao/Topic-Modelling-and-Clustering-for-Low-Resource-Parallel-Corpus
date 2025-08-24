import numpy as np
from scipy import cluster as clst
from sklearn.decomposition import PCA


def cluster_based(representations: np.ndarray, n_cluster: int, n_pc: int, hidden_size: int = 768) -> np.ndarray:
    """
    Cluster-Based Isotropy Enhancement (CBIE).
    Adapted from: https://github.com/kathyhaem/outliers (src/post_processing.py).

    Parameters
    ----------
    representations : np.ndarray
        Input embedding matrix [n_samples, hidden_size].
    n_cluster : int
        Number of clusters.
    n_pc : int
        Number of principal components removed per cluster.
    hidden_size : int
        Embedding dimensionality (default=768).

    Returns
    -------
    post_rep : np.ndarray
        Isotropy-enhanced embeddings.
    """
    centroids, labels = clst.vq.kmeans2(representations, n_cluster, minit="points")
    cluster_means = []
    for i in range(max(labels) + 1):
        cluster_means.append(np.mean(representations[labels == i], axis=0, keepdims=True))

    zero_mean_representations = [
        representations[i] - cluster_means[labels[i]]
        for i in range(len(representations))
    ]

    cluster_representations = {i: {} for i in range(n_cluster)}
    for j, rep in enumerate(zero_mean_representations):
        cluster_representations[labels[j]][j] = rep

    cluster_representations2 = [
        list(cluster_representations[i].values()) for i in range(n_cluster)
    ]

    post_rep = np.zeros_like(representations)
    for i in range(n_cluster):
        if not cluster_representations2[i]:
            continue
        model = PCA()
        model.fit(np.array(cluster_representations2[i]))
        component = model.components_

        for index, vec in cluster_representations[i].items():
            sum_vec = np.zeros((hidden_size,))
            for j in range(min(n_pc, component.shape[0])):
                sum_vec += np.dot(vec, component[j]) * component[j]
            post_rep[index] = vec - sum_vec

    return post_rep
