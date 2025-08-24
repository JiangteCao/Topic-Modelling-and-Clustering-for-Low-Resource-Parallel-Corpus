import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans
