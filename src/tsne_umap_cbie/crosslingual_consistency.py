# crosslingual_consistency.py
import numpy as np
from sklearn.cluster import KMeans


def cluster_pairwise_consistency(
    de_embeddings: np.ndarray,
    hsb_embeddings: np.ndarray,
    n_clusters: int = 7,
    random_state: int = 42,
):

    n = min(len(de_embeddings), len(hsb_embeddings))
    de = de_embeddings[:n]
    hsb = hsb_embeddings[:n]

    all_embeddings = np.vstack([de, hsb])  # shape: (2n, dim)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(all_embeddings)

    match_count = sum(labels[i] == labels[i + n] for i in range(n))
    agreement_rate = match_count / n if n > 0 else 0.0
    return agreement_rate, match_count, n


if __name__ == "__main__":
    # Example I/O (adjust paths as needed)
    de = np.load("/content/drive/MyDrive/Colab Notebooks/labse/labse_de.npy")
    hsb = np.load("/content/drive/MyDrive/Colab Notebooks/labse/labse_hsb.npy")

    rate, count, n = cluster_pairwise_consistency(de, hsb, n_clusters=7, random_state=42)
    print(f"Before CBIE consistency rate (de vs hsb): {rate:.4f} ({count}/{n})")
