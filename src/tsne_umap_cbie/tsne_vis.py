import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def smart_sample(
    embeddings: np.ndarray,
    n_samples: int = 3000,
    n_clusters: int = 7,
    seed: int = 42,
) -> np.ndarray:
    """Cluster-balanced sampling of indices from embeddings."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    pca = PCA(n_components=min(50, embeddings.shape[1]))
    emb_pca = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(emb_pca)

    rng = np.random.default_rng(seed)
    sampled_indices = []
    total = len(embeddings)

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_ids == cluster_id)[0]
        n = len(cluster_indices)
        if n == 0:
            continue
        n_to_sample = int(np.round(n_samples * n / total))
        n_to_sample = min(n_to_sample, n)
        if n_to_sample > 0:
            chosen = rng.choice(cluster_indices, size=n_to_sample, replace=False)
            sampled_indices.extend(chosen.tolist())

    if len(sampled_indices) < n_samples:
        remaining = n_samples - len(sampled_indices)
        rest_pool = np.setdiff1d(np.arange(total), np.array(sampled_indices), assume_unique=False)
        if remaining > 0 and len(rest_pool) >= remaining:
            extra = rng.choice(rest_pool, size=remaining, replace=False)
            sampled_indices.extend(extra.tolist())

    return np.array(sampled_indices[:n_samples])


def tsne_two_corpora(
    de_path: str,
    hsb_path: str,
    n_samples_each: int = 3000,
    pca_dims: int = 50,
    tsne_dims: int = 2,
    tsne_seed: int = 42,
    out_png: str = "tsne_3000_each.png",
    title: str = "t-SNE of embeddings (German + Upper Sorbian)",
):
    """Load two .npy embedding arrays, sample, reduce (PCA+TSNE), and plot."""
    if not os.path.exists(de_path):
        raise FileNotFoundError(de_path)
    if not os.path.exists(hsb_path):
        raise FileNotFoundError(hsb_path)

    de_embeddings = np.load(de_path)
    hsb_embeddings = np.load(hsb_path)

    de_idx = smart_sample(de_embeddings, n_samples=n_samples_each)
    hsb_idx = smart_sample(hsb_embeddings, n_samples=n_samples_each)

    de_sampled = de_embeddings[de_idx]
    hsb_sampled = hsb_embeddings[hsb_idx]

    all_embs = np.concatenate([de_sampled, hsb_sampled], axis=0)

    pca = PCA(n_components=min(pca_dims, all_embs.shape[1]))
    all_embs_pca = pca.fit_transform(all_embs)

    tsne = TSNE(n_components=tsne_dims, random_state=tsne_seed, init="pca")
    all_embs_tsne = tsne.fit_transform(all_embs_pca)

    n_de = len(de_sampled)

    plt.figure(figsize=(14, 10))
    plt.scatter(all_embs_tsne[:n_de, 0], all_embs_tsne[:n_de, 1], s=10, label="German")
    plt.scatter(all_embs_tsne[n_de:, 0], all_embs_tsne[n_de:, 1], s=10, label="Upper Sorbian")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
