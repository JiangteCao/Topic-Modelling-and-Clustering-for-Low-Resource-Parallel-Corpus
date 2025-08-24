import numpy as np
import matplotlib.pyplot as plt
import umap


def plot_umap_two_sets(
    de_embeddings: np.ndarray,
    hsb_embeddings: np.ndarray,
    n_samples_each: int = 3000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    out_path: str = None,
    title: str = "UMAP of embeddings (German + Upper Sorbian)"
):

    rng = np.random.default_rng(random_state)

    # subsample
    de_idx = rng.choice(de_embeddings.shape[0], size=min(n_samples_each, de_embeddings.shape[0]), replace=False)
    hsb_idx = rng.choice(hsb_embeddings.shape[0], size=min(n_samples_each, hsb_embeddings.shape[0]), replace=False)

    de_sampled = de_embeddings[de_idx]
    hsb_sampled = hsb_embeddings[hsb_idx]

    all_embs = np.concatenate([de_sampled, hsb_sampled], axis=0)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    all_embs_umap = reducer.fit_transform(all_embs)

    n_de = de_sampled.shape[0]

    plt.figure(figsize=(14, 10))
    plt.scatter(all_embs_umap[:n_de, 0], all_embs_umap[:n_de, 1], s=10, label="German")
    plt.scatter(all_embs_umap[n_de:, 0], all_embs_umap[n_de:, 1], s=10, label="Upper Sorbian")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
