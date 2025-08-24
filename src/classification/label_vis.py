from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def visualize_labelled_sentences(
    final_topic_sentences: Dict[str, List[str]],
    model_name: str = "sentence-transformers/LaBSE",
    k: int = 9,
    save_path: Optional[str] = None,
    show: bool = True,
    random_state: int = 42,
) -> None:
    """
    Build embeddings for labelled sentences, cluster with KMeans, reduce with PCA,
    and draw a 2D scatter plot.

    Parameters
    ----------
    final_topic_sentences : dict[str, list[str]]
        Mapping from topic label to a list of sentences.
    model_name : str
        SentenceTransformer model name. Default: LaBSE.
    k : int
        Number of KMeans clusters.
    save_path : str | None
        If provided, save the figure to this path (e.g., 'pca_clusters.png').
    show : bool
        If True, display the figure via plt.show().
    random_state : int
        Random seed for reproducibility.
    """
    # flatten sentences and labels
    sentences: List[str] = []
    true_labels: List[str] = []
    for topic, sents in final_topic_sentences.items():
        sentences.extend(sents)
        true_labels.extend([topic] * len(sents))

    if len(sentences) == 0:
        raise ValueError("No sentences found in final_topic_sentences.")

    # embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    cluster_ids = kmeans.fit_predict(embeddings)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    # plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=cluster_ids, cmap="tab10", s=25
    )
    plt.title(f"KMeans Clustering of Labelled Sentences (K={k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster ID")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
