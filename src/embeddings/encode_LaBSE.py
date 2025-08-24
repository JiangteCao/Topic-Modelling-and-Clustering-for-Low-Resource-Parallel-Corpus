import numpy as np
from sentence_transformers import SentenceTransformer


def embed_and_save(
    sorbian_sentences,
    german_sentences,
    model_name: str = "sentence-transformers/LaBSE",
    out_dir: str = "./",
    batch_size: int = 64,
):
    """
    Generate and save embeddings for Sorbian and German sentences.

    Parameters
    ----------
    sorbian_sentences : list[str]
        List of Upper Sorbian sentences.
    german_sentences : list[str]
        List of German sentences.
    model_name : str
        Pretrained embedding model name (default: 'sentence-transformers/LaBSE').
    out_dir : str
        Directory to save embeddings (default: './').
    batch_size : int
        Batch size for encoding (default: 64).
    """
    model = SentenceTransformer(model_name)

    print("Encoding Sorbian sentences...")
    hsb_embeddings = model.encode(sorbian_sentences, show_progress_bar=True, batch_size=batch_size)
    np.save(f"{out_dir}/labse_hsb.npy", hsb_embeddings)

    print("Encoding German sentences...")
    de_embeddings = model.encode(german_sentences, show_progress_bar=True, batch_size=batch_size)
    np.save(f"{out_dir}/labse_de.npy", de_embeddings)

    print(f"Embeddings saved to {out_dir}")
