import numpy as np
from laserembeddings import Laser


def embed_and_save(sorbian_sentences, german_sentences, out_dir="./"):
    """
    Generate and save embeddings for Sorbian and German sentences using LASER.

    Parameters
    ----------
    sorbian_sentences : list[str]
        List of Upper Sorbian sentences.
    german_sentences : list[str]
        List of German sentences.
    out_dir : str
        Directory to save embeddings (default: "./").
    """
    # Initialize LASER model (auto-downloads if not already available)
    laser = Laser()

    print("Encoding Sorbian sentences with LASER...")
    embeddings_hsb = laser.embed_sentences(sorbian_sentences, lang="hsb")
    np.save(f"{out_dir}/laser_hsb.npy", embeddings_hsb)

    print("Encoding German sentences with LASER...")
    embeddings_de = laser.embed_sentences(german_sentences, lang="de")
    np.save(f"{out_dir}/laser_de.npy", embeddings_de)

    print(f"LASER embeddings saved to {out_dir}")
