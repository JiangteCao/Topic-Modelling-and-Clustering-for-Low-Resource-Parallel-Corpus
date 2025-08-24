import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_glot500(device=None):
    """
    Load Glot500-base model and tokenizer from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")
    model = AutoModel.from_pretrained("cis-lmu/glot500-base")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    return tokenizer, model, device


def embed_sentences(sentences, tokenizer, model, device):
    """
    Encode a list of sentences into embeddings using mean pooling.
    """
    embeddings = []
    for sent in tqdm(sentences, desc="Encoding"):
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        masked_hidden = last_hidden * mask_expanded
        sentence_embedding = masked_hidden.sum(0) / attention_mask.sum()

        embeddings.append(sentence_embedding.cpu().numpy())
    return np.array(embeddings)


def embed_and_save(sorbian_sentences, german_sentences, out_dir="./"):
    """
    Generate and save embeddings for Sorbian and German sentences using Glot500.
    """
    tokenizer, model, device = load_glot500()

    print("Encoding Sorbian sentences...")
    hsb_embeddings = embed_sentences(sorbian_sentences, tokenizer, model, device)
    np.save(f"{out_dir}/glot500_hsb.npy", hsb_embeddings)

    print("Encoding German sentences...")
    de_embeddings = embed_sentences(german_sentences, tokenizer, model, device)
    np.save(f"{out_dir}/glot500_de.npy", de_embeddings)

    print(f"Embeddings saved to {out_dir}")
