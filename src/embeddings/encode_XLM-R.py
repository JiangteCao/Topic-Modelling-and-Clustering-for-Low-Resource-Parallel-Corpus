from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

# load model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

#（mean pooling）embedding function
def embed_sentences(sentences):
    embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for sent in tqdm(sentences):
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)
        attention_mask = inputs['attention_mask'].squeeze(0)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        masked_hidden = last_hidden * mask_expanded
        sentence_embedding = masked_hidden.sum(0) / attention_mask.sum()
        embeddings.append(sentence_embedding.cpu().numpy())
    return np.array(embeddings)

# generate two languages' vector
hsb_embeddings = embed_sentences(sorbian_sentences)
np.save("/content/drive/MyDrive/Colab Notebooks/xlmr_hsb.npy", hsb_embeddings)

de_embeddings = embed_sentences(german_sentences)
np.save("/content/drive/MyDrive/Colab Notebooks/xlmr_de.npy", de_embeddings)
