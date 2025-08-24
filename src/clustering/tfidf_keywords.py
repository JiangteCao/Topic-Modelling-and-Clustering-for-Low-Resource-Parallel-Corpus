# src/text/keywords.py
from typing import Dict, List, Tuple
import spacy
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy German model (make sure 'de_core_news_sm' is installed)
_nlp = spacy.load("de_core_news_sm")

# Build combined German stopword set from spaCy + NLTK
_spacy_stopwords = _nlp.Defaults.stop_words
_nltk_stopwords = set(stopwords.words("german"))
_COMBINED_STOPWORDS = _spacy_stopwords.union(_nltk_stopwords)


def clean_sentences_combined(sentences: List[str]) -> List[str]:
    """
    Lemmatize and filter tokens: keep alphabetic tokens with length > 2,
    exclude stopwords, keep POS in {NOUN, PROPN, ADJ}.
    Returns a list of cleaned strings.
    """
    cleaned: List[str] = []
    for doc in _nlp.pipe(sentences, batch_size=32):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and len(token.text) > 2
            and token.lemma_.lower() not in _COMBINED_STOPWORDS
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        cleaned.append(" ".join(tokens))
    return cleaned


def extract_keywords_by_cluster(
    df: pd.DataFrame,
    text_col: str = "sentence",
    cluster_col: str = "cluster_label",
    top_n: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 1000,
    min_df: int = 1,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    For each cluster, compute TF-IDF over cleaned texts and return top-N terms.

    Returns:
        dict: {cluster_id: [(term, score), ...]}
    """
    cluster_keywords: Dict[int, List[Tuple[str, float]]] = {}
    grouped = df.groupby(cluster_col)[text_col].apply(list)

    for cluster_id, sentences in grouped.items():
        cleaned = clean_sentences_combined(sentences)
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
        X = vectorizer.fit_transform(cleaned)
        if X.shape[1] == 0:
            cluster_keywords[cluster_id] = []
            continue

        scores = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:top_n]
        cluster_keywords[cluster_id] = [(vocab[i], float(round(scores[i], 3))) for i in top_idx]

    return cluster_keywords
