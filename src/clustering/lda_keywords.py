from typing import Dict, List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from nltk.corpus import stopwords


def build_combined_stopwords(nlp: "spacy.Language") -> set:
    """
    Combine spaCy and NLTK German stopwords.
    NOTE: assumes the spaCy model and NLTK stopwords are already installed.
    """
    return set(nlp.Defaults.stop_words).union(set(stopwords.words("german")))


def clean_sentences_combined(
    sentences: List[str],
    nlp: "spacy.Language",
    combined_stopwords: set,
) -> List[str]:
    """
    Lemmatize and keep alpha tokens only; filter by POS {NOUN, PROPN, ADJ} and stopwords.
    """
    cleaned = []
    for doc in nlp.pipe(sentences, batch_size=32):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_punct
            and len(token.text) > 2
            and token.lemma_.lower() not in combined_stopwords
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        cleaned.append(" ".join(tokens))
    return cleaned


def extract_lda_topics_by_cluster(
    sentences: List[str],
    labels: List[int],
    nlp: "spacy.Language",
    combined_stopwords: set,
    n_topics: int = 1,
    n_words: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 1000,
    min_df: int = 1,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    For each cluster id in `labels`, fit an LDA topic model on cleaned sentences and
    return top words with (normalized) topic weights.
    """
    df = pd.DataFrame({"sentence": sentences, "cluster_label": labels})
    cluster_topics: Dict[int, List[Tuple[str, float]]] = {}

    for cid in sorted(df["cluster_label"].unique()):
        cluster_sents = df[df["cluster_label"] == cid]["sentence"].tolist()
        if len(cluster_sents) < 3:
            continue

        cleaned = clean_sentences_combined(cluster_sents, nlp=nlp, combined_stopwords=combined_stopwords)

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
        X = vectorizer.fit_transform(cleaned)
        if X.shape[1] == 0:
            cluster_topics[cid] = [("(no valid terms)", 0.0)]
            continue

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        feature_names = vectorizer.get_feature_names_out()

        topic_keywords: List[Tuple[str, float]] = []
        for topic in lda.components_:
            top_indices = topic.argsort()[:-n_words - 1:-1]
            topic = topic / topic.sum()
            top_words = [(feature_names[i], float(topic[i])) for i in top_indices]
            topic_keywords.extend(top_words)

        cluster_topics[cid] = topic_keywords

    return cluster_topics
