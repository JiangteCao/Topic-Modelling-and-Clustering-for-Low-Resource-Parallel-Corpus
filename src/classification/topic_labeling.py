"""
Topic labeling module for German sentences.
Step 1: Rule-based assignment using predefined keyword sets.
Step 2: Embedding-assisted reassignment of 'Others' sentences using LaBSE and cosine similarity.(e.g. LaBSE)
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from typing import Dict, List


# predefined topic keywords
TOPIC_KEYWORDS: Dict[str, set] = {
    "development": {"neu", "projekt", "zukunft", "arbeit", "entwicklung", "ziel", "stadt"},
    "local_identity": {"bautzen", "lausitz", "sorbisch", "oberlausitz", "freund", "bautzener"},
    "org_politics": {"domowina", "mitglied", "sachsen", "verein", "bundesvorstand"},
    "culture_language": {"sorbisch", "sprache", "kultur", "volk", "institut", "deutsch"},
    "religion_faith": {"gott", "herr", "jesus", "christus", "geist", "himmel"},
    "education": {"kind", "schule", "schüler", "lehrer", "klasse", "eltern"},
    "household_life": {
        "kind", "kinder", "mensch", "haus", "wohnung", "familie", "mutter", "vater",
        "mädchen", "junge", "baby", "essen", "trinken", "bett", "zimmer", "stuhl",
        "fenster", "tisch", "milch", "wasser", "brot", "spiel", "leben", "frau", "mann"
    },
    "nature_environment": {"baum", "wind", "wasser", "luft", "erde", "regen", "wiese", "natur", "pflanze", "blume"}
}


def initial_rule_based_labeling(
    sentences: List[str],
    topic_keywords: Dict[str, set] = TOPIC_KEYWORDS,
    min_keywords: int = 2,
    n_per_topic: int = 60
) -> Dict[str, List[str]]:
    """
    Perform initial rule-based topic assignment based on keyword overlap.
    Sentences not matched are stored in 'Others'.
    """
    topic_sentences = {topic: [] for topic in topic_keywords}
    topic_sentences["Others"] = []

    for sent in sentences:
        words = set(sent.lower().split())
        matched = False
        for topic, keywords in topic_keywords.items():
            if len(words & keywords) >= min_keywords:
                topic_sentences[topic].append(sent)
                matched = True
        if not matched:
            topic_sentences["Others"].append(sent)

    # keep only top-n per topic
    for topic in topic_keywords:
        topic_sentences[topic] = topic_sentences[topic][:n_per_topic]

    return topic_sentences


def embedding_assisted_reassignment(
    topic_sentences: Dict[str, List[str]],
    model_name: str = "sentence-transformers/LaBSE",
    threshold: float = 0.5,
    max_per_topic: int = 150,
    max_others: int = 150
) -> Dict[str, List[str]]:
    """
    Refine topic assignment for 'Others' sentences using sentence embeddings.
    """
    model = SentenceTransformer(model_name)

    # build centroid embedding for each topic
    topic_centroids = {}
    for topic, sents in topic_sentences.items():
        if topic == "Others" or not sents:
            continue
        embs = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
        topic_centroids[topic] = np.mean(embs, axis=0)

    # encode 'Others'
    others = topic_sentences.get("Others", [])
    others_embeddings = model.encode(others, convert_to_numpy=True, show_progress_bar=True)

    newly_labeled = {topic: [] for topic in topic_centroids}
    still_others = []

    for i, emb in enumerate(others_embeddings):
        sims = {topic: cosine_similarity([emb], [centroid])[0][0] for topic, centroid in topic_centroids.items()}
        best_topic, best_score = max(sims.items(), key=lambda x: x[1])
        if best_score > threshold:
            newly_labeled[best_topic].append(others[i])
        else:
            still_others.append(others[i])

    # merge and cap
    final_topic_sentences = defaultdict(list)
    for topic, sents in topic_centroids.items():
        combined = topic_sentences[topic] + newly_labeled[topic]
        final_topic_sentences[topic] = combined[:max_per_topic]
    final_topic_sentences["Others"] = still_others[:max_others]

    return final_topic_sentences
