"""
Topic bootstrap + sentence embedding + classifiers.

- Uses LaBSE as an example encoder (model_name="sentence-transformers/LaBSE").
  To switch to another embedding model, change `model_name` and consult that
  model's documentation for `encode`.
- Provides: keyword bootstrapping, centroid-based relabeling, embedding,
  and three classifiers (LogReg / RandomForest / XGBoost) with simple tuning.

Dependencies:
  sentence-transformers, scikit-learn, numpy
  (optional) xgboost for XGBClassifier

"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from scipy.stats import loguniform, randint

# Optional import: guarded so the rest of the file still works without xgboost
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier



# 1) Keyword-based bootstrapping


def bootstrap_topics(
    sentences: List[str],
    topic_keywords: Dict[str, set],
    min_keywords: int = 2,
    seed_per_topic: int = 60,
) -> Dict[str, List[str]]:
    """
    Split raw sentences into coarse topics via keyword matching.
    - A sentence is assigned to any topic whose keyword overlap >= min_keywords.
    - Non-matching sentences go to "Others".
    - For each topic (except "Others") keep up to `seed_per_topic` sentences.

    Returns:
        dict topic -> list of sentences (includes "Others").
    """
    topics = {t: [] for t in topic_keywords}
    topics["Others"] = []

    for sent in sentences:
        w = set(sent.lower().split())
        matched = False
        for topic, kw in topic_keywords.items():
            if len(w & kw) >= min_keywords:
                topics[topic].append(sent)
                matched = True
        if not matched:
            topics["Others"].append(sent)

    # cap each topic (except Others)
    for topic in topic_keywords:
        topics[topic] = topics[topic][:seed_per_topic]

    return topics


def centroid_refine_others(
    topics: Dict[str, List[str]],
    model_name: str = "sentence-transformers/LaBSE",
    others_cap: int = 150,
    max_per_topic: int = 150,
    assign_threshold: float = 0.5,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """
    Compute topic centroids from current topic sentences, then
    re-assign "Others" sentences to the nearest topic if similarity >= threshold.

    Args:
        topics: output of `bootstrap_topics`
        model_name: embedding model name (LaBSE used as example)
        others_cap: keep at most this many final "Others"
        max_per_topic: cap per topic after reassignment
        assign_threshold: cosine similarity threshold to move sentence from Others
        show_progress: pass to SentenceTransformer.encode

    Returns:
        dict topic -> list of sentences (refined)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)

    # build centroids
    centroids: Dict[str, np.ndarray] = {}
    for topic in [t for t in topics.keys() if t != "Others"]:
        if len(topics[topic]) == 0:
            continue
        embs = model.encode(topics[topic], convert_to_numpy=True, show_progress_bar=False)
        centroids[topic] = np.mean(embs, axis=0)

    others = topics.get("Others", [])
    if others:
        others_emb = model.encode(others, convert_to_numpy=True, show_progress_bar=show_progress)
    else:
        others_emb = np.zeros((0, next(iter(centroids.values())).shape[0])) if centroids else np.zeros((0, 0))

    newly: Dict[str, List[str]] = {t: [] for t in centroids.keys()}
    still_others: List[str] = []

    for i, emb in enumerate(others_emb):
        if not centroids:
            still_others.append(others[i])
            continue
        sims = {t: float(cosine_similarity([emb], [cent])[0][0]) for t, cent in centroids.items()}
        best_t, best_s = max(sims.items(), key=lambda x: x[1])
        if best_s >= assign_threshold:
            newly[best_t].append(others[i])
        else:
            still_others.append(others[i])

    # compose final caps
    final_topics: Dict[str, List[str]] = defaultdict(list)
    for topic in [t for t in topics.keys() if t != "Others"]:
        combined = topics[topic] + newly.get(topic, [])
        final_topics[topic] = combined[:max_per_topic]

    final_topics["Others"] = still_others[:others_cap]
    return dict(final_topics)



# 2) Embedding utility


def encode_sentences(
    sentences: List[str],
    model_name: str = "sentence-transformers/LaBSE",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode sentences to embeddings using SentenceTransformer-compatible models.
    LaBSE is used as the default example. To switch to another embedding model,
    pass its model name here and follow that model's encode documentation.
    """
    model = SentenceTransformer(model_name)
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=show_progress)



# 3) Dataset assembly


def build_supervised_dataset(topic_to_sentences: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Flatten topic->sentences into parallel lists (texts, labels).
    """
    texts: List[str] = []
    labels: List[str] = []
    for topic, sents in topic_to_sentences.items():
        for s in sents:
            texts.append(s)
            labels.append(topic)
    return texts, labels



# 4) Classifiers + tuning


def train_eval_logreg(
    X: np.ndarray,
    y: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 30,
) -> str:
    """
    Logistic Regression with RandomizedSearch over C, then evaluation.
    Returns classification report (str).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # quick search for C
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions={'C': loguniform(1e-3, 1e3)},
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    search.fit(X, y_enc)  # fit on full embeddings for a more stable C estimate

    best_C = float(search.best_params_['C'])
    final = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs', multi_class='multinomial')
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return classification_report(y_test, y_pred)


def train_eval_random_forest(
    X: np.ndarray,
    y: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 30,
) -> str:
    """
    Random Forest with RandomizedSearch, then evaluation.
    Returns classification report (str).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(random_state=random_state)
    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': [None] + list(range(5, 31, 5)),
        'min_samples_split': randint(2, 10),
        'max_features': ['sqrt', 'log2'],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    search.fit(X, y)

    best = search.best_estimator_
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    return classification_report(y_test, y_pred)


def train_eval_xgboost(
    X: np.ndarray,
    y: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 20,
) -> str:
    """
    XGBoost with RandomizedSearch, then evaluation.
    Requires xgboost installed. Returns classification report (str).
    """
    if not _HAS_XGB:
        return "XGBoost not available: please `pip install xgboost` to use this function."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    param_dist = {
        'n_estimators': randint(100, 150),
        'max_depth': randint(4, 6),
        'learning_rate': loguniform(0.05, 0.15),
    }

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=5,
        random_state=random_state,
        verbose=0,
        n_jobs=-1
    )
    search.fit(X, y_train_enc)

    best = search.best_estimator_
    best.fit(X_train, y_train_enc)
    y_pred_enc = best.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    return classification_report(y_test, y_pred)
