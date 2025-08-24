from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def evaluate_classifiers(
    embeddings: np.ndarray,
    labels,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Run k-fold cross validation on three classifiers (LogReg, RandomForest, XGBoost)
    using the provided embedding features and labels.

    Returns a dict: {model_name: (mean_accuracy, std_accuracy)}.
    """
    # encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = embeddings

    # models with the tuned params you used earlier
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.217, max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=289, max_depth=25, max_features="log2", min_samples_split=8, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            learning_rate=0.123, max_depth=4, n_estimators=120,
            use_label_encoder=False, eval_metric="mlogloss", random_state=random_state
        ),
    }

    results: Dict[str, Tuple[float, float]] = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results[name] = (float(scores.mean()), float(scores.std()))

    return results
