"""Stage 3 proxy: is 'food vs. not-food' linearly decodable from the representation?

Deliberately a *linear* probe with cross-validation: anything a logistic
regression can recover is present in the representation in a linearly accessible
form.  A negative result here is informative, not a failure of the pipeline.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

SCORING = ("accuracy", "f1", "balanced_accuracy", "roc_auc")


def run(X_pos: np.ndarray, X_neg: np.ndarray, seed: int = C.SEED) -> dict:
    X = np.vstack([X_pos, X_neg])
    y = np.r_[np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg), dtype=int)]
    splitter = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    scores = cross_validate(clf, X, y, cv=splitter, scoring=SCORING, n_jobs=1)
    return {
        "n_pos": int(len(X_pos)), "n_neg": int(len(X_neg)),
        "folds": C.CV_FOLDS,
        **{m: float(scores[f"test_{m}"].mean()) for m in SCORING},
        **{f"{m}_std": float(scores[f"test_{m}"].std()) for m in SCORING},
        "majority_baseline": float(max(len(X_pos), len(X_neg)) / len(y)),
    }
