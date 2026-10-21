"""Phase 4: anomaly scores fitted without any fraud label."""
from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns only, missing values set to the column median (from the rows given)."""
    cols = [c for c in X.columns if str(X[c].dtype) != "category"]
    Z = X[cols].astype("float32")
    return Z


def fit_isolation_forest(Z_train: pd.DataFrame, seed: int = 0, n_estimators: int = 200, max_samples: int = 4096):
    from sklearn.ensemble import IsolationForest

    med = Z_train.median()
    forest = IsolationForest(n_estimators=n_estimators, max_samples=min(max_samples, len(Z_train)),
                             random_state=seed, n_jobs=-1)
    forest.fit(Z_train.fillna(med))
    return lambda Z: -forest.score_samples(Z.fillna(med))   # higher = more anomalous


def fit_robust_distance(Z_train: pd.DataFrame, top: int = 5):
    """Mean of the `top` largest robust z-scores (|x - median| / MAD) across features: a payment
    that is extreme on a few dimensions at once scores high. Transparent and cheap."""
    med = Z_train.median()
    mad = (Z_train - med).abs().median().replace(0, np.nan)
    scale = mad.fillna((Z_train - med).abs().mean()).replace(0, 1.0) * 1.4826

    def score(Z):
        z = ((Z - med).abs() / scale).fillna(0).to_numpy()
        k = min(top, z.shape[1])
        return np.sort(z, axis=1)[:, -k:].mean(axis=1)

    return score


def top_k_mask(score: np.ndarray, share: float) -> np.ndarray:
    k = max(1, int(round(share * len(score))))
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def blend_capacity(primary: np.ndarray, secondary: np.ndarray, capacity: float, secondary_share: float) -> np.ndarray:
    """Spend `secondary_share` of the review capacity on the most anomalous payments the primary
    model did not already pick, and the rest on the primary model's top scores."""
    n = len(primary)
    k_total = max(1, int(round(capacity * n)))
    k_sec = int(round(secondary_share * k_total))
    picked = np.zeros(n, bool)
    picked[np.argsort(-primary, kind="stable")[:k_total - k_sec]] = True
    order = np.argsort(-secondary, kind="stable")
    extra = order[~picked[order]][:k_sec]
    picked[extra] = True
    return picked
