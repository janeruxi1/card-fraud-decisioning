"""Phase 6: decay, drift and what can be seen when labels arrive late."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected``, on the expected quantile bins.
    Missing values are a bin of their own. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large."""
    e = pd.to_numeric(pd.Series(expected), errors="coerce").astype(float)
    a = pd.to_numeric(pd.Series(actual), errors="coerce").astype(float)
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(x):
        counts = pd.cut(x, edges, include_lowest=True).value_counts(sort=False).to_numpy().astype(float)
        counts = np.r_[counts, x.isna().sum()]
        return np.clip(counts / max(len(x), 1), 1e-6, None)

    pe, pa = dist(e), dist(a)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def label_arrival(n: int, median_days: float, seed: int = 0) -> np.ndarray:
    """Days from payment to chargeback, exponential with the given median."""
    rng = np.random.default_rng(seed)
    return rng.exponential(median_days / np.log(2), n)


def windows(day: np.ndarray, start: int, width: int) -> np.ndarray:
    """Window index for each day from ``start`` (−1 before it). A last window shorter than half
    ``width`` is merged into the one before it, so no window is a sliver of a few days."""
    w = np.where(day >= start, (day - start) // width, -1)
    last = int(day.max())
    n_full = (last - start + 1) // width
    if n_full >= 1 and (last - start + 1) % width < width / 2:
        w = np.where(w >= n_full, n_full - 1, w)
    return w
