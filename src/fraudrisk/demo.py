"""The policy, re-solved on aggregated cells (fraud probability × amount band) for the demo.

The demo cannot ship transactions, so it works on the cell counts written by notebook 05. Within a
cell every payment gets the cell's average probability and amounts; a cell can be split between
review and its default action when capacity runs out. Results are close to, not equal to, the
transaction-level numbers in reports/metrics/05_policy.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _amounts(cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n, f = cells["payments"].to_numpy(float), cells["frauds"].to_numpy(float)
    a = cells["mean_amount"].to_numpy(float)
    af = cells["mean_fraud_amount"].fillna(cells["mean_amount"]).to_numpy(float)
    good = np.maximum(n - f, 0)
    ag = np.where(good > 0, (n * a - f * af) / np.where(good > 0, good, 1), a)
    return af, np.maximum(ag, 0)


def solve(cells: pd.DataFrame, fee=15.0, margin=0.3, churn=0.1, future=50.0, review_cost=6.0,
          catch=0.9, good_pass=0.99, capacity=0.015, decline_min_p=0.93, order: str = "cost") -> dict:
    """Return actions per cell and realized cost per 1,000 payments.

    order="cost": expected-cost policy (reviews where they save most, declines above the threshold).
    order="score": review the highest-probability cells up to capacity, approve the rest.
    """
    c = cells.reset_index(drop=True)
    n, f, p = c["payments"].to_numpy(float), c["frauds"].to_numpy(float), c["mean_p"].to_numpy(float)
    a = c["mean_amount"].to_numpy(float)
    af, ag = _amounts(c)
    fraud_loss, good_loss = a + fee, margin * a + churn * future
    exp_approve, exp_decline = p * fraud_loss, (1 - p) * good_loss
    exp_review = review_cost + p * (1 - catch) * fraud_loss + (1 - p) * (1 - good_pass) * good_loss
    if order == "cost":
        default_decline = (exp_decline < exp_approve) & (p >= decline_min_p)
        saving = np.where(default_decline, exp_decline, exp_approve) - exp_review
        rank = np.argsort(-saving, kind="stable")
        eligible = saving > 0
    else:
        default_decline = np.zeros(len(c), bool)
        rank = np.argsort(-p, kind="stable")
        eligible = np.ones(len(c), bool)
    budget = capacity * n.sum()
    review_share = np.zeros(len(c))
    for i in rank:
        if budget <= 0 or not eligible[i]:
            continue
        take = min(n[i], budget)
        review_share[i] = take / n[i]
        budget -= take
    # realized cost from the cell's actual frauds
    r_fl, r_gl = f * (af + fee), (n - f) * (margin * ag + churn * future)
    approve_cost, decline_cost = r_fl, r_gl
    review_cost_total = n * review_cost + f * (1 - catch) * (af + fee) + (n - f) * (1 - good_pass) * (margin * ag + churn * future)
    default_cost = np.where(default_decline, decline_cost, approve_cost)
    cost = review_share * review_cost_total + (1 - review_share) * default_cost
    fraud_missed = np.where(default_decline, 0, r_fl) * (1 - review_share) + review_share * f * (1 - catch) * (af + fee)
    total_fraud = r_fl.sum()
    return {
        "cost_per_1000": 1000 * cost.sum() / n.sum(),
        "fraud_dollars_stopped": 1 - fraud_missed.sum() / max(total_fraud, 1e-9),
        "review_rate": float((review_share * n).sum() / n.sum()),
        "decline_rate": float(((1 - review_share) * default_decline * n).sum() / n.sum()),
        "good_decline_rate": float(((1 - review_share) * default_decline * (n - f)).sum() / max((n - f).sum(), 1)),
        "actions": pd.DataFrame({"band": c["band"], "pbin": c["pbin"], "review_share": review_share,
                                 "decline": default_decline & (review_share < 1)}),
    }


def approve_all(cells: pd.DataFrame, fee=15.0) -> float:
    af, _ = _amounts(cells)
    return 1000 * (cells["frauds"].to_numpy(float) * (af + fee)).sum() / cells["payments"].sum()
