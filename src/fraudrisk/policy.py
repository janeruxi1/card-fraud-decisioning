"""Phase 5: approve / review / decline by expected cost, under the review team's daily capacity."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config

APPROVE, REVIEW, DECLINE = 0, 1, 2
ACTIONS = {APPROVE: "approve", REVIEW: "review", DECLINE: "decline"}


@dataclass(frozen=True)
class Costs:
    chargeback_fee: float = config.CHARGEBACK_FEE
    margin: float = config.MARGIN
    churn_prob: float = config.CHURN_PROB_AFTER_DECLINE
    future_margin: float = config.CUSTOMER_FUTURE_MARGIN
    review_cost: float = config.REVIEW_COST
    catch_rate: float = config.REVIEW_CATCH_RATE
    good_pass_rate: float = config.REVIEW_GOOD_PASS_RATE
    capacity: float = config.REVIEW_CAPACITY

    def fraud_loss(self, amount):
        return amount + self.chargeback_fee

    def good_decline_loss(self, amount):
        return self.margin * amount + self.churn_prob * self.future_margin


def action_costs(p: np.ndarray, amount: np.ndarray, c: Costs) -> np.ndarray:
    """Cost of each action (columns approve, review, decline) for fraud probability ``p``.
    With p = 0 or 1 (the realized label) this is the realized cost."""
    fl, gl = c.fraud_loss(amount), c.good_decline_loss(amount)
    approve = p * fl
    decline = (1 - p) * gl
    review = c.review_cost + p * (1 - c.catch_rate) * fl + (1 - p) * (1 - c.good_pass_rate) * gl
    return np.column_stack([approve, review, decline])


def choose(p: np.ndarray, amount: np.ndarray, day: np.ndarray, c: Costs, decline_min_p: float = 0.0) -> np.ndarray:
    """Expected-cost policy. Each payment gets the cheaper of approve and decline (decline only allowed
    when p >= ``decline_min_p``, the guardrail); then, day by day, the payments where a review saves the
    most are reviewed, up to capacity (greedy is optimal: every review uses one unit of capacity)."""
    cost = action_costs(p, amount, c)
    base = np.where((cost[:, DECLINE] < cost[:, APPROVE]) & (p >= decline_min_p), DECLINE, APPROVE)
    saving = np.minimum(cost[:, APPROVE], cost[:, DECLINE]) - cost[:, REVIEW]
    action = base.copy()
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        k = int(np.floor(c.capacity * len(idx)))
        cand = idx[saving[idx] > 0]
        if k and len(cand):
            top = cand[np.argsort(-saving[cand], kind="stable")[:k]]
            action[top] = REVIEW
    return action


def review_top(score: np.ndarray, day: np.ndarray, c: Costs, decline: np.ndarray | None = None) -> np.ndarray:
    """Review the top-scored payments each day up to capacity; approve the rest (or decline where given)."""
    action = np.full(len(score), APPROVE)
    if decline is not None:
        action[decline] = DECLINE
    for d in np.unique(day):
        idx = np.flatnonzero((day == d) & (action != DECLINE))
        k = int(np.floor(c.capacity * (day == d).sum()))
        if k:
            action[idx[np.argsort(-score[idx], kind="stable")[:k]]] = REVIEW
    return action


def realized(action: np.ndarray, y: np.ndarray, amount: np.ndarray, c: Costs) -> dict:
    """Cost of the actions given what actually happened (expected over reviewer outcomes)."""
    cost = action_costs(y.astype(float), amount, c)[np.arange(len(y)), action]
    fl = c.fraud_loss(amount)
    fraud_missed = np.where(action == APPROVE, y * fl, np.where(action == REVIEW, y * (1 - c.catch_rate) * fl, 0.0))
    n = len(y)
    return {
        "cost_per_1000": 1000 * cost.sum() / n,
        "fraud_loss_per_1000": 1000 * fraud_missed.sum() / n,
        "fraud_dollars_stopped": 1 - fraud_missed.sum() / max((y * fl).sum(), 1e-9),
        "review_rate": float((action == REVIEW).mean()),
        "decline_rate": float((action == DECLINE).mean()),
        "good_decline_rate": float(((action == DECLINE) & (y == 0)).sum() / max((y == 0).sum(), 1)),
    }


def decline_threshold(p, amount, day, y, c: Costs, max_good_decline_rate: float,
                      grid=np.r_[0.0, np.linspace(0.05, 0.99, 95), 1.01]) -> float:
    """Smallest decline threshold whose good-customer decline rate (on the data given, the validation
    months) is within ``max_good_decline_rate``. 1.01 means: never decline."""
    for t in grid:
        if realized(choose(p, amount, day, c, t), y, amount, c)["good_decline_rate"] <= max_good_decline_rate + 1e-12:
            return float(t)
    return 1.01


def sensitivity(p, amount, day, y, c: Costs, baseline, factors=(0.5, 2.0), decline_min_p: float = 0.0) -> pd.DataFrame:
    """Re-solve the policy with each assumption scaled, and report the saving against the baseline
    (``baseline(costs) -> actions``), both priced under the same scaled assumptions."""
    rows = []
    fields = ["chargeback_fee", "margin", "churn_prob", "future_margin", "review_cost", "capacity"]
    for f in fields:
        for k in factors:
            v = getattr(c, f) * k
            if f == "churn_prob":
                v = min(v, 1.0)
            ck = replace(c, **{f: v})
            pol = realized(choose(p, amount, day, ck, decline_min_p), y, amount, ck)
            base = realized(baseline(ck), y, amount, ck)
            rows.append({"assumption": f, "factor": k, "value": v,
                         "saving_per_1000": base["cost_per_1000"] - pol["cost_per_1000"],
                         "policy_decline_rate": pol["decline_rate"], "policy_review_rate": pol["review_rate"]})
    return pd.DataFrame(rows)
