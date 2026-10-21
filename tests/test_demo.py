import numpy as np
import pandas as pd
import pytest

from fraudrisk import demo, policy


def cells():
    return pd.DataFrame({"pbin": [0, 1, 2, 3], "band": ["<$50", "$100-250", "$500-1k", ">$1k"],
                         "payments": [1000, 500, 100, 50], "frauds": [5, 25, 30, 40],
                         "mean_p": [0.005, 0.05, 0.3, 0.8], "mean_amount": [30.0, 150.0, 700.0, 1500.0],
                         "mean_fraud_amount": [30.0, 150.0, 700.0, 1500.0]})


def test_capacity_and_shares():
    r = demo.solve(cells(), capacity=0.05)
    assert r["review_rate"] == pytest.approx(0.05)
    assert r["actions"]["review_share"].between(0, 1).all()


def test_cost_policy_beats_score_order_and_approve_all():
    c = cells()
    by_cost = demo.solve(c, capacity=0.03)["cost_per_1000"]
    by_score = demo.solve(c, capacity=0.03, order="score")["cost_per_1000"]
    assert by_cost <= by_score + 1e-9
    assert by_cost < demo.approve_all(c)


def test_matches_transaction_level_costs_when_cells_are_single_payments():
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0, 1, n)
    amt = rng.uniform(10, 2000, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    c = pd.DataFrame({"pbin": np.arange(n), "band": "x", "payments": 1, "frauds": y, "mean_p": p,
                      "mean_amount": amt, "mean_fraud_amount": np.where(y == 1, amt, np.nan)})
    r = demo.solve(c, capacity=0.0, decline_min_p=0.0)
    costs = policy.Costs(capacity=0.0)
    a = policy.choose(p, amt, np.zeros(n, int), costs)
    assert r["cost_per_1000"] == pytest.approx(policy.realized(a, y, amt, costs)["cost_per_1000"])
