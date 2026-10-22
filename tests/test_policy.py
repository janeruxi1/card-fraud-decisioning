import numpy as np
import pytest

from fraudrisk import policy

C = policy.Costs(chargeback_fee=15, margin=0.3, churn_prob=0.1, future_margin=50, review_cost=6,
                 catch_rate=0.9, good_pass_rate=0.99, capacity=0.5)


def test_action_costs_known_answer():
    c = policy.action_costs(np.array([0.1]), np.array([100.0]), C)[0]
    assert c[policy.APPROVE] == pytest.approx(0.1 * 115)
    assert c[policy.DECLINE] == pytest.approx(0.9 * (30 + 5))
    assert c[policy.REVIEW] == pytest.approx(6 + 0.1 * 0.1 * 115 + 0.9 * 0.01 * 35)


def test_choose_picks_the_cheapest_and_respects_capacity():
    p = np.array([0.001, 0.9, 0.2, 0.3])
    amt = np.array([50.0, 500.0, 400.0, 20.0])
    day = np.zeros(4, int)
    a = policy.choose(p, amt, day, C)
    assert a[0] == policy.APPROVE
    assert (a == policy.REVIEW).sum() <= 2
    cost = policy.action_costs(p, amt, C)
    chosen = cost[np.arange(4), a].sum()
    # no single change of action within capacity does better
    for i in range(4):
        for alt in (policy.APPROVE, policy.DECLINE):
            b = a.copy()
            b[i] = alt
            assert cost[np.arange(4), b].sum() >= chosen - 1e-9


def test_decline_threshold_blocks_declines():
    p = np.array([0.6, 0.95])
    amt = np.array([1000.0, 1000.0])
    day = np.zeros(2, int)
    c = policy.Costs(capacity=0.0)
    assert (policy.choose(p, amt, day, c) == policy.DECLINE).all()
    assert policy.choose(p, amt, day, c, decline_min_p=0.9).tolist() == [policy.APPROVE, policy.DECLINE]


def test_reviews_go_where_the_guardrail_leaves_the_largest_risk():
    # payment 0: declining would be cheapest, but the guardrail forces approve, so a review saves a lot;
    # payment 1: allowed to decline cheaply, so a review saves less. One review: it must go to payment 0.
    p = np.array([0.8, 0.97])
    amt = np.array([1000.0, 1000.0])
    c = policy.Costs(capacity=0.5)
    a = policy.choose(p, amt, np.zeros(2, int), c, decline_min_p=0.9)
    assert a.tolist() == [policy.REVIEW, policy.DECLINE]


def test_realized_counts():
    y = np.array([1, 0, 1, 0])
    amt = np.array([100.0, 100.0, 100.0, 100.0])
    a = np.array([policy.APPROVE, policy.DECLINE, policy.REVIEW, policy.APPROVE])
    r = policy.realized(a, y, amt, C)
    assert r["good_decline_rate"] == 0.5 and r["review_rate"] == 0.25
    assert r["fraud_dollars_stopped"] == pytest.approx(1 - (115 + 0.1 * 115) / 230)
