"""README, memo and model card quote numbers from reports/metrics; this checks them."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
NAMES = ["01_audit", "02_features", "03_model", "04_anomaly", "05_policy", "06_monitoring"]

pytestmark = pytest.mark.skipif(not all((M / f"{n}.json").exists() for n in NAMES), reason="records not present")


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


def usd(x):
    return f"${x:,.0f}"


def expected():
    a, f, m, an, p, mo = (rec(n) for n in NAMES)
    prim = m["primary_model"]
    t = m["test"][prim]
    pol = p["policies"]
    pc, top, free = pol["Model: expected-cost policy"], pol["Model: review the top 1.5%"], pol["Model: expected cost, no guardrail"]
    lp, lk = a["label_propagation"], a["leakage"]
    sel = m["selection"]
    sel_prim = sel[prim]
    sel_runner = sorted(sel.values(), key=lambda r: r["valid_policy_cost_per_1000"])[1]
    test_best = min(sel.values(), key=lambda r: r["test_policy_cost_per_1000"])
    decay = [r for r in mo["decay"] if r["model"] == "frozen"]
    hit = list(mo["review_queue_hit_rate"].values())
    week = [r["share_of_fraud_labels_known"] for r in mo["label_lag"] if r["days after window"] == 7]
    shared = {
        usd(pc["cost_per_1000"]), usd(top["cost_per_1000"]), usd(free["cost_per_1000"]),
        usd(p["saving_vs_model_top_per_1000"]), f"{free['good_decline_rate']:.2%}",
        usd(pc["cost_per_1000"] - free["cost_per_1000"]),
        f"{decay[0]['recall_at_capacity']:.0%}", f"{decay[-1]['recall_at_capacity']:.0%}",
        f"{hit[0]:.0%}", f"{hit[-1]:.0%}", f"{sum(week) / len(week):.0%}",
        f"{m['known_customers']['recall_at_capacity'][prim]['new to fraud']:.0%}",
    }
    readme = shared | {
        f"{a['shape']['transactions']:,}", f"{a['shape']['fraud_rate']:.2%}", f"{t['recall_at_capacity']:.0%}",
        f"**{lp['previous transaction fraud']['fraud_rate']:.0%}**", f"**{lp['previous transaction not fraud']['fraud_rate']:.2%}**",
        f"**{lk['random 20% hold-out']['recall']:.0%}**", f"**{lk['test']['recall']:.0%}**",
        f"[95% interval {100 * m['test_ci'][prim]['recall_at_capacity'][0]:.0f}–{m['test_ci'][prim]['recall_at_capacity'][1]:.0%}]",
        f"PSI ≤ {max(mo['score_psi'].values()):.3f}",
        f"{t['dollar_recall_at_capacity']:.0%} of fraud dollars", f"{t['precision_at_capacity']:.0%} precision",
        f"Brier {m['calibration']['brier']['calibrated']:.4f} vs {m['calibration']['brier']['base rate only']:.4f}",
        f"PR-AUC **{an['test']['Isolation Forest']['pr_auc']:.3f}**", f"{usd(pol['Approve everything']['cost_per_1000'])}",
        f"{p['decline_threshold']:.2f}",
        f"from {m['tuning']['default_pr_auc']:.3f} to {m['tuning']['best_pr_auc']:.3f}",
        f"**{usd(sel_runner['valid_policy_cost_per_1000'] - sel_prim['valid_policy_cost_per_1000'])}**",
        f"({usd(test_best['test_policy_cost_per_1000'])} vs {usd(sel_prim['test_policy_cost_per_1000'])})",
        f"{m['known_customers']['recall_at_capacity'][prim]['customer already labelled fraud']:.0%} of fraud from customers already",
        f"[95% interval {usd(p['saving_vs_model_top_ci_per_1000'][0])}–{usd(p['saving_vs_model_top_ci_per_1000'][1])}]",
        f"{f['point_in_time']['changed_values']}** values",
    }
    memo = shared | {
        usd(test_best["test_policy_cost_per_1000"]),
        f"(95% interval {usd(p['saving_vs_model_top_ci_per_1000'][0])}–{usd(p['saving_vs_model_top_ci_per_1000'][1])})",
        usd(pol["Today's rules"]["cost_per_1000"]), f"at least {p['decline_threshold']:.2f}",
    }
    return readme, memo


def test_readme_numbers_match_records():
    readme, _ = expected()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not sorted(s for s in readme if s not in text)


def test_memo_numbers_match_records():
    _, memo = expected()
    text = (ROOT / "reports" / "decision_memo.md").read_text(encoding="utf-8")
    assert not sorted(s for s in memo if s not in text)
