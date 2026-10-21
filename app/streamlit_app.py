"""Fraud screening demo. Reads only the aggregate records in reports/metrics (no transactions)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudrisk import demo  # noqa: E402

METRICS = ROOT / "reports" / "metrics"


@st.cache_data
def load(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))


st.set_page_config(page_title="Fraud screening", layout="wide")
st.title("Card-not-present fraud: from a score to a decision")
a1 = load("01_audit")
st.caption(f"{a1['shape']['transactions']:,} real e-commerce payments (IEEE-CIS release), {a1['shape']['fraud_rate']:.2%} fraud, "
           f"{a1['shape']['days']} days. Models trained on days 0–119, tuned on 120–149, tested once on 150–181. "
           "Every number here is read from reports/metrics; no transaction leaves the analysis.")

tab1, tab2, tab3, tab4 = st.tabs(["The leakage trap", "Models", "Policy calculator", "Monitoring"])

with tab1:
    lp, lk = a1["label_propagation"], a1["leakage"]
    c1, c2 = st.columns(2)
    c1.metric("Fraud rate after a fraud-labelled payment", f"{lp['previous transaction fraud']['fraud_rate']:.0%}")
    c1.metric("…after a payment not labelled fraud", f"{lp['previous transaction not fraud']['fraud_rate']:.1%}")
    c2.write("**Recognizing customers already labelled fraud finds this share of fraud:**")
    c2.bar_chart(pd.Series({"random 20% hold-out": lk["random 20% hold-out"]["recall"], "validation months": lk["valid"]["recall"],
                            "test months": lk["test"]["recall"]}, name="recall"))
    st.write("Labels are shared within a customer, so a random split lets a model look good by memorizing customers. "
             "Every result in this project is measured on later months.")

with tab2:
    m3 = load("03_model")
    t = pd.DataFrame(m3["test"]).T[["pr_auc", "recall_at_capacity", "dollar_recall_at_capacity", "precision_at_capacity"]]
    t.columns = ["PR-AUC", "fraud caught in top 1.5%", "fraud $ caught in top 1.5%", "precision in top 1.5%"]
    st.dataframe(t.style.format("{:.3f}"), width="stretch")
    k = m3["known_customers"]
    st.write(f"Primary model: **{m3['primary_model']}**. {k['share_of_test_fraud']:.0%} of test fraud came from customers "
             "already labelled fraud in training; recall at capacity by group:")
    st.dataframe(pd.DataFrame(k["recall_at_capacity"]).style.format("{:.0%}"), width="stretch")

with tab3:
    cells = pd.DataFrame(load("05_policy_cells")["cells"])
    p5 = load("05_policy")
    st.write("Change the assumptions; the policy is re-solved on aggregated test payments (probability × amount cells), "
             "so results are close to, not equal to, the transaction-level numbers in the memo.")
    c1, c2, c3 = st.columns(3)
    fee = c1.slider("Chargeback fee ($)", 0.0, 50.0, 15.0, 1.0)
    margin = c1.slider("Margin lost on a declined good order", 0.05, 0.8, 0.30, 0.05)
    churn = c2.slider("Chance a declined good customer leaves", 0.0, 0.5, 0.10, 0.01)
    future = c2.slider("Their future margin ($)", 0.0, 300.0, 50.0, 10.0)
    review_cost = c3.slider("Cost per review ($)", 1.0, 30.0, 6.0, 1.0)
    capacity = c3.slider("Review capacity (% of payments)", 0.0, 5.0, 1.5, 0.25) / 100
    tau = st.slider("Decline only above this fraud probability (the good-customer guardrail)", 0.0, 1.0,
                    float(p5.get("decline_threshold", 0.93)), 0.01)
    kw = dict(fee=fee, margin=margin, churn=churn, future=future, review_cost=review_cost, capacity=capacity)
    pol = demo.solve(cells, decline_min_p=tau, **kw)
    top = demo.solve(cells, order="score", **kw)
    m1, m2, m3_, m4 = st.columns(4)
    m1.metric("Approve everything", f"${demo.approve_all(cells, fee):,.0f}")
    m2.metric("Review the top scores", f"${top['cost_per_1000']:,.0f}")
    m3_.metric("Expected-cost policy", f"${pol['cost_per_1000']:,.0f}",
               delta=f"{pol['cost_per_1000'] - top['cost_per_1000']:,.0f} vs top scores", delta_color="inverse")
    m4.metric("Good customers declined", f"{pol['good_decline_rate']:.2%}")
    st.caption("Cost per 1,000 payments, test months. Fraud dollars stopped by the policy: "
               f"{pol['fraud_dollars_stopped']:.0%}; reviews {pol['review_rate']:.2%}, declines {pol['decline_rate']:.2%}.")
    mix = pol["actions"].assign(n=cells["payments"].to_numpy())
    mix["reviewed"] = mix["review_share"] * mix["n"]
    st.bar_chart(mix.groupby("band")["reviewed"].sum().rename("payments reviewed by amount band"))

with tab4:
    m6 = load("06_monitoring")
    d = pd.DataFrame(m6["decay"]).pivot(index="first_day", columns="model", values="recall_at_capacity")
    st.write("**Recall at review capacity by 15-day window** (frozen model vs refitted on labels available at the time)")
    st.line_chart(d)
    lag = pd.DataFrame(m6["label_lag"])
    lag = lag[lag["days after window"] != "final"]
    st.write(f"**Share of a window's fraud labels known**, with a {m6['label_delay_median_days']}-day median chargeback delay")
    st.line_chart(lag.pivot(index="window_first_day", columns="days after window", values="share_of_fraud_labels_known"))
    if "review_queue_hit_rate" in m6:
        st.write("**Review-queue hit rate** (reviewers' verdicts, known the same day)")
        st.line_chart(pd.Series(m6["review_queue_hit_rate"], name="hit rate"))
