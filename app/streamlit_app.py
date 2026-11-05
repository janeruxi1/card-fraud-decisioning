"""Interactive demo: card-not-present fraud screening on the IEEE-CIS transactions.

Runs only on aggregate records committed to the repository (reports/metrics), so it deploys without
the transaction data, which the data licence does not allow to be shared.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudrisk import demo, policy  # noqa: E402

METRICS = ROOT / "reports" / "metrics"
REPO = "https://github.com/janeruxi1/card-fraud-decisioning"
ACTION_COLOR = {"approve": "#1baf7a", "review": "#eda100", "decline": "#e34948"}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8985"
BANDS = ["<$50", "$50-100", "$100-250", "$250-500", "$500-1k", ">$1k"]
BASE = policy.Costs()

st.set_page_config(page_title="Card fraud decisioning", page_icon="💳", layout="wide")


def money(x: float, digits: int = 0) -> str:
    """Dollar amount with the $ escaped, so Streamlit's Markdown doesn't read it as maths."""
    return f"\\${x:,.{digits}f}" if x >= 0 else f"−\\${-x:,.{digits}f}"


def usd(x: float, digits: int = 0) -> str:
    """Dollar amount for metrics and chart labels (ASCII minus, so st.metric reads the sign of a delta)."""
    return f"${x:,.{digits}f}" if x >= 0 else f"-${-x:,.{digits}f}"


def styled(fig: go.Figure, height: int = 320, title: str | None = None, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70 if title else 30, b=10),
                      title=dict(text=title, font=dict(size=14), y=0.98, yanchor="top") if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", **layout)
    return fig


@st.cache_data
def load_metrics() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in METRICS.glob("*.json")}


M = load_metrics()
a1, m3, m4, m5, m6 = (M[k] for k in ["01_audit", "03_model", "04_anomaly", "05_policy", "06_monitoring"])
cells = pd.DataFrame(M["05_policy_cells"]["cells"])
PRIMARY = m3["primary_model"]
POL = m5["policies"]
TAU = float(m5["decline_threshold"])
m3b, m5b = M.get("03b_model_checks"), M.get("05b_policy_checks")   # checks added after the first results

# ------------------------------------------------------------------------------------- sidebar
PRESETS = {
    "Baseline (decision memo)": dict(fee=BASE.chargeback_fee, margin=BASE.margin, churn=BASE.churn_prob,
                                     future=BASE.future_margin, review_cost=BASE.review_cost,
                                     capacity=100 * BASE.capacity, tau=round(TAU, 2)),
    "Chargebacks cost twice as much": dict(fee=2 * BASE.chargeback_fee),
    "Declined customers leave more often": dict(churn=2 * BASE.churn_prob, future=2 * BASE.future_margin),
    "Review team doubled": dict(capacity=200 * BASE.capacity),
    "No good-customer guardrail": dict(tau=0.0),
}


def apply_preset() -> None:
    values = {**PRESETS["Baseline (decision memo)"], **PRESETS[st.session_state["preset"]]}
    for k, v in values.items():
        st.session_state[k] = v


if "fee" not in st.session_state:
    st.session_state["preset"] = "Baseline (decision memo)"
    apply_preset()

with st.sidebar:
    st.header("Business assumptions")
    st.caption("Used by the first two tabs. Start from a scenario, then adjust any value.")
    st.selectbox("Scenario", list(PRESETS), key="preset", on_change=apply_preset)
    st.markdown("**If fraud is approved**")
    fee = st.slider("Chargeback fee on top of the order ($)", 0.0, 50.0, step=1.0, key="fee")
    st.markdown("**If a good customer is declined**")
    margin = st.slider("Share of the order lost as margin", 0.05, 0.80, step=0.05, key="margin")
    churn = st.slider("Chance the customer never comes back", 0.0, 0.5, step=0.01, key="churn")
    future = st.slider("Their future margin ($)", 0.0, 300.0, step=10.0, key="future")
    st.markdown("**Fraud review team**")
    review_cost = st.slider("Analyst cost per review ($)", 1.0, 30.0, step=1.0, key="review_cost")
    capacity_pct = st.slider("Capacity (% of payments reviewed per day)", 0.0, 5.0, step=0.25, key="capacity")
    st.markdown("**Guardrail**")
    tau = st.slider("Decline only above this fraud probability", 0.0, 1.0, step=0.01, key="tau",
                    help="Protects good customers: below this probability a risky payment is reviewed or "
                         f"approved, never declined. {TAU:.2f} was chosen on the validation months so good "
                         "customers are declined no more often than under the rules.")
    st.caption(f"Reviewers stop {BASE.catch_rate:.0%} of fraud and pass {BASE.good_pass_rate:.0%} of good orders.")

costs = policy.Costs(chargeback_fee=fee, margin=margin, churn_prob=churn, future_margin=future,
                     review_cost=review_cost, capacity=capacity_pct / 100)

# ------------------------------------------------------------------------------------- header
st.title("💳 Card fraud: from a risk score to a decision")
st.markdown(
    f"Which online card payments should be **approved**, sent to an analyst for **review**, or **declined**? "
    f"Built on **{a1['shape']['transactions']:,} real e-commerce payments** (IEEE-CIS Fraud Detection release, "
    f"{a1['shape']['fraud_rate']:.1%} fraud, {a1['shape']['days']} days). Models were trained on the first four "
    "months, chosen on the fifth and scored once on the sixth. "
    f"[Code, notebooks and decision memo]({REPO})"
)

pol_best, rules = POL["Model: expected-cost policy"], POL["Today's rules"]
r_model, r_rules = m3["test"][PRIMARY]["recall_at_capacity"], m3["test"]["Rules"]["recall_at_capacity"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fraud caught", f"{r_model:.0%}", f"{100 * (r_model - r_rules):+.0f} pts vs rules", border=True,
          help="Share of test-month fraud in the top 1.5% of scores, the share of payments the review team can check.")
k2.metric("Fraud dollars stopped", f"{pol_best['fraud_dollars_stopped']:.0%}",
          f"{100 * (pol_best['fraud_dollars_stopped'] - rules['fraud_dollars_stopped']):+.0f} pts vs rules",
          border=True, help="By the expected-cost policy: reviews where they save the most, declines above the guardrail.")
k3.metric("Cost per 1,000 payments", usd(pol_best["cost_per_1000"]),
          f"{usd(pol_best['cost_per_1000'] - rules['cost_per_1000'])} vs rules", delta_color="inverse", border=True,
          help="Fraud losses, chargeback fees, lost margin and customers, and analyst time, under the baseline assumptions.")
k4.metric("Good customers declined", f"{pol_best['good_decline_rate']:.3%}",
          f"{100 * (pol_best['good_decline_rate'] - rules['good_decline_rate']):+.3f} pts vs rules",
          delta_color="inverse", border=True, help="The guardrail aims to keep this at or below the rules' rate.")
lo, hi = m5["saving_vs_model_top_ci_per_1000"]
st.caption(f"Test months, against a rules stand-in. Turning scores into decisions by expected cost, rather than "
           f"reviewing the highest scores, saves {money(m5['saving_vs_model_top_per_1000'])} per 1,000 payments "
           f"(95% interval {money(lo)} – {money(hi)}).")

with st.expander("How to use this demo", icon="🧭"):
    st.markdown(
        "1. **Decide one payment**: set a fraud probability and an order amount and see whether to approve, review "
        "or decline, and why.\n"
        "2. **Policy calculator**: the same rule applied to every test payment; change the business assumptions in "
        "the sidebar (or pick a scenario) and watch cost, fraud stopped and good customers declined move.\n"
        "3. **The model**: how well each model ranks fraud, whether its probabilities can be trusted, and why.\n"
        "4. **Monitoring**: how fast the model decays after launch, and what to watch while fraud labels arrive late.\n"
        "5. **Why a random split misleads**: the labelling trap in this data, and how the project avoids it.")

tab_one, tab_policy, tab_model, tab_monitor, tab_leak = st.tabs(
    ["⚖️ Decide one payment", "🧮 Policy calculator", "🤖 The model", "📉 Monitoring", "⚠️ Why a random split misleads"])

# ------------------------------------------------------------------------------------- one payment
with tab_one:
    st.subheader("What should happen to this payment?")
    st.caption("Every action has an expected cost. Approving risks the order plus a chargeback fee; declining "
               "risks a good customer's margin and sometimes the customer; a review costs analyst time and stops "
               "most fraud. The policy takes the cheapest. Assumptions come from the sidebar.")
    c_in, c_out = st.columns([1, 2], gap="large")
    with c_in:
        p = st.slider("Model's fraud probability", 0.0, 1.0, 0.30, 0.01, key="one_p",
                      help="Calibrated on later months: of payments scored 0.30, about 30% were fraud.")
        amount = st.number_input("Order amount ($)", 1.0, 5000.0, 400.0, 10.0, key="one_amount")
    cost = policy.action_costs(np.array([p]), np.array([amount]), costs)[0]
    approve_c, review_c, decline_c = (float(x) for x in cost)
    decline_allowed = p >= tau
    default = "decline" if (decline_c < approve_c and decline_allowed) else "approve"
    default_c = decline_c if default == "decline" else approve_c
    action = "review" if review_c < default_c else default
    with c_out:
        why = {
            "approve": f"Approving is expected to cost {money(approve_c, 2)}, less than a review "
                       f"({money(review_c, 2)}).",
            "review": f"A review is expected to cost {money(review_c, 2)}, against {money(default_c, 2)} "
                      f"if the payment were {default}d without one.",
            "decline": f"Declining is expected to cost {money(decline_c, 2)}, less than approving "
                       f"({money(approve_c, 2)}) or a review ({money(review_c, 2)}).",
        }[action]
        box = {"approve": st.success, "review": st.warning, "decline": st.error}[action]
        box(f"**{action.upper()}** — {why}", icon={"approve": "✅", "review": "🔎", "decline": "⛔"}[action])
        names = ["approve", "review", "decline"]
        fig = go.Figure(go.Bar(
            y=names, x=[approve_c, review_c, decline_c], orientation="h",
            marker_color=[ACTION_COLOR[n] for n in names],
            marker_opacity=[1.0 if n == action else 0.35 for n in names],
            text=[usd(v, 2) for v in (approve_c, review_c, decline_c)], textposition="outside",
            hovertemplate="%{y}: %{x:$,.2f}<extra></extra>"))
        styled(fig, 200, "Expected cost of each action", xaxis_title="expected cost ($)",
               yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1.25 * max(cost)]))
        st.plotly_chart(fig, width="stretch")
    notes = []
    if not decline_allowed and decline_c < approve_c:
        notes.append(f"Declining would be cheaper than approving, but the guardrail allows declines only above "
                     f"{tau:.2f}.")
    if action == "review":
        notes.append(f"A review also needs a free analyst. Each day the queue takes the payments where a review saves "
                     f"the most, up to {capacity_pct:g}% of payments; if this one misses the queue it is "
                     f"{'declined' if default == 'decline' else 'approved'}.")
    for n in notes:
        st.info(n, icon="ℹ️")

    st.markdown("#### The whole decision map")
    ps = np.logspace(-3, 0, 160)
    amts = np.logspace(np.log10(5), np.log10(5000), 160)
    P, A = np.meshgrid(ps, amts)
    c = policy.action_costs(P.ravel(), A.ravel(), costs)
    base = np.where((c[:, 2] < c[:, 0]) & (P.ravel() >= tau), 2, 0)
    code = np.where(c[:, 1] < c[np.arange(len(base)), base], 1, base).reshape(P.shape)
    scale = [[0, ACTION_COLOR["approve"]], [1 / 3, ACTION_COLOR["approve"]], [1 / 3, ACTION_COLOR["review"]],
             [2 / 3, ACTION_COLOR["review"]], [2 / 3, ACTION_COLOR["decline"]], [1, ACTION_COLOR["decline"]]]
    fig = go.Figure(go.Heatmap(x=ps, y=amts, z=code, zmin=-0.5, zmax=2.5, colorscale=scale, showscale=False,
                               opacity=0.85, hovertemplate="p %{x:.3f}, $%{y:,.0f}<extra></extra>"))
    for n in names:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=n,
                                 marker=dict(size=12, symbol="square", color=ACTION_COLOR[n])))
    fig.add_trace(go.Scatter(x=[max(p, 1e-3)], y=[amount], mode="markers", name="this payment",
                             marker=dict(size=14, color="white", line=dict(color="black", width=2))))
    styled(fig, 420, xaxis=dict(type="log", title="fraud probability", tickformat=".1%"),
           yaxis=dict(type="log", title="order amount", tickvals=[10, 100, 1000],
                      ticktext=["$10", "$100", "$1,000"]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Cheapest action for every probability and amount, before the daily capacity limit. Reviews pay off "
               "in a band that widens with the amount, which is why large orders get most of the analysts' time; "
               "declines need both a high probability and the guardrail.")

# ------------------------------------------------------------------------------------- policy
with tab_policy:
    st.subheader("The policy on the test months, under your assumptions")
    if m5b:
        cp = pd.DataFrame(m5b["capacity"])
        now = cp.loc[(cp["capacity"] - BASE.capacity).abs().idxmin()]
        nxt = cp[cp["capacity"] > BASE.capacity].head(1)
        if len(nxt):
            st.caption(f"Recorded on the test months at the brief's costs: around today's {BASE.capacity:.1%} capacity "
                       f"({now['reviews_per_day']:.0f} reviews a day), one more review a day saves about "
                       f"{money(float(nxt['net_value_per_extra_review'].iloc[0]))} after its "
                       f"{money(BASE.review_cost)} cost. Try the capacity slider to see the curve.")
    st.caption(f"Re-solved on {len(cells)} cells (fraud-probability bin × amount band) covering "
               f"{M['05_policy_cells']['payments']:,} test payments, so figures are close to, not identical to, the "
               "transaction-level results in the decision memo. Change the assumptions in the sidebar.")
    kw = dict(fee=fee, margin=margin, churn=churn, future=future, review_cost=review_cost,
              capacity=capacity_pct / 100)
    best = demo.solve(cells, decline_min_p=tau, **kw)
    top = demo.solve(cells, order="score", **kw)
    free = demo.solve(cells, decline_min_p=0.0, **kw)
    approve_all = demo.approve_all(cells, fee)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cost per 1,000 payments", usd(best["cost_per_1000"]),
              f"{usd(best['cost_per_1000'] - top['cost_per_1000'])} vs reviewing top scores", delta_color="inverse")
    k2.metric("Fraud dollars stopped", f"{best['fraud_dollars_stopped']:.0%}")
    k3.metric("Good customers declined", f"{best['good_decline_rate']:.3%}")
    k4.metric("Payments reviewed / declined", f"{best['review_rate']:.2%} / {best['decline_rate']:.2%}")

    left, right = st.columns(2, gap="large")
    with left:
        rows = [("Approve everything", approve_all, GREY),
                ("Model: review the top scores", top["cost_per_1000"], BLUE),
                ("Model: expected-cost policy", best["cost_per_1000"], ORANGE)]
        if tau > 0:
            rows.append(("…without the guardrail", free["cost_per_1000"], "#c9c8c3"))
        fig = go.Figure(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
                               marker_color=[r[2] for r in rows], text=[usd(r[1]) for r in rows],
                               textposition="outside", hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
        styled(fig, 300, "Cost per 1,000 payments (lower is better)", yaxis=dict(autorange="reversed"),
               xaxis=dict(range=[0, 1.2 * max(r[1] for r in rows)], tickprefix="$"))
        st.plotly_chart(fig, width="stretch")
        if tau > 0:
            st.caption(f"Without the guardrail the policy would cost {money(free['cost_per_1000'])} but decline "
                       f"{free['good_decline_rate']:.2%} of good customers. The difference, "
                       f"{money(best['cost_per_1000'] - free['cost_per_1000'])} per 1,000 payments, is the price of "
                       "protecting them, under these assumptions.")
    with right:
        act = best["actions"].assign(n=cells["payments"].to_numpy())
        act["review"] = act["review_share"] * act["n"]
        act["decline"] = (1 - act["review_share"]) * act["decline"].astype(float) * act["n"]
        act["approve"] = act["n"] - act["review"] - act["decline"]
        mix = act.groupby("band")[["approve", "review", "decline"]].sum().reindex(BANDS)
        mix = mix.div(mix.sum(axis=1), axis=0)
        fig = go.Figure([go.Bar(x=mix.index, y=mix[a], name=f"{a}ed" if a == "review" else f"{a}d",
                                marker_color=ACTION_COLOR[a], text=[f"{v:.1%}" for v in mix[a]],
                                textposition="outside", hovertemplate="%{x}: %{y:.2%}<extra></extra>")
                         for a in ["review", "decline"]])
        styled(fig, 300, "Share of payments reviewed or declined, by order size", barmode="group",
               yaxis=dict(tickformat=".0%", range=[0, 1.2 * max(mix[["review", "decline"]].max().max(), 0.01)]))
        st.plotly_chart(fig, width="stretch")
        st.caption("Everything else is approved. Reviews go mostly to large orders, where a caught fraud saves the "
                   "most." + (" Declines stay rare because of the guardrail." if tau >= 0.5 else ""))

# ------------------------------------------------------------------------------------- model
with tab_model:
    st.subheader("How well each model ranks payments")
    st.caption("Test months, scored once. The review team can look at 1.5% of payments, so the question is how much "
               "fraud lands in the top 1.5% of scores. Bars show 95% intervals from a customer-level bootstrap.")
    names_map = {"Rules": "Rules stand-in", "Logistic regression": "Logistic regression", PRIMARY: PRIMARY}
    t = m3["test"]
    left, right = st.columns([1, 1], gap="large")
    with left:
        fig = go.Figure()
        for metric, label, color in [("recall_at_capacity", "fraud payments caught", BLUE),
                                     ("dollar_recall_at_capacity", "fraud dollars caught", ORANGE)]:
            y = [t[k][metric] for k in names_map]
            err = None
            if metric == "recall_at_capacity":
                ci = [m3["test_ci"].get(k, {}).get(metric) for k in names_map]
                err = dict(type="data", symmetric=False, color="#52514e",
                           array=[c[1] - v if c else 0 for c, v in zip(ci, y)],
                           arrayminus=[v - c[0] if c else 0 for c, v in zip(ci, y)])
            fig.add_trace(go.Bar(x=list(names_map.values()), y=y, name=label, marker_color=color, error_y=err,
                                 text=[f"{v:.0%}" for v in y], textposition="inside",
                                 hovertemplate="%{x}: %{y:.1%}<extra>" + label + "</extra>"))
        styled(fig, 340, "Share of fraud in the top 1.5% of scores", barmode="group",
               yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")
    with right:
        table = pd.DataFrame({
            "PR-AUC": [t[k]["pr_auc"] for k in names_map],
            "ROC AUC": [t[k]["roc_auc"] for k in names_map],
            "Fraud in top 1.5%": [t[k]["precision_at_capacity"] for k in names_map],
        }, index=list(names_map.values()))
        st.dataframe(table.style.format({"PR-AUC": "{:.3f}", "ROC AUC": "{:.3f}", "Fraud in top 1.5%": "{:.0%}"}),
                     width="stretch")
        st.caption(f"Fraud is {t[PRIMARY]['base_rate']:.1%} of test payments, so a random ranking scores a PR-AUC "
                   f"of about {t[PRIMARY]['base_rate']:.3f}. Of the payments the model would send to review, "
                   f"{t[PRIMARY]['precision_at_capacity']:.0%} are fraud.")
        k = m3["known_customers"]["recall_at_capacity"][PRIMARY]
        st.markdown(f"**Not just repeat offenders.** "
                    f"{m3['known_customers']['share_of_test_fraud']:.0%} of test fraud came from customers already "
                    f"labelled fraud in training. The model catches {k['customer already labelled fraud']:.0%} of "
                    f"theirs and {k['new to fraud']:.0%} of fraud from customers new to fraud.")

    left, right = st.columns(2, gap="large")
    with left:
        cal = pd.DataFrame(m3["calibration"]["table"])
        fig = go.Figure([
            go.Scatter(x=[1e-3, 1], y=[1e-3, 1], mode="lines", name="perfect", line=dict(color=GREY, dash="dash")),
            go.Scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name="model",
                       line=dict(color=BLUE), hovertemplate="predicted %{x:.2%}<br>observed %{y:.2%}<extra></extra>")])
        styled(fig, 320, "Can the probability be trusted? (deciles of test payments)",
               xaxis=dict(type="log", title="predicted fraud probability", tickformat=".1%"),
               yaxis=dict(type="log", title="observed fraud rate", tickformat=".1%"))
        st.plotly_chart(fig, width="stretch")
        st.caption("The policy multiplies the probability by dollars, so it has to be a real probability. Isotonic "
                   f"calibration brings the Brier score to {m3['calibration']['brier']['calibrated']:.4f} "
                   f"(raw {m3['calibration']['brier']['raw']:.4f}).")
    with right:
        rs = pd.Series(m3["top_reason_share"]).sort_values()
        fig = go.Figure(go.Bar(x=rs.to_numpy(), y=rs.index, orientation="h", marker_color=BLUE,
                               text=[f"{v:.0%}" for v in rs], textposition="outside",
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        styled(fig, 320, "Top reason code on payments sent to review",
               xaxis=dict(tickformat=".0%", range=[0, 1.15 * rs.max()]))
        st.plotly_chart(fig, width="stretch")
        top3 = ", ".join(f"`{c}`" for c in list(m3["top_reason_share"])[:3])
        st.caption(f"Each reviewed payment comes with its largest SHAP contributions; the most common are {top3}. "
                   "The release masks what most columns mean (`C` are counts, `V` the provider's engineered "
                   "features), so a reviewer learns which signal fired, not why.")

    if m5b:
        st.markdown("#### Where the model is weak")
        sg = pd.DataFrame(m5b["segments"])
        sg["label"] = sg["dimension"] + ": " + sg["group"]
        sg = sg.iloc[::-1]
        left, right = st.columns([3, 2], gap="large")
        with left:
            fig = go.Figure(go.Bar(x=sg["model_recall_top"], y=sg["label"], orientation="h", marker_color=BLUE,
                                   customdata=np.c_[sg["share_of_fraud"], sg["frauds"]],
                                   hovertemplate="%{y}: %{x:.0%} found<br>%{customdata[0]:.0%} of all fraud, "
                                                 "%{customdata[1]:,} frauds<extra></extra>"))
            styled(fig, 460, "Share of each group's fraud found in the top 1.5%", xaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, width="stretch")
        with right:
            seg = {(r["dimension"], r["group"]): r for r in m5b["segments"]}
            w, c = seg.get(("product", "W")), seg.get(("product", "C"))
            miss, pres = seg.get(("device data", "missing")), seg.get(("device data", "present"))
            big = seg.get(("amount", ">$1k"))
            lines = []
            if w and c:
                lines.append(f"- **Product W** carries {w['share_of_fraud']:.0%} of the fraud, but the top 1.5% finds "
                             f"only {w['model_recall_top']:.0%} of it (product C: {c['model_recall_top']:.0%}).")
            if miss and pres:
                lines.append(f"- **No device data**: {miss['model_recall_top']:.0%} found, against "
                             f"{pres['model_recall_top']:.0%} when the device is known.")
            if big:
                sv = big["saving_vs_rules_per_1000"]
                lines.append(f"- **Orders over \\$1,000**: the policy costs {money(-sv)} more per 1,000 such payments "
                             f"than the rules (on {big['frauds']} frauds), so keep the large-order rule as a floor."
                             if sv < 0 else
                             f"- **Orders over \\$1,000**: the policy saves {money(sv)} per 1,000 such payments "
                             f"against the rules, on {big['frauds']} frauds.")
            st.markdown("\n".join(lines))

    with st.expander("How the primary model was chosen"):
        sel = pd.DataFrame(m3["selection"]).T
        sel = sel[["valid_pr_auc", "valid_policy_cost_per_1000"]].rename(
            columns={"valid_pr_auc": "PR-AUC (validation)", "valid_policy_cost_per_1000": "Policy cost per 1,000 (validation)"})
        st.dataframe(sel.style.format({"PR-AUC (validation)": "{:.3f}", "Policy cost per 1,000 (validation)": "${:,.0f}"}),
                     width="stretch")
        tu = m3["tuning"]
        st.markdown(f"Four gradient-boosting candidates, with and without the provider's anonymized `V` columns and "
                    f"with default or tuned settings. Tuning ran {tu['trials']} Optuna trials inside the training "
                    f"months only (PR-AUC {tu['default_pr_auc']:.3f} → {tu['best_pr_auc']:.3f} on the later training "
                    f"days). The model with the lowest **policy cost on the validation months** became primary: "
                    f"**{PRIMARY}**.")
        if m3b:
            pv, pt = m3b["paired_policy_cost"]["valid"], m3b["paired_policy_cost"]["test"]
            st.markdown(f"Priced on the same payments, with customers resampled, its lead over **{m3b['runner_up']}** "
                        f"is {money(pv['diff_per_1000'])} per 1,000 payments on the validation months (95% interval "
                        f"{money(pv['ci'][0])} to {money(pv['ci'][1])}) and {money(pt['diff_per_1000'])} on the test months "
                        f"({money(pt['ci'][0])} to {money(pt['ci'][1])}): the two cannot be told apart.")

    with st.expander("Can unsupervised anomaly detection find fraud without labels?"):
        a = m4["test"]
        prof = m4["anomaly_profile"]
        st.markdown(
            f"Not here. Isolation Forest reaches a PR-AUC of **{a['Isolation Forest']['pr_auc']:.3f}**, barely above "
            f"the {a['Supervised']['base_rate']:.3f} of a random ranking, against "
            f"**{a['Supervised']['pr_auc']:.3f}** for the supervised model. The most unusual payments come from "
            f"heavy legitimate users: a median of {prof['Isolation Forest / top 1.5%']['median cust_prior_tx']:,.0f} "
            f"earlier payments on the customer, against {prof['Isolation Forest / rest']['median cust_prior_tx']:,.0f} "
            "for the rest. Fraud here looks ordinary, which is why labels matter.")

# ------------------------------------------------------------------------------------- monitoring
with tab_monitor:
    st.subheader("After launch: decay, and labels that arrive late")
    st.caption(f"The model was frozen after training and followed over {m6['window_days']}-day windows. A chargeback "
               f"takes {m6['label_delay_median_days']} days to arrive at the median (simulated), so recent fraud "
               "is invisible for weeks.")
    d = pd.DataFrame(m6["decay"])
    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure()
        for model_name, color in [("frozen", BLUE), ("refitted", ORANGE)]:
            s = d[d["model"] == model_name]
            err = (dict(type="data", symmetric=False, array=s["recall_ci_high"] - s["recall_at_capacity"],
                        arrayminus=s["recall_at_capacity"] - s["recall_ci_low"], color=color)
                   if "recall_ci_low" in s else None)
            fig.add_trace(go.Scatter(x=s["first_day"], y=s["recall_at_capacity"], mode="lines+markers",
                                     name=f"{model_name} model", line=dict(color=color), error_y=err,
                                     hovertemplate="day %{x}: %{y:.0%}<extra></extra>"))
        styled(fig, 320, "Fraud caught in the top 1.5%, by window",
               xaxis=dict(title="first day of window (days since the first payment)"), yaxis=dict(tickformat=".0%", rangemode="tozero"))
        st.plotly_chart(fig, width="stretch")
        if "frozen_recall_change_ci" in m6:
            lo, hi = m6["frozen_recall_change_ci"]
            st.caption(f"Bars are 95% intervals. The last window is {-100 * m6['frozen_recall_change_first_to_last']:.0f} "
                       f"points below the first ({-100 * hi:.0f} to {-100 * lo:.0f}), though not in a straight line. "
                       "Refitting on the labels known at the time does not help: the newest fraud is not labelled yet.")
        else:
            st.caption("Recall falls over the two months after training. Refitting on the labels known at the time "
                       "does not help: the newest fraud is not labelled yet.")
    with right:
        lag = pd.DataFrame(m6["label_lag"])
        lag = lag[lag["days after window"] != "final"]
        known = lag.groupby("days after window", sort=False)["share_of_fraud_labels_known"].mean()
        fig = go.Figure(go.Bar(x=[f"{k} days" for k in known.index], y=known.to_numpy(), marker_color=BLUE,
                               text=[f"{v:.0%}" for v in known], textposition="outside",
                               hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        styled(fig, 320, "Share of a window's fraud labels known, after…",
               yaxis=dict(tickformat=".0%", range=[0, 1.05]))
        st.plotly_chart(fig, width="stretch")
        st.caption("Averaged over windows. A week after a window closes, most of its fraud is still unlabelled.")

    left, right = st.columns(2, gap="large")
    with left:
        hit = pd.Series(m6["review_queue_hit_rate"])
        psi = pd.Series(m6["score_psi"])
        fig = go.Figure(go.Scatter(x=[int(k) for k in hit.index], y=hit.to_numpy(), mode="lines+markers",
                                   line=dict(color=ORANGE), name="review-queue hit rate",
                                   hovertemplate="day %{x}: %{y:.0%}<extra></extra>"))
        styled(fig, 300, "Early warning: share of reviewed payments that are fraud",
               xaxis=dict(title="first day of window (days since the first payment)"), yaxis=dict(tickformat=".0%", range=[0, 1]))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### What to watch")
        st.markdown(
            f"- **Reviewers' verdicts**, known the same day: the queue's hit rate fell from "
            f"**{hit.iloc[0]:.0%}** to **{hit.iloc[-1]:.0%}**, a decay signal weeks before chargebacks arrive.\n"
            f"- **The score distribution** barely moved (PSI ≤ {psi.max():.3f}), so drift checks on the score "
            "alone would have missed it.\n"
            f"- **Input drift** is real but mostly in date-like columns: {m6['features_psi_over_0_25']} features "
            "have PSI above 0.25.")

    if m5b:
        st.markdown("#### How long a shadow trial needs")
        sz = m5b["shadow_trial"]["sizing"]
        main, half = sz["policy vs rules: the measured saving"], sz["policy vs rules: half the measured saving"]
        models = sz.get("leading model vs runner-up: the validation margin")
        k1, k2, k3 = st.columns(3)
        k1.metric("Traffic to detect the saving", f"{main['days_of_payments']:.0f} days", border=True,
                  help="Days of payments for 80% power at 5% significance, with the policy and the rules deciding "
                       f"the same payments. Half the saving: {half['days_of_payments']:.0f} days.")
        k2.metric("Wait for 80% of chargebacks", f"{main['days_until_80pct_labels']} days", border=True,
                  help=f"After a {main['trial_days']:.0f}-day trial, with a {m5b['shadow_trial']['label_delay_median_days']}"
                       "-day median chargeback delay (a scenario assumption).")
        if models:
            k3.metric("Traffic to tell the two leading models apart", f"{models['days_of_payments']:,.0f} days",
                      border=True, help="Their validation margin is far inside the noise.")
        st.caption("Labels, not traffic, set the length of the trial; and no trial of sensible length can choose "
                   "between the two leading models.")

# ------------------------------------------------------------------------------------- leakage
with tab_leak:
    st.subheader("Why a random train/test split would mislead")
    lp, lk = a1["label_propagation"], a1["leakage"]
    st.markdown(
        "In this release, once a customer's payment is labelled fraud, their later payments usually are too. "
        f"After a fraud-labelled payment, the next one is fraud **{lp['previous transaction fraud']['fraud_rate']:.0%}** "
        f"of the time, against **{lp['previous transaction not fraud']['fraud_rate']:.2%}** otherwise. A model can "
        "look excellent by recognizing customers it has already seen, which does not help with new fraud.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        rows = [("Random 20% hold-out", lk["random 20% hold-out"]["recall"], GREY),
                ("Validation months", lk["valid"]["recall"], BLUE),
                ("Test months", lk["test"]["recall"], BLUE)]
        fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], marker_color=[r[2] for r in rows],
                               text=[f"{r[1]:.0%}" for r in rows], textposition="outside",
                               hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        styled(fig, 320, "Fraud found just by flagging customers already labelled fraud",
               yaxis=dict(tickformat=".0%", range=[0, 1]))
        st.plotly_chart(fig, width="stretch")
        st.caption("On a random hold-out, recognizing known customers alone finds three quarters of the fraud. On "
                   "later months it finds far less, which is the situation a deployed model faces.")
    with right:
        st.markdown("#### What this project does instead")
        st.markdown(
            "- **Split by time**: train on days 0–119, choose on 120–149, test once on 150–181.\n"
            "- **Point-in-time features**: every customer history window ends before the payment; a test truncates "
            "the data and checks that no feature changes.\n"
            "- **Report both groups**: recall on customers already labelled fraud and on customers new to fraud.")

st.divider()
st.caption("Data: IEEE-CIS Fraud Detection (Kaggle, 2019), anonymized card-not-present payments; the transactions are "
           "not redistributed and this app reads only aggregate results. Costs, review capacity and the chargeback "
           f"delay are scenario assumptions documented in the repository. [Source code]({REPO})")
