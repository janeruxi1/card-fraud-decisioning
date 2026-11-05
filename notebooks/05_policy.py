# %% [markdown]
# # 05 · Approve, review or decline: the decision
#
# A score is not a decision. Each payment gets the action with the lowest **expected cost**, using
# the calibrated fraud probability from phase 3 and the scenario's costs (`config.py`):
#
# | Action | Expected cost |
# |---|---|
# | Approve | p × (amount + chargeback fee) |
# | Decline | (1 − p) × (margin × amount + chance the customer leaves × their future margin) |
# | Review | review cost + fraud the reviewer misses + good orders the reviewer wrongly stops |
#
# Reviews are limited to 1.5% of each day's payments, so each day the reviews go to the payments
# where a review saves the most. Policies are compared on the test months by **realized** cost (the
# true labels), never by the model's own expectations.
#
# One guardrail binds: the policy may not decline more good customers than today's rules. Cost
# alone would decline many payments with a moderate fraud probability, so declines are only allowed
# above a probability threshold, chosen on the **validation** months as the lowest threshold that
# keeps good-customer declines within the rules' rate there. The unconstrained policy is reported
# alongside, so the guardrail's price is visible.


# %% [markdown]
# | | |
# |---|---|
# | **Question** | Given a fraud score, should each payment be approved, reviewed or declined, and is the decision layer worth anything on its own? |
# | **Answer** | Choose the action with the lowest expected cost. On the test months this costs $3,430 per 1,000 payments against $5,795 for approving everything. Even with the same model and the same review capacity, sending reviews where they save the most dollars (rather than to the highest scores) is worth about $1,150 per 1,000 payments. The model is weakest on product `W` and on payments without device data; a shadow trial needs only days of traffic but about ten weeks for chargebacks to arrive. |
# | **Next** | [Notebook 06](06_monitoring.ipynb) looks at decay, drift and late labels after launch. |

# %%
import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "fraudrisk" / "__init__.py").exists():
            return p
    raise FileNotFoundError("run from inside the project folder")


sys.path.insert(0, str(_project_root() / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import replace

from fraudrisk import config, data, features, model, policy
from fraudrisk.records import metrics_dir, read_metrics, show_verdict, write_metrics

SOURCE = data.source()
con = data.connect()
from fraudrisk import audit
audit.create_base(con)
S = model.load_or_fit_scores(con, SOURCE)
C = policy.Costs()
te = (S["split"] == "test").to_numpy()
T = S[te].reset_index(drop=True)
V = S[(S["split"] == "valid").to_numpy()].reset_index(drop=True)
p, amt, day, y = T["prob"].to_numpy(), T["amount"].to_numpy(), T["day"].to_numpy(), T["isFraud"].to_numpy()
print(SOURCE, f"{len(T):,} test payments, {int(y.sum()):,} fraud")

# %% [markdown]
# ## 1. Policies on the test months

# %%
def rules_actions(frame, c):
    hits = np.floor(frame["rules_score"].to_numpy())
    a = policy.review_top(np.where(hits >= 1, frame["rules_score"].to_numpy(), -np.inf), frame["day"].to_numpy(), c,
                          decline=hits >= config.RULES_DECLINE_HITS)
    a[(a == policy.REVIEW) & (hits < 1)] = policy.APPROVE      # rules only review payments that trip a rule
    return a


def rules_policy(c):
    return rules_actions(T, c)


# the decline threshold, chosen on the validation months
rules_valid = policy.realized(rules_actions(V, C), V["isFraud"].to_numpy(), V["amount"].to_numpy(), C)
TAU = policy.decline_threshold(V["prob"].to_numpy(), V["amount"].to_numpy(), V["day"].to_numpy(),
                               V["isFraud"].to_numpy(), C, rules_valid["good_decline_rate"])
print(f"decline only when fraud probability >= {TAU:.2f} (validation good-decline rate of the rules: "
      f"{rules_valid['good_decline_rate']:.3%})")

actions = {
    "Approve everything": np.full(len(T), policy.APPROVE),
    "Today's rules": rules_policy(C),
    "Model: review the top 1.5%": policy.review_top(T["score"].to_numpy(), day, C),
    "Model: expected cost, no guardrail": policy.choose(p, amt, day, C),
    "Model: expected-cost policy": policy.choose(p, amt, day, C, TAU),
}
results = pd.DataFrame({k: policy.realized(a, y, amt, C) for k, a in actions.items()}).T
results["saving_per_1000_vs_rules"] = results.loc["Today's rules", "cost_per_1000"] - results["cost_per_1000"]
results.round(4)

# %% [markdown]
# ## 2. Guardrails (`reports/metrics_framework.md`)
#
# The expected-cost policy must not decline more good customers than today's rules, must stay within
# review capacity every day, and must not let more fraud dollars through.

# %%
pol, rules = results.loc["Model: expected-cost policy"], results.loc["Today's rules"]
reviews_by_day = pd.Series(actions["Model: expected-cost policy"] == policy.REVIEW).groupby(day).mean()
guard = {
    "good_decline_rate_not_above_rules": bool(pol["good_decline_rate"] <= rules["good_decline_rate"] + 1e-12),
    "within_capacity_every_day": bool((reviews_by_day <= C.capacity + 1e-12).all()),
    "fraud_loss_not_above_rules": bool(pol["fraud_loss_per_1000"] <= rules["fraud_loss_per_1000"] + 1e-9),
}
guard

# %% [markdown]
# ## 3. Uncertainty: bootstrap over customers

# %%
rng = np.random.default_rng(0)
codes, uniq = pd.factorize(T["uid"])
idx_by = pd.Series(np.arange(len(T))).groupby(codes).apply(np.array).to_numpy()
a_pol, a_rules = actions["Model: expected-cost policy"], actions["Today's rules"]
cost_pol = policy.action_costs(y.astype(float), amt, C)[np.arange(len(T)), a_pol]
cost_rules = policy.action_costs(y.astype(float), amt, C)[np.arange(len(T)), a_rules]
cost_top = policy.action_costs(y.astype(float), amt, C)[np.arange(len(T)), actions["Model: review the top 1.5%"]]
draws, draws_top = [], []
for _ in range(500):
    idx = np.concatenate(idx_by[rng.integers(0, len(uniq), len(uniq))])
    draws.append(1000 * (cost_rules[idx].sum() - cost_pol[idx].sum()) / len(idx))
    draws_top.append(1000 * (cost_top[idx].sum() - cost_pol[idx].sum()) / len(idx))
saving_ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
saving_top_ci = (float(np.percentile(draws_top, 2.5)), float(np.percentile(draws_top, 97.5)))
saving_top = results.loc["Model: review the top 1.5%", "cost_per_1000"] - results.loc["Model: expected-cost policy", "cost_per_1000"]
print(f"saving per 1,000 payments vs simply reviewing the model's top 1.5%: {saving_top:.0f} "
      f"(95% interval {saving_top_ci[0]:.0f} to {saving_top_ci[1]:.0f})")
print(f"saving per 1,000 payments vs today's rules: {results.loc['Model: expected-cost policy', 'saving_per_1000_vs_rules']:.0f} "
      f"(95% interval {saving_ci[0]:.0f} to {saving_ci[1]:.0f})")

# %% [markdown]
# ### 3b. Is the good-customer guardrail really broken?
#
# The guardrail compares two very small rates. Count the customers behind them, and resample customers
# to see how far apart the two rates could plausibly be.

# %%
good = y == 0
bad_pol = (a_pol == policy.DECLINE) & good
bad_rules = (a_rules == policy.DECLINE) & good
gcodes = pd.factorize(T["uid"])[0]
g_pol = np.bincount(gcodes, weights=bad_pol.astype(float))
g_rules = np.bincount(gcodes, weights=bad_rules.astype(float))
g_good = np.bincount(gcodes, weights=good.astype(float))
gd = []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(g_good), len(g_good)), minlength=len(g_good))
    gd.append(10_000 * (w @ g_pol - w @ g_rules) / max(w @ g_good, 1))
guard_test = {
    "good_payments": int(good.sum()), "good_declined_policy": int(bad_pol.sum()), "good_declined_rules": int(bad_rules.sum()),
    "extra_per_10000_good": float(10_000 * (bad_pol.sum() - bad_rules.sum()) / good.sum()),
    "extra_per_10000_good_ci": [float(np.percentile(gd, 2.5)), float(np.percentile(gd, 97.5))],
}
guard_test

# %% [markdown]
# ### 3c. How long must the shadow trial run?
#
# In shadow mode the policy and the rules decide on the same payments, so the trial measures a paired
# difference in cost. Its spread per payment comes from the bootstrap above (resampling customers), and
# shrinks with the square root of the number of payments. Two clocks run: **payments** (enough to detect
# a difference with 80% power at 5% significance) and **labels** (a payment's cost is only known once its
# chargeback could have arrived; delays are exponential with a 30-day median, a scenario assumption).

# %%
per_day = len(T) / len(np.unique(day))
pr = policy.paired_bootstrap(cost_rules, cost_pol, T["uid"])
Z = 1.96 + 0.84


def days_to_detect(effect, se_full):
    return float(len(T) * (Z * se_full / abs(effect)) ** 2 / per_day) if effect else float("inf")


def labels_known(trial_days, wait_days, median=config.LABEL_DELAY_MEDIAN_DAYS):
    """Share of the trial's fraud labels in, ``wait_days`` after it ends (payments spread evenly)."""
    lam = np.log(2) / median
    return 1 - np.exp(-lam * wait_days) * (1 - np.exp(-lam * trial_days)) / (lam * trial_days)


try:
    m3b = read_metrics("03b_model_checks", metrics_dir(SOURCE))
except FileNotFoundError:
    m3b = None
effects = {"policy vs rules: the measured saving": (pr["diff_per_1000"], pr["se"]),
           "policy vs rules: half the measured saving": (pr["diff_per_1000"] / 2, pr["se"])}
if m3b:
    mt = m3b["paired_policy_cost"]
    effects["leading model vs runner-up: the validation margin"] = (mt["valid"]["diff_per_1000"], mt["test"]["se"])
sizing = {}
for name, (eff, se) in effects.items():
    d = days_to_detect(eff, se)
    trial = max(7.0, np.ceil(d))
    wait = next((w for w in range(0, 366) if labels_known(trial, w) >= 0.8), None)
    sizing[name] = {"effect_per_1000": float(eff), "days_of_payments": d, "trial_days": float(trial),
                    "days_until_80pct_labels": wait, "total_days": None if wait is None else float(trial + wait)}
pd.DataFrame(sizing).T.round(1)

# %% [markdown]
# ## 4. Sensitivity to every assumption
#
# Each assumption halved and doubled; the policy is re-solved under the new value and compared with
# today's rules priced the same way. If the saving changes sign anywhere, that assumption needs an owner.

# %%
sens = policy.sensitivity(p, amt, day, y, C, rules_policy, decline_min_p=TAU)
sens.round(3)

# %% [markdown]
# ## 5. What the policy does, by amount

# %%
bands = pd.cut(amt, [0, 50, 100, 250, 500, 1000, np.inf], labels=["<$50", "$50-100", "$100-250", "$250-500", "$500-1k", ">$1k"])
mix = pd.crosstab(bands, pd.Series(a_pol).map(policy.ACTIONS), normalize="index")
mix.round(4)

# %% [markdown]
# ## 5b. Where the model misses fraud
#
# The test months split by what the model can see: a customer's first payment (no history), product
# code, whether device data came with the payment, and order size. For each group: its share of fraud,
# the recall of the model's top 1.5% (ranked over all payments, as the team would), and what the
# policy saves there against the rules.

# %%
features.build(con)
seg = (con.sql("SELECT f.TransactionID, f.cust_prior_tx, t.ProductCD, t.DeviceInfo IS NOT NULL AS has_device "
               "FROM features f JOIN tx t USING (TransactionID)").df()
       .set_index("TransactionID").loc[T["TransactionID"]].reset_index())
segments = {
    "history": np.where(seg["cust_prior_tx"].fillna(0).to_numpy() == 0, "first payment", "has earlier payments"),
    "product": seg["ProductCD"].astype(str).to_numpy(),
    "device data": np.where(seg["has_device"].fillna(False).to_numpy(), "present", "missing"),
    "amount": np.asarray(bands.astype(str)),
}
flag_top = actions["Model: review the top 1.5%"] == policy.REVIEW
seg_rows = []
for dim, lab in segments.items():
    for v in pd.unique(lab):
        m = lab == v
        if (y[m] == 1).sum() < 20:
            continue
        rp, rr = policy.realized(a_pol[m], y[m], amt[m], C), policy.realized(a_rules[m], y[m], amt[m], C)
        seg_rows.append({"dimension": dim, "group": str(v), "payments": int(m.sum()), "frauds": int(y[m].sum()),
                         "fraud_rate": float(y[m].mean()), "share_of_fraud": float(y[m].sum() / y.sum()),
                         "model_recall_top": float(flag_top[m & (y == 1)].mean()),
                         "policy_fraud_dollars_stopped": rp["fraud_dollars_stopped"],
                         "saving_vs_rules_per_1000": rr["cost_per_1000"] - rp["cost_per_1000"]})
seg_tab = pd.DataFrame(seg_rows)
seg_tab.round(3)

# %% [markdown]
# ## 5c. What is more review capacity worth?
#
# The policy re-solved at other capacities, with everything else fixed. The slope is the net value of
# one more review a day, after its $6 cost: the number to set against an analyst's cost.

# %%
CAPS = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
cap_rows = []
for cap in CAPS:
    ck = replace(C, capacity=cap)
    r_ = policy.realized(policy.choose(p, amt, day, ck, TAU), y, amt, ck)
    cap_rows.append({"capacity": cap, "reviews_per_day": cap * per_day, "cost_per_1000": r_["cost_per_1000"],
                     "fraud_dollars_stopped": r_["fraud_dollars_stopped"]})
cap_tab = pd.DataFrame(cap_rows)
total = cap_tab["cost_per_1000"] * len(T) / 1000
cap_tab["net_value_per_extra_review"] = (-total.diff() / (cap_tab["capacity"].diff() * len(T))).round(2)
cap_now = cap_tab.set_index("capacity")
value_next = float(cap_now.loc[0.02, "net_value_per_extra_review"])
cap_tab.round(3)

# %% [markdown]
# ## 6. Aggregates for the demo
#
# The demo re-solves the policy when a viewer changes the costs. It cannot use transactions (the
# data may not be redistributed), so it gets test payments grouped into cells of fraud probability ×
# amount band: counts, frauds and average amounts only. Probability bins hold equal counts below 1%,
# where most payments sit, and are narrow above it, where reviews and declines are decided; one edge
# sits at the decline threshold. Cells with fewer than 10 payments are left out.

# %%
EDGES = sorted({0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, round(TAU, 4)})
low = p < EDGES[0]
pbin = np.empty(len(p), int)
pbin[low] = pd.qcut(pd.Series(p[low]).rank(method="first"), min(20, int(low.sum())), labels=False).to_numpy() if low.sum() else 0
pbin[~low] = 20 + np.searchsorted(EDGES, p[~low], side="right") - 1
cells = (pd.DataFrame({"pbin": pbin, "band": bands.astype(str), "p": p, "amount": amt, "fraud": y})
         .groupby(["pbin", "band"], observed=True)
         .agg(payments=("fraud", "size"), frauds=("fraud", "sum"), mean_p=("p", "mean"),
              mean_amount=("amount", "mean"))
         .reset_index())
fa = (pd.DataFrame({"pbin": pbin, "band": bands.astype(str), "amount": amt, "fraud": y}).query("fraud == 1")
      .groupby(["pbin", "band"], observed=True)["amount"].mean())
cells["mean_fraud_amount"] = [fa.get((r.pbin, r.band), np.nan) for r in cells.itertuples()]
cells = cells[cells["payments"] >= 10].round({"mean_p": 5, "mean_amount": 2, "mean_fraud_amount": 2})
days_in_test = int(len(np.unique(day)))
print(f"{len(cells)} cells covering {cells['payments'].sum() / len(T):.1%} of test payments")
write_metrics("05_policy_cells", {"days": days_in_test, "payments": int(len(T)), "cells": cells.to_dict(orient="records")}, SOURCE)

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
r = results.drop(index="Approve everything")
ax.barh(r.index, r["cost_per_1000"], color=["#95a5a6", "#7f8c8d", "#bdc3c7", "#2c3e50"])
ax.axvline(results.loc["Approve everything", "cost_per_1000"], color="#c0392b", ls=":", label="approve everything")
ax.set(title="Realized cost per 1,000 payments, test months", xlabel="$")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
sv = sens.assign(label=sens["assumption"] + " ×" + sens["factor"].astype(str))
ax.barh(sv["label"], sv["saving_per_1000"], color=np.where(sv["saving_per_1000"] > 0, "#2c3e50", "#c0392b"))
ax.axvline(0, color="grey", lw=1)
ax.set(title="Saving vs today's rules when an assumption changes", xlabel="$ per 1,000 payments")
fig.tight_layout()
data.savefig(fig, "05_policy", SOURCE)


# %% [markdown]
# **How to read the figure.** Left: the realized cost per 1,000 test payments of each policy, from approving everything to the expected-cost policy; lower is better. Right: a sensitivity check; the saving against today's rules when one assumption (chargeback fee, margin, how likely a declined customer is to leave, and so on) is halved or doubled. Every bar staying positive means the conclusion does not hinge on any single assumption.

# %% [markdown]
# **How to read the checks figure** (`05b_policy_checks`). Left: in each group, how much of its fraud the model's top 1.5% finds; bars well short of the dotted line are where the model is weakest. Middle: the policy's cost if the team could review more or fewer payments a day; where the curve flattens, extra reviewers stop paying for themselves. Right: how long after a shadow trial ends its chargebacks are mostly in, which is when its cost can be read.

# %% [markdown]
# ## Record and verdict

# %%
record = {
    "costs": C.__dict__, "test_payments": int(len(T)), "test_frauds": int(y.sum()), "decline_threshold": TAU,
    "policies": {k: {kk: float(vv) for kk, vv in row.items()} for k, row in results.iterrows()},
    "saving_ci_per_1000": list(saving_ci), "saving_vs_model_top_per_1000": float(saving_top),
    "saving_vs_model_top_ci_per_1000": list(saving_top_ci), "guardrails": guard,
    "sensitivity": sens.to_dict(orient="records"),
    "action_mix_by_amount": {str(k): {kk: float(vv) for kk, vv in row.items()} for k, row in mix.iterrows()},
}
write_metrics("05_policy", record, SOURCE)
write_metrics("05b_policy_checks", {
    "added": "2026-11-05, after the results above; see PROJECT_PLAN.md",
    "guardrail_test": guard_test,
    "shadow_trial": {"payments_per_day": per_day, "policy_vs_rules": {**pr, "ci": list(pr["ci"])},
                     "label_delay_median_days": config.LABEL_DELAY_MEDIAN_DAYS, "power": 0.8, "alpha": 0.05,
                     "sizing": sizing},
    "segments": seg_tab.to_dict(orient="records"),
    "capacity": cap_tab.to_dict(orient="records"),
}, SOURCE)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ax = axes[0]
st_ = seg_tab.assign(label=seg_tab["dimension"] + ": " + seg_tab["group"]).iloc[::-1]
ax.barh(st_["label"], 100 * st_["model_recall_top"], color="#2c3e50")
ax.axvline(100 * flag_top[y == 1].mean(), color="#c0392b", ls=":", label="all payments")
ax.set(title="Fraud found in the top 1.5%, by group", xlabel="% of the group's fraud")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
ax.plot(cap_tab["reviews_per_day"], cap_tab["cost_per_1000"], marker="o", color="#2c3e50")
ax.axvline(C.capacity * per_day, color="#c0392b", ls=":", label="today's team")
ax.set(title="Cost against review capacity", xlabel="reviews per day", ylabel="$ per 1,000 payments")
ax.legend(fontsize=8, frameon=False)
ax = axes[2]
tt = np.arange(0, 181)
for trial in sorted({v["trial_days"] for v in sizing.values() if np.isfinite(v["trial_days"]) and v["trial_days"] <= 120}):
    ax.plot(tt, 100 * labels_known(trial, tt), label=f"{trial:.0f}-day trial")
ax.axhline(80, color="grey", ls=":")
ax.set(title="Share of the trial's fraud labels in", xlabel="days after the trial ends", ylabel="%")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
data.savefig(fig, "05b_policy_checks", SOURCE)

# %%
always_positive = bool((sens["saving_per_1000"] > 0).all())
worst = sens.loc[sens["saving_per_1000"].idxmin()]
def usd(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


free = results.loc["Model: expected cost, no guardrail"]
top_q = results.loc["Model: review the top 1.5%"]
big = mix.loc[mix.index[-1]]
show_verdict(
    "Verdict: phase 5, the decision",
    [
        f"**The expected-cost policy costs ${pol['cost_per_1000']:,.0f} per 1,000 payments on the test months**, against "
        f"${results.loc['Approve everything', 'cost_per_1000']:,.0f} for approving everything and ${rules['cost_per_1000']:,.0f} "
        f"for the rules stand-in (saving ${pol['saving_per_1000_vs_rules']:,.0f}, 95% interval ${saving_ci[0]:,.0f} to "
        f"${saving_ci[1]:,.0f}; the stand-in is weak, see phase 3).",
        (f"**The decision layer is worth {usd(saving_top)} per 1,000 payments on its own** "
         if saving_top_ci[0] > 0 else
         f"**The decision layer's own contribution is {usd(saving_top)} per 1,000 payments, not distinguishable from zero** ")
        + f"(95% interval {usd(saving_top_ci[0])} to {usd(saving_top_ci[1])}): same model, same review capacity, but reviews "
        f"go where a review saves the most dollars rather than to the highest fraud scores. It reviews "
        f"{big.get('review', 0):.0%} of orders over $1,000; simply reviewing the top scores costs "
        f"${top_q['cost_per_1000']:,.0f}.",
        f"It stops {pol['fraud_dollars_stopped']:.0%} of fraud dollars (rules {rules['fraud_dollars_stopped']:.0%}), reviews "
        f"{pol['review_rate']:.2%} of payments and declines {pol['decline_rate']:.2%}; good customers declined "
        f"{pol['good_decline_rate']:.3%} vs {rules['good_decline_rate']:.3%} under the rules. Guardrails: "
        + ", ".join(f"{k.replace('_', ' ')} {'met' if v else 'NOT met'}" for k, v in guard.items()) + ".",
        f"**The good-customer guardrail has a price**: without it the policy would decline {free['good_decline_rate']:.2%} of "
        f"good customers and cost ${free['cost_per_1000']:,.0f} per 1,000 payments, "
        f"${pol['cost_per_1000'] - free['cost_per_1000']:,.0f} less, by these costs. Whether that is worth it depends on how "
        "much a wrongly declined customer is really worth, which the brief prices only roughly.",
        (f"**The guardrail is met, by {guard_test['good_declined_rules'] - guard_test['good_declined_policy']} customer(s)**: "
         if guard_test['good_declined_policy'] <= guard_test['good_declined_rules'] else
         f"**The guardrail is missed by {guard_test['good_declined_policy'] - guard_test['good_declined_rules']} customer(s)**: ")
        + f"{guard_test['good_declined_policy']} good payments declined against {guard_test['good_declined_rules']} under the rules, "
        f"out of {guard_test['good_payments']:,}; the difference is {guard_test['extra_per_10000_good']:+.1f} per 10,000 good "
        f"payments (95% interval {guard_test['extra_per_10000_good_ci'][0]:+.1f} to {guard_test['extra_per_10000_good_ci'][1]:+.1f}). "
        "A strict 'no more than the rules' test cannot be decided at this volume; finance should set a margin.",
        f"**Shadow trial**: detecting the measured saving against the rules takes "
        f"{sizing['policy vs rules: the measured saving']['days_of_payments']:.1f} days of payments (half of it: "
        f"{sizing['policy vs rules: half the measured saving']['days_of_payments']:.0f} days), but its cost is only known once "
        f"chargebacks arrive: 80% of a {sizing['policy vs rules: half the measured saving']['trial_days']:.0f}-day trial's labels are in "
        f"{sizing['policy vs rules: half the measured saving']['days_until_80pct_labels']} days after it ends. Labels, not traffic, set the length."
        + (f" Telling the two leading models apart would take {sizing['leading model vs runner-up: the validation margin']['days_of_payments']:,.0f} "
           "days of payments" + (", so the trial cannot settle that choice." if
                                 sizing['leading model vs runner-up: the validation margin']['days_of_payments'] > 90 else ".") if "leading model vs runner-up: the validation margin" in sizing else ""),
        "**Where it is weakest**: " + ", ".join(
            f"{r.dimension} {r.group} {r.model_recall_top:.0%}" for r in seg_tab.nsmallest(3, "model_recall_top").itertuples())
        + f" of fraud found in the top 1.5% (all payments {flag_top[y == 1].mean():.0%}). One more review a day beyond today's team "
        f"is worth ${value_next:,.2f} net.",
        ("**The saving survives every assumption halved or doubled**; the smallest is "
         if always_positive else "**The saving changes sign when an assumption moves**: the worst case is ")
        + f"${worst['saving_per_1000']:,.0f} per 1,000 payments with {worst['assumption'].replace('_', ' ')} ×{worst['factor']}.",
    ],
    "Phase 6 measures how fast the model decays month by month, and how long label delay hides it.",
)
