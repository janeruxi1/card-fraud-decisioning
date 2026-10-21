# %% [markdown]
# # 03 · Supervised models, judged on later months
#
# Four ways to rank payments for review, fitted on the training months (days 0–119), tuned on the
# validation months (120–149) and scored **once** on the test months (150 onwards):
#
# | Model | What it is |
# |---|---|
# | Rules | A stand-in for today's rules (`config.RULES`), written before any model was fitted |
# | Logistic regression | Numeric fields and phase-2 features, missing flags, standardized |
# | Gradient boosting | LightGBM on the phase-2 features and the raw fields, **without** the provider's `V` columns |
# | Gradient boosting + V | The same with the 339 `V` columns |
#
# With about 3.5% fraud, ROC AUC flatters every model, so the headline metrics are PR-AUC and
# **recall at review capacity**: the share of fraud (and of fraud dollars) in the top 1.5% of scores,
# the payments the team can actually look at.

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

from fraudrisk import config, data, features, model
from fraudrisk.records import show_verdict, write_metrics

SOURCE = data.source()
# the synthetic CI run only checks that the code runs: a faster learning rate keeps it short
GBM_PARAMS = {} if SOURCE == "real" else {"learning_rate": 0.2, "num_leaves": 15}
GBM_ROUNDS = 2000 if SOURCE == "real" else 150
con = data.connect()
features.build(con)
F = features.frame(con)
uid = con.sql("SELECT TransactionID, uid FROM base").df()
raw = con.sql(f"SELECT {', '.join(model.raw_columns(with_v=False))} FROM tx").df()
print(SOURCE, F.shape, raw.shape)

# %% [markdown]
# ## 1. Fit

# %%
d_nov = model.design(F, raw, with_v=False)
del raw
d_v = model.with_v(d_nov, con)
assert (d_nov.transaction_id == d_v.transaction_id).all()
y, split, amount = d_nov.y, d_nov.split, d_nov.amount
va, te = split == "valid", split == "test"

rule_frame = F.merge(d_nov.X[[]].assign(TransactionID=d_nov.transaction_id), on="TransactionID")
rule_frame = rule_frame.merge(con.sql("SELECT TransactionID, ProductCD, TransactionAmt FROM tx").df(), on="TransactionID")
rule_frame = rule_frame.set_index("TransactionID").loc[d_nov.transaction_id].reset_index()
scores = {"Rules": model.rules_score(rule_frame)}

logit = model.fit_logistic(d_nov)
scores["Logistic regression"] = logit(d_nov.X)

# %%
gbm = model.fit_lightgbm(d_nov, GBM_PARAMS, GBM_ROUNDS)
scores["Gradient boosting"] = gbm.predict(d_nov.X, num_iteration=gbm.best_iteration)
print("rounds (no V):", gbm.best_iteration)

# %%
gbm_v = model.fit_lightgbm(d_v, GBM_PARAMS, GBM_ROUNDS)
scores["Gradient boosting + V"] = gbm_v.predict(d_v.X, num_iteration=gbm_v.best_iteration)
print({"rounds (no V)": gbm.best_iteration, "rounds (+V)": gbm_v.best_iteration})

# %% [markdown]
# ## 2. Validation months: choose the model
#
# The primary model is the one with the higher validation PR-AUC among the two boosting models,
# unless the `V` columns add less than 0.01 PR-AUC, in which case the model without them is preferred
# (a reviewer can be told what its inputs mean). This rule was set before the test months were scored.

# %%
valid = pd.DataFrame({k: model.evaluate(s[va], y[va], amount[va]) for k, s in scores.items()}).T
gain_v = valid.loc["Gradient boosting + V", "pr_auc"] - valid.loc["Gradient boosting", "pr_auc"]
PRIMARY = "Gradient boosting + V" if gain_v >= 0.01 else "Gradient boosting"
booster, D = (gbm_v, d_v) if PRIMARY == "Gradient boosting + V" else (gbm, d_nov)
print(f"V columns add {gain_v:+.3f} validation PR-AUC -> primary model: {PRIMARY}")
valid[["pr_auc", "roc_auc", "recall_at_capacity", "dollar_recall_at_capacity", "precision_at_capacity"]].round(3)

# %% [markdown]
# ## 3. Test months, scored once

# %%
test = pd.DataFrame({k: model.evaluate(s[te], y[te], amount[te]) for k, s in scores.items()}).T
groups = uid.set_index("TransactionID").loc[d_nov.transaction_id, "uid"].fillna(
    pd.Series(d_nov.transaction_id.astype(str), index=uid.set_index("TransactionID").loc[d_nov.transaction_id].index)).to_numpy()
ci = {k: model.bootstrap(scores[k][te], y[te], amount[te], groups[te]) for k in ("Rules", PRIMARY)}
test[["pr_auc", "roc_auc", "recall_at_capacity", "dollar_recall_at_capacity", "precision_at_capacity"]].round(3)

# %% [markdown]
# ## 4. How much comes from customers already known as fraud?
#
# Phase 1 showed labels are shared within a customer. Split the test fraud by whether the customer
# already had a fraud label in the training months, and measure recall at capacity within each group.
# The second group is the honest test of whether the model recognizes *fraud*, not *fraudsters*.

# %%
known = set(uid.loc[uid["TransactionID"].isin(d_nov.transaction_id[(split == "train") & (y == 1)]), "uid"].dropna())
seen = np.array([g in known for g in groups])
k = max(1, int(round(config.REVIEW_CAPACITY * te.sum())))
by_group = {}
for name in ("Rules", PRIMARY):
    s = scores[name][te]
    flagged = np.zeros(te.sum(), bool)
    flagged[np.argsort(-s, kind="stable")[:k]] = True
    yt, st = y[te], seen[te]
    by_group[name] = {"customer already labelled fraud": float(flagged[(yt == 1) & st].mean()) if ((yt == 1) & st).any() else None,
                      "new to fraud": float(flagged[(yt == 1) & ~st].mean()) if ((yt == 1) & ~st).any() else None}
share_known = float(seen[te][y[te] == 1].mean())
print(f"share of test fraud from customers already labelled fraud in training: {share_known:.0%}")
pd.DataFrame(by_group).round(3)

# %% [markdown]
# ## 5. Calibration
#
# Phase 5 turns scores into decisions by expected cost, which needs probabilities. An isotonic map is
# fitted on the validation months and checked on the test months.

# %%
to_prob = model.calibrate(scores[PRIMARY][va], y[va])
prob = to_prob(scores[PRIMARY])
calib = model.calibration_table(prob[te], y[te])
from sklearn.metrics import brier_score_loss
brier = {"calibrated": brier_score_loss(y[te], prob[te]), "raw": brier_score_loss(y[te], scores[PRIMARY][te]),
         "base rate only": brier_score_loss(y[te], np.full(te.sum(), y[split == "train"].mean()))}
print({k: round(v, 5) for k, v in brier.items()})
calib.round(4)

# %% [markdown]
# ## 6. What drives the score, and what a reviewer sees

# %%
imp = model.global_importance(booster, D.X[te])
reasons = model.reason_codes(booster, D.X[te].iloc[np.argsort(-scores[PRIMARY][te], kind="stable")[:k]])
top_reasons = reasons["reason_1"].value_counts(normalize=True).head(8)
imp.head(15).round(4)

# %%
top_reasons.round(3)

# %% [markdown]
# ## 7. Figure

# %%
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
for name, s in scores.items():
    p, r, _ = precision_recall_curve(y[te], s[te])
    ax.plot(r, p, label=f"{name} ({test.loc[name, 'pr_auc']:.2f})", lw=1.4)
ax.axhline(y[te].mean(), color="grey", ls=":", lw=1)
ax.set(title="Precision–recall, test months", xlabel="recall", ylabel="precision")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
x = np.arange(len(test))
ax.bar(x - 0.2, 100 * test["recall_at_capacity"], 0.4, label="fraud caught")
ax.bar(x + 0.2, 100 * test["dollar_recall_at_capacity"], 0.4, label="fraud dollars caught")
ax.set_xticks(x, [n.replace("Gradient boosting", "GBM") for n in test.index], fontsize=8)
ax.set(title=f"Reviewing the top {config.REVIEW_CAPACITY:.1%}, test months", ylabel="%")
ax.legend(fontsize=8, frameon=False)
ax = axes[2]
ax.plot(100 * calib["predicted"], 100 * calib["observed"], marker="o")
lim = 100 * max(calib["predicted"].max(), calib["observed"].max()) * 1.05
ax.plot([0, lim], [0, lim], color="grey", ls=":")
ax.set(title="Calibration, test months", xlabel="predicted fraud %", ylabel="observed fraud %")
fig.tight_layout()
data.savefig(fig, "03_model", SOURCE)

# %% [markdown]
# ## 8. Save scores for phases 4–6 (real data only; never committed) and the record

# %%
if SOURCE == "real":
    out = data.PARQUET.parent / "scores.parquet"
    pd.DataFrame({"TransactionID": d_nov.transaction_id, "split": split, "day": d_nov.day, "isFraud": y,
                  "amount": amount, "uid": groups, "score": scores[PRIMARY], "prob": prob,
                  "rules_score": scores["Rules"]}).to_parquet(out)
    print("wrote", out.relative_to(data.ROOT).as_posix())


def _rec(df):
    return {k: {kk: float(vv) for kk, vv in r.items()} for k, r in df.iterrows()}


record = {
    "primary_model": PRIMARY, "v_gain_valid_pr_auc": float(gain_v),
    "boosting_rounds": {"without V": int(gbm.best_iteration), "with V": int(gbm_v.best_iteration)},
    "valid": _rec(valid), "test": _rec(test),
    "test_ci": {k: {m: list(v) for m, v in c.items()} for k, c in ci.items()},
    "known_customers": {"share_of_test_fraud": share_known, "recall_at_capacity": by_group},
    "calibration": {"brier": brier, "table": calib.to_dict(orient="records")},
    "importance_top15": {k: float(v) for k, v in imp.head(15).items()},
    "top_reason_share": {k: float(v) for k, v in top_reasons.items()},
}
write_metrics("03_model", record, SOURCE)

# %%
t, r = test.loc[PRIMARY], test.loc["Rules"]
c = record["test_ci"][PRIMARY]
bg = by_group[PRIMARY]
show_verdict(
    "Verdict: phase 3, supervised models",
    [
        f"**{PRIMARY} is the primary model** (the `V` columns add {gain_v:+.3f} validation PR-AUC). On the test months it "
        f"finds **{t['recall_at_capacity']:.0%} of fraud and {t['dollar_recall_at_capacity']:.0%} of fraud dollars** in the top "
        f"{config.REVIEW_CAPACITY:.1%} of scores [recall 95% interval {c['recall_at_capacity'][0]:.0%}–{c['recall_at_capacity'][1]:.0%}], "
        f"against {r['recall_at_capacity']:.0%} and {r['dollar_recall_at_capacity']:.0%} for the rules; PR-AUC "
        f"{t['pr_auc']:.3f} vs {r['pr_auc']:.3f}.",
        (f"**The rules stand-in is close to chance on this data** (ROC AUC {r['roc_auc']:.2f}): it was written without "
         "looking, and the merchant's real rules are not in the data. "
         if r["roc_auc"] < 0.6 else "")
        + f"**The fair baseline is logistic regression**: {test.loc['Logistic regression', 'recall_at_capacity']:.0%} of fraud at "
        f"capacity, PR-AUC {test.loc['Logistic regression', 'pr_auc']:.3f}. Gradient boosting's gain over it is the "
        "interactions and the categorical fields a linear model cannot use.",
        f"**{share_known:.0%} of test fraud comes from customers already labelled fraud** in the training months. The model "
        f"flags {bg['customer already labelled fraud'] or 0:.0%} of that fraud and {bg['new to fraud'] or 0:.0%} of fraud from "
        "customers new to fraud; the second number is the one that measures recognizing fraud rather than fraudsters.",
        f"**Calibrated probabilities** (isotonic, fitted on validation): Brier {brier['calibrated']:.4f} against "
        f"{brier['base rate only']:.4f} for the base rate alone, so phase 5 can price decisions with them.",
        "**Most common top reason among flagged payments**: " + ", ".join(f"`{k}` ({v:.0%})" for k, v in top_reasons.head(3).items()) + ".",
    ],
    "Phase 4 asks what an anomaly detector that never sees a fraud label adds to this.",
)
