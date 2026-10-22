# %% [markdown]
# # 01 · Audit: what the data contains, and what would leak
#
# Before any model: how big the data is and over what period, how stable the fraud rate is, which
# columns are missing together, and, most important, **how the fraud label is assigned**. The
# organisers describe a rule under which a defrauded customer's later transactions are labelled
# fraud too. If that holds, the label belongs partly to the customer, and a random train/test split
# would let a model score well by recognizing customers instead of fraud.
#
# All data work is SQL (`sql/`), run by DuckDB on the Parquet file. Outputs are aggregates only.

# %%
import sys
from pathlib import Path


def _project_root() -> Path:
    """The folder holding src/fraudrisk, wherever the script or notebook was started."""
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "fraudrisk" / "__init__.py").exists():
            return p
    raise FileNotFoundError("run from inside the project folder")


sys.path.insert(0, str(_project_root() / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from fraudrisk import audit, data
from fraudrisk.records import show_verdict, write_metrics

SOURCE = data.source()
con = data.connect()
audit.create_base(con)
print(f"source: {SOURCE}")

# %% [markdown]
# ## 1. Size, period and the time split
#
# `TransactionDT` is seconds from an undisclosed start, so days are counted from the first day in
# the data. The split was fixed in `src/fraudrisk/data.py` before looking at results: days 0–119
# train, 120–149 validation, 150 onwards test.

# %%
overview = audit.query(con, "01_overview")
overview["split"] = overview["split"].fillna("all")
overview

# %% [markdown]
# ## 2. Is the fraud rate stable over time?

# %%
weekly = audit.query(con, "02_weekly")
weekly.describe()[["transactions", "fraud_rate"]].T

# %% [markdown]
# ## 3. Where fraud concentrates: product and device data

# %%
segments = audit.query(con, "03_segments")
segments

# %% [markdown]
# ## 4. Columns that go missing together
#
# The provider's `V` columns come in groups that are missing on exactly the same rows. Each group
# is one source of information, not dozens of independent features, which matters for feature
# selection and for explaining the model.

# %%
groups = audit.column_groups(con)
miss = {fam: audit.missingness(con, cols) for fam, cols in groups.items() if cols}
family = pd.DataFrame({fam: {"columns": len(m), "median missing": m.median(), "min": m.min(), "max": m.max()}
                       for fam, m in miss.items()}).T
v_blocks = audit.missingness_blocks(miss["V"])
same_rows = [audit.same_rows_check(con, list(miss["V"][miss["V"].round(6) == r].index)) for r in v_blocks["missing"]]
v_blocks["missing on the same rows"] = same_rows
print(f"V columns: {len(miss['V'])}, in {len(v_blocks)} blocks by missing rate; "
      f"{sum(same_rows)} of them missing on identical rows")
family

# %%
v_blocks

# %% [markdown]
# ## 5. Customers, and how the label is shared within them
#
# There is no customer ID. `D1` counts days since the card was first seen, so `calendar day − D1`
# is constant for one card; with the card and billing-address codes it gives an approximate
# customer key (`uid`). Then: when a customer's previous transaction was fraud, how often is the
# next one fraud?

# %%
customers = audit.query(con, "05_customers")
propagation = audit.query(con, "04_label_propagation").set_index("previous")
display_cols = ["transactions", "fraud_rate"]
print(customers.T.to_string(header=False))
propagation[display_cols]

# %% [markdown]
# ## 6. What memorizing customers would buy
#
# Flag a transaction when its customer already carries a fraud label in the training data, and
# measure how much fraud that finds. Under the time split this is knowledge a deployed system could
# have (if chargebacks arrived instantly, which they don't: phase 6 deals with the delay). Under a
# random split, the "training" labels include the customer's own future, so it is leakage.

# %%
leak_time = audit.query(con, "06_leakage").set_index("split")
leak_random = audit.query(con, "07_random_split_leakage")
leak = pd.concat([leak_time[["frauds", "recall", "precision"]],
                  leak_random.assign(split="random 20% hold-out").set_index("split")])
leak

# %% [markdown]
# ## 7. Amounts

# %%
amount = audit.query(con, "08_amount")
amount["share_of_fraud_dollars"] = amount["fraud_dollars"] / amount["fraud_dollars"].sum()
amount

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
ax = axes[0]
ax.plot(weekly["week"], 100 * weekly["fraud_rate"], marker="o", ms=3, color="#c0392b")
for lo, hi, name in ((0, data.TRAIN_END, "train"), (data.TRAIN_END, data.VALID_END, "valid"),
                     (data.VALID_END, weekly["week"].max() * 7 + 7, "test")):
    ax.axvspan(lo / 7, hi / 7, alpha=0.08 if name != "valid" else 0.16, color="grey")
    ax.text((lo + hi) / 14, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 1, name, ha="center", fontsize=8)
ax.set(title="Fraud rate by week", xlabel="week", ylabel="% of transactions")
ax = axes[1]
order = ["first transaction of customer", "previous transaction not fraud", "previous transaction fraud"]
p = propagation.reindex(order)
ax.barh(["first of customer", "previous not fraud", "previous fraud"], 100 * p["fraud_rate"], color=["#95a5a6", "#95a5a6", "#c0392b"])
ax.set(title="Fraud rate, by the customer's previous label", xlabel="% fraud")
ax = axes[2]
ax.bar(amount["decile"], 100 * amount["share_of_fraud_dollars"], color="#2c3e50")
ax.set(title="Share of fraud dollars by amount decile", xlabel="amount decile (1 = smallest)", ylabel="%")
fig.tight_layout()
data.savefig(fig, "01_audit", SOURCE)

# %% [markdown]
# ## Record and verdict

# %%
o = overview.set_index("split")
p = propagation["fraud_rate"]
record = {
    "shape": {"transactions": int(o.loc["all", "transactions"]), "frauds": int(o.loc["all", "frauds"]),
              "fraud_rate": float(o.loc["all", "fraud_rate"]), "days": int(o.loc["all", "last_day"]) + 1,
              "identity_share": float(o.loc["all", "identity_share"]),
              "fraud_dollar_share": float(o.loc["all", "fraud_dollar_share"])},
    "splits": {s: {"transactions": int(o.loc[s, "transactions"]), "frauds": int(o.loc[s, "frauds"]),
                   "fraud_rate": float(o.loc[s, "fraud_rate"]), "days": [int(o.loc[s, "first_day"]), int(o.loc[s, "last_day"])]}
               for s in ("train", "valid", "test")},
    "weekly_fraud_rate": {"min": float(weekly["fraud_rate"].min()), "max": float(weekly["fraud_rate"].max()),
                          "weeks": int(len(weekly))},
    "segments": segments.to_dict(orient="records"),
    "missingness": {"families": family.reset_index(names="family").to_dict(orient="records"),
                    "v_blocks": int(len(v_blocks)), "v_blocks_same_rows": int(sum(same_rows))},
    "customers": {k: float(v) for k, v in customers.iloc[0].items()},
    "label_propagation": {k: {"transactions": int(r["transactions"]), "fraud_rate": float(r["fraud_rate"])}
                          for k, r in propagation.iterrows()},
    "leakage": {k: {"frauds": int(r["frauds"]), "recall": float(r["recall"]), "precision": float(r["precision"])}
                for k, r in leak.iterrows()},
    "amount_deciles": amount[["decile", "min_amount", "max_amount", "fraud_rate", "share_of_fraud_dollars"]].to_dict(orient="records"),
}
write_metrics("01_audit", record, SOURCE)

# %%
sh, lp, lk = record["shape"], record["label_propagation"], record["leakage"]
id_rate = segments[segments["dimension"] == "identity"].set_index("level")["fraud_rate"]
ratio = lp["previous transaction fraud"]["fraud_rate"] / max(lp["previous transaction not fraud"]["fraud_rate"], 1e-9)
show_verdict(
    "Verdict: phase 1, audit",
    [
        f"**{sh['transactions']:,} transactions over {sh['days']} days, {sh['fraud_rate']:.2%} fraud** "
        f"({sh['fraud_dollar_share']:.1%} of dollars); weekly fraud rate between {record['weekly_fraud_rate']['min']:.1%} and "
        f"{record['weekly_fraud_rate']['max']:.1%}. Each split holds enough fraud to measure: "
        + ", ".join(f"{s} {v['frauds']:,}" for s, v in record["splits"].items()) + ".",
        f"**The label is largely a property of the customer.** When an approximate customer's previous transaction is "
        f"labelled fraud, the next one is fraud {lp['previous transaction fraud']['fraud_rate']:.0%} of the time, against "
        f"{lp['previous transaction not fraud']['fraud_rate']:.1%} otherwise ({ratio:.0f}×).",
        f"**So a random split would leak.** Recognizing customers already labelled fraud finds "
        f"{lk['random 20% hold-out']['recall']:.0%} of the fraud in a random hold-out, but {lk['test']['recall']:.0%} in the "
        "later test months, where only the past is known. Validation is by time, and phase 3 reports how much of the model's "
        "skill comes from returning customers.",
        f"**The {len(miss['V'])} provider `V` columns form {record['missingness']['v_blocks']} blocks** by missing rate "
        f"({record['missingness']['v_blocks_same_rows']} missing on identical rows). Device data exists for "
        f"{sh['identity_share']:.0%} of transactions; fraud rate {id_rate['device data']:.1%} with it, "
        f"{id_rate['no device data']:.1%} without.",
    ],
    "Phase 2 builds features from each customer's earlier transactions only, and tests that none can see the future.",
)
