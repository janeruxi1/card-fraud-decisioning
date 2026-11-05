"""Draw the README's summary figure from the recorded results (no data download needed).

    python scripts/make_readme_figure.py   ->  reports/figures/00_readme_summary.png
"""
import json
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


audit, pol, mon = rec("01_audit"), rec("05_policy")["policies"], rec("06_monitoring")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [0.9, 1.4, 0.9]})

ax = axes[0]
lk = audit["leakage"]
vals = [lk["random 20% hold-out"]["recall"], lk["test"]["recall"]]
bars = ax.bar(["Random 20%\nhold-out", "Later months\n(how it is used)"], [100 * v for v in vals],
              color=[GREY, BLUE], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, 100 * v + 2, f"{v:.0%}", ha="center", fontsize=11)
ax.set_ylim(0, 100)
ax.set_ylabel("fraud found just by recognizing\ncustomers already labelled fraud (%)")
ax.grid(axis="x", visible=False)
ax.set_title("1 · A random split would leak:\nlabels follow the customer")

ax = axes[1]
shown = [("Approve everything", "Approve every payment"), ("Today's rules", "Today's fixed rules"),
         ("Model: review the top 1.5%", "Model: review the top 1.5%"),
         ("Model: expected-cost policy", "Model + expected-cost decisions")]
cost = [pol[k]["cost_per_1000"] for k, _ in shown]
stopped = [pol[k]["fraud_dollars_stopped"] for k, _ in shown]
bars = ax.barh([lab for _, lab in shown], cost, color=[GREY, GREY, GREY, BLUE], height=0.55)
for b, c, s in zip(bars, cost, stopped):
    ax.text(b.get_width() + 80, b.get_y() + b.get_height() / 2, f"\\${c:,.0f}  ·  {s:.0%} of fraud dollars stopped",
            va="center", fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, max(cost) * 1.6)
ax.set_xlabel("cost per 1,000 payments, test months ($)")
ax.grid(axis="y", visible=False)
saving = 1 - cost[3] / cost[1]
ax.set_title(f"2 · Deciding each payment by expected cost\ncosts {saving:.0%} less than today's rules")

ax = axes[2]
lag = [r for r in mon["label_lag"] if r["days after window"] != "final"]
days = sorted({r["days after window"] for r in lag})
known = [sum(r["share_of_fraud_labels_known"] for r in lag if r["days after window"] == d)
         / sum(1 for r in lag if r["days after window"] == d) for d in days]
bars = ax.bar([f"{d} days" for d in days], [100 * k for k in known], color=[BLUE, GREY, GREY], width=0.55)
for b, k in zip(bars, known):
    ax.text(b.get_x() + b.get_width() / 2, 100 * k + 2, f"{k:.0%}", ha="center", fontsize=11)
ax.set_ylim(0, 100)
ax.set_xlabel("time after a period ends")
ax.set_ylabel("share of its fraud labels known (%)")
ax.grid(axis="x", visible=False)
ax.set_title(f"3 · Chargebacks are slow: a week on,\nonly {known[0]:.0%} of fraud is labelled")

save(fig, ROOT / "reports" / "figures" / "00_readme_summary.png")


# ---- hero: where the saving comes from (a bridge from approving everything) --------------------
steps = [("Approve every\npayment", pol["Approve everything"]["cost_per_1000"]),
         ("Today's fixed\nrules", pol["Today's rules"]["cost_per_1000"]),
         ("Model: review\nthe top 1.5%", pol["Model: review the top 1.5%"]["cost_per_1000"]),
         ("Model + expected-\ncost decisions", pol["Model: expected-cost policy"]["cost_per_1000"])]
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, (lab, v) in enumerate(steps):
    if i == 0 or i == len(steps) - 1:
        ax.bar(i, v, color=GREY if i == 0 else BLUE, width=0.6)
        ax.text(i, v / 2, f"\\${v:,.0f}", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    if i > 0:
        prev = steps[i - 1][1]
        drop = prev - v
        ax.bar(i, drop, bottom=v, color=AQUA if i < len(steps) - 1 else BLUE, width=0.6, alpha=0.9 if i < 3 else 0.45)
        ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color=MUTED, lw=1)
        ax.text(i, v + drop / 2, f"−\\${drop:,.0f}", ha="center", va="center", fontsize=10.5, color="white",
                fontweight="bold")
labels = ["Approve every\npayment", "Today's fixed\nrules", "A model picks\nwhat to review",
          "The model + deciding each\npayment by expected cost"]
ax.set_xticks(range(len(steps)), labels)
ax.set_ylim(0, steps[0][1] * 1.12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"\\${v:,.0f}"))
ax.set_ylabel("cost per 1,000 payments\n(fraud losses + reviews + declines)")
ax.grid(axis="x", visible=False)
saving = steps[-1][1] - steps[2][1]
ax.set_title(f"From \\${steps[0][1]:,.0f} to \\${steps[-1][1]:,.0f} per 1,000 payments: the model's ranking saves "
             f"\\${steps[1][1] - steps[2][1]:,.0f} over the rules,\nand pricing each decision saves another \\${-saving:,.0f} with the same model "
             "and the same reviewers (test months)")
save(fig, ROOT / "reports" / "figures" / "00_readme_hero.png")
