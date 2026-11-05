"""Draw the README's summary figure from the recorded results (no data download needed).

    python scripts/make_readme_figure.py      # writes reports/figures/00_readme_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "00_readme_summary.png"
SHOWN = [("Approve everything", "Approve every payment"),
         ("Today's rules", "Today's fixed rules"),
         ("Model: review the top 1.5%", "Model: review the 1.5% highest scores"),
         ("Model: expected-cost policy", "Model + cost-based approve / review / decline")]


def main() -> Path:
    p = json.loads((ROOT / "reports" / "metrics" / "05_policy.json").read_text(encoding="utf-8"))["policies"]
    names = [label for _, label in SHOWN]
    cost = [p[key]["cost_per_1000"] for key, _ in SHOWN]
    stopped = [p[key]["fraud_dollars_stopped"] for key, _ in SHOWN]
    colours = ["#BFBFBF", "#BFBFBF", "#8FA8C8", "#2C5282"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    bars = ax.barh(names, cost, color=colours, height=0.6)
    for b, c, s in zip(bars, cost, stopped):
        ax.text(b.get_width() + 60, b.get_y() + b.get_height() / 2,
                f"${c:,.0f}   ({s:.0%} of fraud dollars stopped)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(cost) * 1.55)
    ax.set_xlabel("cost of fraud and of the decisions, per 1,000 payments (test months)")
    saving = 1 - cost[3] / cost[1]
    ax.set_title(f"Deciding each payment by expected cost costs {saving:.0%} less than today's rules",
                 loc="left", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    return OUT


if __name__ == "__main__":
    print("wrote", main().relative_to(ROOT))
