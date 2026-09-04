"""
Answer the four questions a franchise owner actually asks, and draw them.

Run:  python src/analyze.py
Out:  charts/*.png  and  results.md
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

# Palette: slots 1 and 2 of a validated categorical set.
# Checked for colour-vision deficiency separation (worst adjacent pair
# deltaE 24.7 protan) rather than picked by eye.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
})


def _frame(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _cut(ax, df):
    """Mark where the operating change lands, labelled rather than colour-coded."""
    i = int((df["period"] == "before").sum())
    ax.axvline(i - 0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(i - 0.2, ax.get_ylim()[1], " analytics + organic work begins",
            fontsize=8.5, color=MUTED, va="top")


def _xticks(ax, df):
    idx = range(0, len(df), 6)
    ax.set_xticks(list(idx))
    ax.set_xticklabels([df["month"].iloc[i] for i in idx])


def chart_cost_per_enrolled(df):
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.plot(range(len(df)), df["cost_per_enrolled"], color=BLUE, linewidth=2)
    _frame(ax, "Paid acquisition cost per enrolled student",
           "Monthly paid spend divided by enrollments closed that month")
    ax.set_ylabel("dollars per enrollment")
    _xticks(ax, df)
    _cut(ax, df)
    b = df.loc[df.period == "before", "cost_per_enrolled"].mean()
    a = df.loc[df.period == "after", "cost_per_enrolled"].mean()
    for x, y, lab in [(2, b, f"${b:,.0f} avg"), (len(df) - 3, a, f"${a:,.0f} avg")]:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 12),
                    fontsize=9.5, color=INK, ha="center")
    fig.tight_layout()
    fig.savefig(CHARTS / "cost_per_enrolled.png", bbox_inches="tight")
    plt.close(fig)
    return b, a


def chart_channel_mix(df):
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = range(len(df))
    ax.plot(x, df["paid_leads"], color=ORANGE, linewidth=2, label="Paid")
    ax.plot(x, df["organic_leads"], color=BLUE, linewidth=2, label="Organic")
    _frame(ax, "Where the leads came from",
           "Monthly inbound leads by channel")
    ax.set_ylabel("leads per month")
    _xticks(ax, df)
    _cut(ax, df)
    ax.legend(frameon=False, loc="upper left", labelcolor=MUTED)
    ax.annotate("Organic", (len(df) - 2, df["organic_leads"].iloc[-2]),
                textcoords="offset points", xytext=(-6, 8), fontsize=9.5,
                color=INK, ha="right")
    ax.annotate("Paid", (len(df) - 2, df["paid_leads"].iloc[-2]),
                textcoords="offset points", xytext=(-6, -16), fontsize=9.5,
                color=INK, ha="right")
    fig.tight_layout()
    fig.savefig(CHARTS / "channel_mix.png", bbox_inches="tight")
    plt.close(fig)
    return df.loc[df.period == "after", "organic_share"].iloc[-6:].mean()


def chart_funnel(df):
    """Two periods, three stages. Grouped bars with a 2px surface gap."""
    stages = ["Leads", "Trials", "Enrollments"]
    before = [df.loc[df.period == "before", c].mean() for c in ("leads", "trials", "enrollments")]
    after = [df.loc[df.period == "after", c].mean() for c in ("leads", "trials", "enrollments")]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    w = 0.38
    xs = range(len(stages))
    ax.bar([x - w / 2 for x in xs], before, w * 0.96, color=ORANGE, label="Before")
    ax.bar([x + w / 2 for x in xs], after, w * 0.96, color=BLUE, label="After")
    _frame(ax, "Monthly funnel, before and after",
           "Averages per month across each period")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(stages)
    ax.set_ylabel("students per month")
    ax.legend(frameon=False, labelcolor=MUTED)
    for x, (bv, av) in enumerate(zip(before, after)):
        ax.text(x - w / 2, bv, f"{bv:,.0f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(x + w / 2, av, f"{av:,.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(CHARTS / "funnel.png", bbox_inches="tight")
    plt.close(fig)
    return before, after


def chart_enrollment(df):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(range(len(df)), df["active_students"], color=BLUE, linewidth=2)
    _frame(ax, "Active enrollment", "Students on the roster at month end")
    ax.set_ylabel("active students")
    _xticks(ax, df)
    _cut(ax, df)
    fig.tight_layout()
    fig.savefig(CHARTS / "enrollment.png", bbox_inches="tight")
    plt.close(fig)
    return df["active_students"].iloc[0], df["active_students"].iloc[-1]


def main():
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "center_monthly.csv")

    cpe_b, cpe_a = chart_cost_per_enrolled(df)
    org_share = chart_channel_mix(df)
    fun_b, fun_a = chart_funnel(df)
    e0, e1 = chart_enrollment(df)

    churn_b = df.loc[df.period == "before", "monthly_churn"].mean()
    churn_a = df.loc[df.period == "after", "monthly_churn"].mean()
    spend_b = df.loc[df.period == "before", "paid_spend"].mean()
    spend_a = df.loc[df.period == "after", "paid_spend"].mean()

    lines = [
        "# Results",
        "",
        "Recomputed from `data/center_monthly.csv` every time `analyze.py` runs.",
        "",
        f"- Active enrollment: **{e0:,.0f} to {e1:,.0f}** students ({(e1/e0-1)*100:+.0f}%)",
        f"- Paid cost per enrolled student: **${cpe_b:,.0f} to ${cpe_a:,.0f}** ({(cpe_a/cpe_b-1)*100:+.0f}%)",
        f"- Monthly paid spend: **${spend_b:,.0f} to ${spend_a:,.0f}** ({(spend_a/spend_b-1)*100:+.0f}%)",
        f"- Organic share of leads, final six months: **{org_share*100:.0f}%**",
        f"- Monthly churn: **{churn_b*100:.1f}% to {churn_a*100:.1f}%**, "
        f"which moves average student tenure from **{1/churn_b:.0f} to {1/churn_a:.0f} months**",
        f"- Lead to trial to enrollment, monthly averages: "
        f"{fun_b[0]:,.0f}/{fun_b[1]:,.0f}/{fun_b[2]:,.0f} before, "
        f"{fun_a[0]:,.0f}/{fun_a[1]:,.0f}/{fun_a[2]:,.0f} after",
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
