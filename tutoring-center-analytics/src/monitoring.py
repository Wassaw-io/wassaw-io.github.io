"""
The model breaks because the marketing worked.

This is the part of a machine learning system that has nothing to do with
machine learning and decides whether the thing survives contact with a business.

The retention model in risk_model.py was fitted on a roster whose students came
mostly from paid search. The whole point of the marketing work in README.md was
to stop buying those students and start earning organic and referred ones. That
succeeded. And those students churn differently, which means the population the
model scores is no longer the population it learned from.

Nobody sabotaged anything. The business improved, and improving the business is
a covariate shift. A model that is not watched will quietly get worse at exactly
the moment its owner is celebrating.

What this file does:

  1.  Population stability index on every feature, reference window against
      current window, with the usual 0.1 / 0.25 thresholds.
  2.  Calibration drift: predicted versus observed churn by scoring cohort, which
      catches the failure that matters when the output is a dollar figure.
  3.  Discrimination decay: held-out AUC by cohort.
  4.  An alert rule that fires on the combination rather than on any one signal,
      because each of the three alone produces enough false positives to be
      ignored within a month, and an ignored alert is worse than no alert.

Run:  python src/monitoring.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import generate_students
from risk_model import expand, FEATURES

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

PSI_WATCH, PSI_ALERT = 0.10, 0.25
CALIB_ALERT = 0.30          # relative error between predicted and observed
AUC_ALERT = 0.03            # absolute drop from the reference cohort

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
})


def _frame(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=20 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def psi(reference, current, bins=10):
    """Population stability index.

        PSI = sum_i (c_i - r_i) * ln(c_i / r_i)

    It is the symmetrised Kullback-Leibler divergence between two binned
    distributions, which is why it behaves sensibly and why the conventional
    0.1 and 0.25 thresholds transfer across problems. Bin edges come from the
    reference window and are then frozen: recomputing them on current data
    would hide the very shift the statistic exists to find.
    """
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if len(np.unique(ref)) <= 2:                     # binary feature
        edges = np.array([-np.inf, 0.5, np.inf])
    else:
        qs = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(ref, qs))
        edges[0], edges[-1] = -np.inf, np.inf
    r, _ = np.histogram(ref, edges)
    c, _ = np.histogram(cur, edges)
    eps = 1e-6
    r = np.clip(r / max(r.sum(), 1), eps, None)
    c = np.clip(c / max(c.sum(), 1), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


# ---------------------------------------------------------------------------
# The drift event: the marketing change, applied to the intake mix
# ---------------------------------------------------------------------------

def build_cohorts(n_per=1200, seed=31):
    """Four intake cohorts. The channel mix shifts across them exactly the way
    README.md says the centre's did: away from paid, toward organic and referral.
    Nothing else about the generating process changes."""
    mixes = [
        ("reference", [0.20, 0.55, 0.15, 0.10]),
        ("quarter 1", [0.28, 0.45, 0.17, 0.10]),
        ("quarter 2", [0.38, 0.31, 0.19, 0.12]),
        ("quarter 3", [0.48, 0.18, 0.22, 0.12]),
    ]
    frames = []
    for i, (name, mix) in enumerate(mixes):
        old_p, old_n, old_s = (generate_students.CHANNEL_P,
                               generate_students.CONFIG["n_students"],
                               generate_students.CONFIG["seed"])
        generate_students.CHANNEL_P = mix
        generate_students.CONFIG = {**generate_students.CONFIG,
                                    "n_students": n_per, "seed": seed + i}
        try:
            d = generate_students.build()
        finally:
            generate_students.CHANNEL_P = old_p
            generate_students.CONFIG = {**generate_students.CONFIG,
                                        "n_students": old_n, "seed": old_s}
        d = d.copy()
        d.attrs = {}          # the ground-truth arrays break pd.concat's attr merge
        d["cohort"] = name
        d["student_id"] = d.student_id + f"_{i}"
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    CHARTS.mkdir(exist_ok=True)
    df = build_cohorts()
    cohorts = ["reference", "quarter 1", "quarter 2", "quarter 3"]

    pp = expand(df.drop(columns=["cohort"]))
    pp = pp.merge(df[["student_id", "cohort"]], on="student_id", how="left")

    ref = pp[pp.cohort == "reference"]
    model = LogisticRegression(max_iter=2000).fit(ref[FEATURES], ref.y)

    # ---- 1. PSI per feature per cohort ----------------------------------
    psi_tab = []
    for c in cohorts[1:]:
        cur = pp[pp.cohort == c]
        row = {"cohort": c}
        for f in FEATURES:
            row[f] = psi(ref[f].values, cur[f].values)
        psi_tab.append(row)
    psi_df = pd.DataFrame(psi_tab).set_index("cohort")

    # ---- 2 & 3. calibration and discrimination --------------------------
    perf = []
    for c in cohorts:
        cur = pp[pp.cohort == c]
        p = model.predict_proba(cur[FEATURES])[:, 1]
        perf.append({
            "cohort": c,
            "predicted": float(p.mean()),
            "observed": float(cur.y.mean()),
            "auc": float(roc_auc_score(cur.y, p)),
            "n": int(len(cur)),
        })
    perf = pd.DataFrame(perf)
    perf["calib_error"] = (perf.predicted - perf.observed).abs() / perf.observed
    ref_auc = float(perf.loc[perf.cohort == "reference", "auc"].iloc[0])
    perf["auc_drop"] = ref_auc - perf.auc

    # ---- 4. the alert ---------------------------------------------------
    alerts = []
    for c in cohorts[1:]:
        worst_feat = psi_df.loc[c].idxmax()
        worst_psi = float(psi_df.loc[c].max())
        r = perf[perf.cohort == c].iloc[0]
        signals = {
            "psi": worst_psi >= PSI_ALERT,
            "calibration": r.calib_error >= CALIB_ALERT,
            "discrimination": r.auc_drop >= AUC_ALERT,
        }
        fired = sum(signals.values()) >= 2
        alerts.append({
            "cohort": c, "worst_feature": worst_feat, "worst_psi": worst_psi,
            "calib_error": float(r.calib_error), "auc_drop": float(r.auc_drop),
            "signals": signals, "alert": bool(fired),
            "verdict": "RETRAIN" if fired else
                       ("watch" if any(signals.values()) else "healthy"),
        })

    # ---- chart -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    ax = axes[0]
    top = psi_df.max().sort_values(ascending=False).head(4).index.tolist()
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for f, col in zip(top, cols):
        ax.plot(range(len(psi_df)), psi_df[f], "o-", color=col, linewidth=2, markersize=6)
        ax.annotate(f, (len(psi_df) - 1, psi_df[f].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=col, va="center")
    ax.axhline(PSI_WATCH, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(PSI_ALERT, color="#b03a1a", linewidth=1.2, linestyle=(0, (5, 3)))
    ax.annotate("watch 0.10", (0, PSI_WATCH), xytext=(0, 4),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.annotate("alert 0.25", (0, PSI_ALERT), xytext=(0, 4),
                textcoords="offset points", fontsize=8.5, color="#b03a1a")
    _frame(ax, "Feature drift", "Population stability index against the reference cohort")
    ax.set_xticks(range(len(psi_df)))
    ax.set_xticklabels(psi_df.index, fontsize=9)
    ax.set_xlim(-0.15, len(psi_df) + 0.55)
    ax.set_ylabel("PSI")

    ax = axes[1]
    x = np.arange(len(perf))
    w = 0.38
    ax.bar(x - w / 2, perf.predicted * 100, w * 0.94, color=ORANGE, label="predicted")
    ax.bar(x + w / 2, perf.observed * 100, w * 0.94, color=BLUE, label="observed")
    _frame(ax, "Calibration drift", "Mean monthly churn probability, predicted vs actual")
    ax.set_xticks(x); ax.set_xticklabels(perf.cohort, fontsize=9)
    ax.set_ylabel("monthly churn, %")
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9)

    ax = axes[2]
    ax.plot(x, perf.auc, "o-", color=BLUE, linewidth=2, markersize=7)
    ax.axhline(ref_auc - AUC_ALERT, color="#b03a1a", linewidth=1.2, linestyle=(0, (5, 3)))
    ax.annotate(f"alert below {ref_auc - AUC_ALERT:.3f}", (0, ref_auc - AUC_ALERT),
                xytext=(0, -14), textcoords="offset points", fontsize=8.5, color="#b03a1a")
    for i, r in perf.iterrows():
        ax.annotate(f"{r.auc:.3f}", (i, r.auc), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    _frame(ax, "Discrimination decay", "Held-out AUC of the frozen reference model")
    ax.set_xticks(x); ax.set_xticklabels(perf.cohort, fontsize=9)
    ax.set_ylabel("AUC")

    fig.tight_layout()
    fig.savefig(CHARTS / "monitoring.png", bbox_inches="tight")
    plt.close(fig)

    out = {"psi": psi_df.round(4).to_dict("index"),
           "performance": perf.round(4).to_dict("records"),
           "alerts": alerts,
           "thresholds": {"psi_watch": PSI_WATCH, "psi_alert": PSI_ALERT,
                          "calibration": CALIB_ALERT, "auc": AUC_ALERT}}
    (ROOT / "results_monitoring.json").write_text(json.dumps(out, indent=2, default=str))

    print("PSI by feature and cohort (reference = the roster the model was fitted on)")
    print(psi_df.round(3).to_string())
    print("\nperformance of the frozen model")
    print(perf[["cohort", "n", "predicted", "observed", "calib_error", "auc", "auc_drop"]]
          .round(4).to_string(index=False))
    print("\nalerts")
    for a in alerts:
        print(f"  {a['cohort']:<11s} {a['verdict']:<8s} "
              f"worst PSI {a['worst_psi']:.3f} on {a['worst_feature']}, "
              f"calib err {a['calib_error']:.1%}, AUC drop {a['auc_drop']:+.3f}")


if __name__ == "__main__":
    main()
