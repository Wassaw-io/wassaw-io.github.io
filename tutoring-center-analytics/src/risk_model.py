"""
Who is going to leave next month, and what is it worth to stop them.

Sections 1-4 of RETENTION.md describe the population. This file turns that into
a list of names, which is the only form an owner can act on.

The method is the discrete-time hazard model. Expand each student into one row
per month they were at risk, with the outcome equal to 1 in the month they left
and 0 otherwise, then fit any binary classifier you like to

    P(leaves in month t | still enrolled at start of t, covariates)

That quantity *is* the discrete hazard. Two things fall out of it for free.

Censoring is handled by construction. A student still enrolled when the data was
cut simply stops contributing rows. Nothing is imputed and nothing is dropped.

And the model becomes a classifier, so the whole gradient-boosting toolbox
applies without giving up the survival semantics: multiply one minus the
predicted hazards along a path and you have a personal survival curve back.

The trap, and it is the one worth checking in an interview: the split must be on
STUDENTS, not on rows. A student contributes many correlated rows, so a random
row split puts the same person in train and test and the held-out score is
meaningless. `train_test_split` on the expanded frame is the single most common
way this model is reported wrong.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
TUITION, MARGIN = 315.0, 0.45
HORIZON_LOOKAHEAD = 3  # months of forward risk the call list is built on

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
})

FEATURES = ["month", "paid", "referral", "walk_in", "two_sessions",
            "assessment_gap", "first8_attendance", "high_school"]


# ---------------------------------------------------------------------------
# Person-period expansion
# ---------------------------------------------------------------------------

def expand(df: pd.DataFrame) -> pd.DataFrame:
    """One row per student per month at risk.

    A student observed for 7.4 months who then churned contributes 8 rows, the
    last carrying y=1. A student censored at 7.4 months contributes 8 rows all
    carrying y=0, and the model correctly reads that as "survived at least 8",
    not as "did not churn, ever".
    """
    rows = []
    for r in df.itertuples(index=False):
        n = int(np.ceil(r.tenure_months))
        for m in range(n):
            last = (m == n - 1)
            rows.append((
                r.student_id, m,
                float(r.channel == "paid"), float(r.channel == "referral"),
                float(r.channel == "walk_in"), float(r.sessions_per_week == 2),
                r.assessment_gap, r.first8_attendance,
                float(r.grade_band == "high"),
                int(last and r.churned == 1),
            ))
    return pd.DataFrame(rows, columns=["student_id"] + FEATURES + ["y"])


def survival_from_hazard(h: np.ndarray) -> np.ndarray:
    """S(t) = prod (1 - h_k). The bridge back from classifier to survival curve."""
    return np.cumprod(1.0 - h)


# ---------------------------------------------------------------------------
# Fit and evaluate
# ---------------------------------------------------------------------------

def fit_and_score(df: pd.DataFrame, seed: int = 3):
    rng = np.random.default_rng(seed)
    ids = df.student_id.unique()
    ids = np.array(ids, dtype=object)
    rng.shuffle(ids)
    cut = int(0.75 * len(ids))
    train_ids, test_ids = set(ids[:cut]), set(ids[cut:])

    pp = expand(df)
    tr = pp[pp.student_id.isin(train_ids)]
    te = pp[pp.student_id.isin(test_ids)]

    gbm = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=320,
        l2_regularization=1.0, min_samples_leaf=40, random_state=seed,
    ).fit(tr[FEATURES], tr.y)

    # Baseline: logistic regression on the same person-period frame. This is the
    # classical discrete-time hazard model, and it is a real competitor rather
    # than a strawman, so beating it has to be earned.
    logit = LogisticRegression(max_iter=2000).fit(tr[FEATURES], tr.y)

    p_gbm = gbm.predict_proba(te[FEATURES])[:, 1]
    p_log = logit.predict_proba(te[FEATURES])[:, 1]

    metrics = {
        "n_students": int(len(ids)),
        "n_person_months": int(len(pp)),
        "train_students": len(train_ids), "test_students": len(test_ids),
        "base_rate": float(te.y.mean()),
        "gbm_auc": float(roc_auc_score(te.y, p_gbm)),
        "logit_auc": float(roc_auc_score(te.y, p_log)),
        "gbm_brier": float(brier_score_loss(te.y, p_gbm)),
        "logit_brier": float(brier_score_loss(te.y, p_log)),
    }

    # How much of the remaining error is unlearnable?
    # The generating process gives each student a Gamma frailty that no feature
    # observes. Scoring the held-out rows with the TRUE individual hazard gives
    # the best AUC any model could reach from the observable covariates plus the
    # part of frailty that leaks into observed tenure. It is a ceiling, not a
    # target, and it is the number that says whether more modelling is worth
    # anyone's afternoon.
    oracle_path = ROOT / "data" / "oracle_hazards.csv"
    if oracle_path.exists():
        orc = pd.read_csv(oracle_path).set_index("student_id")
        metrics["oracle_covariates_auc"] = float(
            roc_auc_score(te.y, te.student_id.map(orc.true_lambda).values))
        metrics["oracle_full_auc"] = float(
            roc_auc_score(te.y, te.student_id.map(orc.true_hazard).values))
        metrics["best_auc"] = max(metrics["gbm_auc"], metrics["logit_auc"])
        metrics["headroom"] = round(metrics["oracle_full_auc"] - metrics["best_auc"], 4)

    imp = permutation_importance(gbm, te[FEATURES], te.y, n_repeats=8,
                                 random_state=seed, scoring="roc_auc")
    metrics["importance"] = sorted(
        [{"feature": f, "drop_in_auc": float(m), "sd": float(s)}
         for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std)],
        key=lambda d: -d["drop_in_auc"])

    return gbm, logit, te, p_gbm, p_log, metrics


# ---------------------------------------------------------------------------
# Calibration matters more than discrimination here
# ---------------------------------------------------------------------------

def chart_calibration(te, p_gbm, p_log, metrics):
    """A ranked call list only needs discrimination. A dollar figure needs
    calibration: if the model says 12% and the true rate is 30%, every expected
    value computed from it is wrong by a factor of two and a half."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    for p, name, col in [(p_gbm, "gradient boosting", BLUE),
                         (p_log, "logistic (classical)", ORANGE)]:
        frac, mean = calibration_curve(te.y, p, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, "o-", color=col, linewidth=2, markersize=6, label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_title("Calibration", loc="left", fontsize=13, color=INK, pad=18)
    ax.text(0, 1.03, "Predicted monthly churn probability against observed",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left")
    ax.annotate("perfect calibration", (lim * 0.62, lim * 0.62), xytext=(6, -16),
                textcoords="offset points", fontsize=9, color=MUTED)

    ax = axes[1]
    imp = metrics["importance"][::-1]
    y = np.arange(len(imp))
    ax.barh(y, [d["drop_in_auc"] for d in imp], 0.6, color=BLUE,
            xerr=[d["sd"] for d in imp], ecolor=GRID, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([d["feature"] for d in imp])
    ax.set_title("What the model is using", loc="left", fontsize=13, color=INK, pad=18)
    ax.text(0, 1.03, "Drop in held-out AUC when the feature is shuffled",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_xlabel("AUC lost")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(CHARTS / "risk_model.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The deliverable: a call list
# ---------------------------------------------------------------------------

def call_list(df, model, top_n=25):
    """For every student still enrolled, the probability of losing them in the
    next three months and what that costs.

        expected loss = P(churn within lookahead) x remaining contribution margin

    Ranking on probability alone is a mistake an owner will notice: a student
    with two months left is not worth the same call as one with two years.
    """
    active = df[df.churned == 0].copy()
    rows = []
    for r in active.itertuples(index=False):
        start = int(np.ceil(r.tenure_months))
        months = np.arange(start, start + 36)
        X = pd.DataFrame({
            "month": months,
            "paid": float(r.channel == "paid"), "referral": float(r.channel == "referral"),
            "walk_in": float(r.channel == "walk_in"),
            "two_sessions": float(r.sessions_per_week == 2),
            "assessment_gap": r.assessment_gap,
            "first8_attendance": r.first8_attendance,
            "high_school": float(r.grade_band == "high"),
        })[FEATURES]
        h = model.predict_proba(X)[:, 1]
        S = survival_from_hazard(h)
        p_leave = float(1 - S[HORIZON_LOOKAHEAD - 1])
        expected_months = float(np.sum(S))
        rows.append({
            "student_id": r.student_id, "channel": r.channel,
            "months_enrolled": round(r.tenure_months, 1),
            "first8_attendance": r.first8_attendance,
            "sessions_per_week": r.sessions_per_week,
            "p_leave_3mo": round(p_leave, 4),
            "expected_remaining_months": round(expected_months, 1),
            "value_at_risk": round(p_leave * expected_months * TUITION * MARGIN, 2),
        })
    out = pd.DataFrame(rows).sort_values("value_at_risk", ascending=False)
    return out.head(top_n).reset_index(drop=True), out


def main():
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "students.csv")

    gbm, logit, te, p_gbm, p_log, metrics = fit_and_score(df)
    chart_calibration(te, p_gbm, p_log, metrics)
    best = logit if metrics["logit_auc"] >= metrics["gbm_auc"] else gbm
    metrics["deployed_model"] = "logistic" if best is logit else "gradient_boosting"
    top, full = call_list(df, best)

    metrics["total_value_at_risk"] = float(full.value_at_risk.sum())
    metrics["top25_value_at_risk"] = float(top.value_at_risk.sum())
    metrics["top25_share"] = metrics["top25_value_at_risk"] / metrics["total_value_at_risk"]
    metrics["active_students"] = int(len(full))

    top.to_csv(ROOT / "data" / "call_list.csv", index=False)
    (ROOT / "results_risk.json").write_text(json.dumps(metrics, indent=2))

    print(f"{metrics['n_person_months']:,} person-months from {metrics['n_students']:,} students")
    print(f"split on students: {metrics['train_students']} train / {metrics['test_students']} test")
    print(f"base rate {metrics['base_rate']:.3f}")
    print(f"AUC   gbm {metrics['gbm_auc']:.3f}  vs logistic {metrics['logit_auc']:.3f}"
          + (f"\n      oracle knowing only the covariates: {metrics['oracle_covariates_auc']:.3f}"
             f"\n      oracle knowing each student's true hazard: {metrics['oracle_full_auc']:.3f}"
             if "oracle_full_auc" in metrics else ""))
    print(f"Brier gbm {metrics['gbm_brier']:.5f}  vs logistic {metrics['logit_brier']:.5f}")
    if "oracle_full_auc" in metrics:
        print(f"unlearnable headroom: {metrics['headroom']:+.3f} AUC "
              f"({metrics['headroom']/(metrics['oracle_full_auc']-0.5)*100:.0f}% of the "
              f"signal above chance is frailty nobody can observe)")
    print("\ntop features by permutation importance:")
    for d in metrics["importance"][:4]:
        print(f"  {d['feature']:<20s} {d['drop_in_auc']:+.4f} AUC")
    print(f"\n{metrics['active_students']} active students, "
          f"${metrics['total_value_at_risk']:,.0f} of margin at risk over 3 months")
    print(f"top 25 carry ${metrics['top25_value_at_risk']:,.0f} "
          f"({metrics['top25_share']:.0%} of it)")
    print("\ncall list head:")
    print(top.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
