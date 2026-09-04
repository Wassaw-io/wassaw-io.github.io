"""
Who to call, which is not who is going to leave.

risk_model.py ranks students by how likely they are to go. That is the wrong
list, and the gap between the two is where retention budgets die.

Split the roster four ways by what a retention call actually does:

    persuadables   stay only if you call        <- the entire return
    sure things    stay whether you call or not <- wasted call
    lost causes    leave whether you call or not<- wasted call
    sleeping dogs  leave only because you called<- worse than wasted

A churn model scores the union of persuadables, lost causes and sleeping dogs,
because all three are high risk. Only the first pays. What you want is not

    P(churn | x)

but the difference the treatment makes,

    tau(x) = P(stay | treated, x) - P(stay | untreated, x)

which is a causal quantity and therefore not estimable from observational data
at all. It needs an experiment. This file simulates one, estimates tau four
ways, and prices the difference between targeting on tau and targeting on risk.

The fundamental problem: no student is ever observed both called and not called,
so tau(x) has no label to fit against. Every estimator below is a way of routing
around that.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

import generate_students

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

TUITION, MARGIN = 315.0, 0.45
CALL_COST = 18.0        # staff time for one retention call and its follow-up
OUTCOME_MONTH = 12      # "retained" means still enrolled at 12 months
VALUE_OF_RETENTION = TUITION * MARGIN * 9.0   # margin on the months a save buys

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
})

FEATURES = ["paid", "referral", "walk_in", "two_sessions",
            "assessment_gap", "first8_attendance", "high_school"]


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def true_tau_multiplier(att, gap):
    """Log-hazard shift from a retention call at month 3.

    Deliberately non-monotone in attendance, because that is what makes the
    problem interesting and it is also what actually happens.

      att around 0.72   a wavering student. The call lands. Large benefit.
      att below 0.5     already disengaged. Nothing to save.
      att above 0.92    would have stayed anyway. Nothing to add,
                        and if they are also struggling academically, a call
                        that names leaving as an option can put it on the table.

    Note the shape against risk: churn risk falls monotonically in attendance,
    so the highest-risk students sit in the lost-cause region. Ranking by risk
    targets the students the call cannot help.
    """
    persuadable = -1.05 * np.exp(-(((att - 0.72) / 0.115) ** 2))
    sleeping_dog = 0.30 * (att > 0.92) * (gap > 2.0)
    return persuadable + sleeping_dog


def simulate_experiment(seed=21, n=6000):
    """A randomised retention-call trial. Half the roster gets the call."""
    gs = generate_students.CONFIG.copy()
    gs["n_students"] = n
    gs["seed"] = seed
    old = generate_students.CONFIG
    generate_students.CONFIG = gs
    try:
        df = generate_students.build()
    finally:
        generate_students.CONFIG = old

    rng = np.random.default_rng(seed + 1)
    lam = np.asarray(df.attrs["truth"]["individual_lambda"], float)
    theta = df.attrs["truth"]["theta"]
    z = rng.gamma(1 / theta, theta, len(df))

    tau_log = true_tau_multiplier(df.first8_attendance.values, df.assessment_gap.values)
    treated = rng.integers(0, 2, len(df))          # randomised, 50/50

    # Potential outcomes. Both are computed; only one is revealed, which is the
    # whole difficulty of the problem and the reason a simulation is worth having.
    base = z * lam
    t0 = rng.exponential(1 / base)
    # Common random numbers: same frailty and same uniform draw under treatment,
    # so the contrast is the treatment effect and not simulation noise.
    u = np.exp(-base * t0)
    t1 = -np.log(u) / (base * np.exp(tau_log))

    y0 = (t0 >= OUTCOME_MONTH).astype(int)
    y1 = (t1 >= OUTCOME_MONTH).astype(int)

    out = pd.DataFrame({
        "student_id": df.student_id,
        "paid": (df.channel == "paid").astype(float),
        "referral": (df.channel == "referral").astype(float),
        "walk_in": (df.channel == "walk_in").astype(float),
        "two_sessions": (df.sessions_per_week == 2).astype(float),
        "assessment_gap": df.assessment_gap.astype(float),
        "first8_attendance": df.first8_attendance.astype(float),
        "high_school": (df.grade_band == "high").astype(float),
        "treated": treated,
        "retained": np.where(treated == 1, y1, y0),
        # kept only for grading the estimators, never fed to them
        "_true_tau": (y1 - y0).astype(float),
        "_y0": y0, "_y1": y1,
    })
    return out


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _gbm(seed=0):
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=260,
        l2_regularization=1.0, min_samples_leaf=45, random_state=seed)


def t_learner(tr, te, seed=0):
    """One model per arm. Simple, and it wastes half the data on each."""
    m1 = _gbm(seed).fit(tr[tr.treated == 1][FEATURES], tr[tr.treated == 1].retained)
    m0 = _gbm(seed).fit(tr[tr.treated == 0][FEATURES], tr[tr.treated == 0].retained)
    return m1.predict_proba(te[FEATURES])[:, 1] - m0.predict_proba(te[FEATURES])[:, 1]


def s_learner(tr, te, seed=0):
    """One model, treatment as a feature. Uses all the data, but a boosted tree
    will happily ignore a single binary column whose effect is small, which
    biases tau toward zero."""
    X = tr[FEATURES + ["treated"]]
    m = _gbm(seed).fit(X, tr.retained)
    a = te[FEATURES].assign(treated=1)
    b = te[FEATURES].assign(treated=0)
    return m.predict_proba(a)[:, 1] - m.predict_proba(b)[:, 1]


def x_learner(tr, te, seed=0):
    """Impute each unit's missing potential outcome with the other arm's model,
    fit tau on the imputed contrasts, then blend the two by propensity. Built
    for imbalanced arms; here the arms are balanced, so it should roughly tie
    the T-learner, and it does."""
    t1, t0 = tr[tr.treated == 1], tr[tr.treated == 0]
    m1 = _gbm(seed).fit(t1[FEATURES], t1.retained)
    m0 = _gbm(seed).fit(t0[FEATURES], t0.retained)
    d1 = t1.retained - m0.predict_proba(t1[FEATURES])[:, 1]
    d0 = m1.predict_proba(t0[FEATURES])[:, 1] - t0.retained
    from sklearn.ensemble import HistGradientBoostingRegressor
    g = dict(max_depth=3, learning_rate=0.06, max_iter=260,
             l2_regularization=1.0, min_samples_leaf=45, random_state=seed)
    tau1 = HistGradientBoostingRegressor(**g).fit(t1[FEATURES], d1).predict(te[FEATURES])
    tau0 = HistGradientBoostingRegressor(**g).fit(t0[FEATURES], d0).predict(te[FEATURES])
    e = tr.treated.mean()
    return e * tau0 + (1 - e) * tau1


def risk_score(tr, te, seed=0):
    """The wrong list, included so the comparison is against what people do."""
    m = _gbm(seed).fit(tr[FEATURES], tr.retained)
    return -m.predict_proba(te[FEATURES])[:, 1]   # high churn risk first


# ---------------------------------------------------------------------------
# Qini
# ---------------------------------------------------------------------------

def qini_curve(score, treated, y, n_points=100):
    """Incremental retained students as a function of how deep you call.

    At depth k, take the top k by score and compute

        Q(k) = R_t(k) - R_c(k) * N_t(k) / N_c(k)

    the treated responders minus the control responders rescaled to the treated
    group's size. A random-targeting policy traces the straight line from the
    origin to Q(N); area between the curve and that line is the Qini coefficient.
    """
    order = np.argsort(-score)
    t, yy = np.asarray(treated)[order], np.asarray(y)[order]
    ks = np.unique(np.linspace(1, len(t), n_points).astype(int))
    q = []
    for k in ks:
        nt, nc = t[:k].sum(), k - t[:k].sum()
        rt = yy[:k][t[:k] == 1].sum()
        rc = yy[:k][t[:k] == 0].sum()
        q.append(rt - (rc * nt / nc if nc > 0 else 0.0))
    return ks / len(t), np.array(q)


def qini_coefficient(x, q):
    rand = q[-1] * x
    return float(np.trapezoid(q - rand, x))


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

def policy_value(score, treated, y, depth):
    """Net margin from calling the top `depth` share, measured on the trial."""
    k = max(1, int(depth * len(score)))
    order = np.argsort(-score)[:k]
    t, yy = np.asarray(treated)[order], np.asarray(y)[order]
    nt, nc = t.sum(), (t == 0).sum()
    if nt == 0 or nc == 0:
        return 0.0
    lift = yy[t == 1].mean() - yy[t == 0].mean()   # retention gain per called student
    return float(k * (lift * VALUE_OF_RETENTION - CALL_COST))


def breakeven_analysis(scorer_s, risk_s, treated, y, costs):
    """At what intervention cost does targeting start to beat calling everyone?

    A phone call is cheap relative to a saved student, and when an intervention
    is that cheap the optimal policy is simply to apply it to everyone: the
    wasted calls cost less than the saves they buy. Targeting earns its keep
    only once the intervention itself is expensive, which is the case the moment
    it stops being a call and becomes an offer, a free month, or an hour of a
    director's time.

    This function finds the crossover instead of asserting it.
    """
    grid = np.linspace(0.03, 1.0, 40)
    rows = []
    for c in costs:
        def val(score, depth):
            k = max(1, int(depth * len(score)))
            o = np.argsort(-score)[:k]
            t, yy = np.asarray(treated)[o], np.asarray(y)[o]
            if t.sum() == 0 or (t == 0).sum() == 0:
                return 0.0
            lift = yy[t == 1].mean() - yy[t == 0].mean()
            return float(k * (lift * VALUE_OF_RETENTION - c))
        blanket = val(np.zeros(len(treated)), 1.0)
        tv = [val(scorer_s, d) for d in grid]
        rv = [val(risk_s, d) for d in grid]
        rows.append({"cost": float(c), "blanket": blanket,
                     "targeted_best": float(max(tv)),
                     "targeted_depth": float(grid[int(np.argmax(tv))]),
                     "risk_best": float(max(rv)),
                     "targeting_wins": bool(max(tv) > blanket)})
    return pd.DataFrame(rows)


def main():
    CHARTS.mkdir(exist_ok=True)
    df = simulate_experiment()
    tr, te = train_test_split(df, test_size=0.4, random_state=7, stratify=df.treated)

    ate = df[df.treated == 1].retained.mean() - df[df.treated == 0].retained.mean()

    scorers = {
        "T-learner": t_learner, "S-learner": s_learner,
        "X-learner": x_learner, "risk score (what people do)": risk_score,
    }
    results, curves = {}, {}
    for name, fn in scorers.items():
        s = fn(tr, te)
        x, q = qini_curve(s, te.treated.values, te.retained.values)
        curves[name] = (x, q)
        results[name] = {
            "qini": qini_coefficient(x, q),
            "corr_with_true_tau": float(np.corrcoef(s, te._true_tau)[0, 1]),
            "value_at_20pct": policy_value(s, te.treated.values, te.retained.values, 0.20),
            "value_at_40pct": policy_value(s, te.treated.values, te.retained.values, 0.40),
        }
    blanket = policy_value(np.zeros(len(te)), te.treated.values, te.retained.values, 1.0)

    # ---- chart -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    cols = {"T-learner": BLUE, "S-learner": AQUA, "X-learner": YELLOW,
            "risk score (what people do)": ORANGE}
    for name, (x, q) in curves.items():
        ax.plot(x * 100, q, color=cols[name], linewidth=2)
    # End labels, pushed apart: the three uplift curves converge at 100%.
    ends = sorted(((q[-1], n) for n, (x, q) in curves.items()), reverse=True)
    yr = max(q.max() for _, q in curves.values()) - min(q.min() for _, q in curves.values())
    step = yr * 0.085
    top = ends[0][0] + step * 1.2
    for i, (v, n) in enumerate(ends):
        y = top - i * step
        ax.annotate(n.split(" (")[0], (102, y), fontsize=9,
                    color=cols[n], va="center")
        ax.plot([100.5, 101.5], [v, y], color=cols[n], linewidth=1, alpha=.5)
    x0, q0 = curves["T-learner"]
    ax.plot([0, 100], [0, q0[-1]], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("call everyone at random", (52, q0[-1] * 0.52), fontsize=9, color=MUTED)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_title("Qini: students saved, by how deep you call",
                 loc="left", fontsize=13, color=INK, pad=20)
    ax.text(0, 1.03, "Incremental retentions above what the control arm did",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_xlabel("share of roster called, %")
    ax.set_ylabel("incremental students retained")
    ax.set_xlim(0, 118)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    ax = axes[1]
    grid = np.linspace(0.03, 1.0, 40)
    for name in ["T-learner", "risk score (what people do)"]:
        s = scorers[name](tr, te)
        v = [policy_value(s, te.treated.values, te.retained.values, d) for d in grid]
        ax.plot(grid * 100, v, color=cols[name], linewidth=2)
        best = grid[int(np.argmax(v))] * 100
        ax.plot([best], [max(v)], "o", color=cols[name], markersize=7)
        ax.annotate(f"{name.split(' (')[0]}: \\${max(v):,.0f} at {best:.0f}%",
                    (best, max(v)), xytext=(6, 8), textcoords="offset points",
                    fontsize=9.5, color=INK)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axhline(blanket, color=GRID, linewidth=1.5, linestyle=(0, (5, 3)))
    ax.annotate(f"call everyone: \\${blanket:,.0f}", (3, blanket), xytext=(0, 6),
                textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_title("Net margin by targeting policy", loc="left", fontsize=13, color=INK, pad=20)
    ax.text(0, 1.03, f"Call costs \\${CALL_COST:.0f}; a save is worth "
                     f"\\${VALUE_OF_RETENTION:,.0f} of margin",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_xlabel("share of roster called, %")
    ax.set_ylabel("net margin, dollars")
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(CHARTS / "uplift.png", bbox_inches="tight")
    plt.close(fig)

    # When is targeting worth doing at all?
    s_best = scorers["S-learner"](tr, te)
    s_risk = scorers["risk score (what people do)"](tr, te)
    be = breakeven_analysis(s_best, s_risk, te.treated.values, te.retained.values,
                            [18, 40, 80, 120, 160, 220, 300, 400])
    cross = be[be.targeting_wins]
    first_win = float(cross.cost.iloc[0]) if len(cross) else None

    out = {"breakeven": be.to_dict("records"), "targeting_wins_from_cost": first_win,
           "ate": float(ate), "n": int(len(df)), "test_n": int(len(te)),
           "blanket_value": blanket, "call_cost": CALL_COST,
           "value_of_retention": VALUE_OF_RETENTION, "models": results}
    (ROOT / "results_uplift.json").write_text(json.dumps(out, indent=2))

    print(f"randomised trial: {len(df):,} students, {df.treated.mean():.0%} treated")
    print(f"average treatment effect: {ate:+.4f} retention at {OUTCOME_MONTH} months")
    print(f"calling everyone nets ${blanket:,.0f}\n")
    print(f"{'model':<30s}{'qini':>8s}{'corr w/ true':>14s}{'$ @20%':>12s}{'$ @40%':>12s}")
    for k, v in results.items():
        print(f"{k:<30s}{v['qini']:>8.2f}{v['corr_with_true_tau']:>14.3f}"
              f"{v['value_at_20pct']:>12,.0f}{v['value_at_40pct']:>12,.0f}")
    print("\nwhen does targeting beat calling everyone?")
    show = be.copy()
    for c in ["blanket", "targeted_best", "risk_best"]:
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    show["targeted_depth"] = (be.targeted_depth * 100).map(lambda v: f"{v:.0f}%")
    show["cost"] = be.cost.map(lambda v: f"${v:.0f}")
    print(show.to_string(index=False))
    print(f"\ncrossover: targeting wins once the intervention costs "
          f"${first_win:,.0f} or more" if first_win else "\nblanket wins at every cost tested")


if __name__ == "__main__":
    main()
