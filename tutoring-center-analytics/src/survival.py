"""
Student retention as a time-to-event problem.

Four things happen here, in order, and each one exists because the previous one
was not good enough.

1.  Kaplan-Meier, because 41% of these students had not left when the data was
    cut and averaging their tenure with the students who did leave is simply
    wrong arithmetic.
2.  Cox proportional hazards, because the owner wants to know which students,
    not just how many.
3.  A gamma-frailty parametric model fitted by maximum likelihood, because the
    falling hazard has two possible causes with opposite operational
    implications and the Cox model is silent on which one it is.
4.  Lifetime value recomputed off the survival curve, because ARPU divided by
    churn rate is a formula with an assumption in it that this data violates.

Run:  python src/survival.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
TUITION = 315.0
HORIZON = 30.0        # months; the observation window, and so the RMST horizon
MARGIN = 0.45         # contribution margin: tuition net of instructor time and space
DISCOUNT_ANNUAL = 0.15  # cost of capital for a single-location small business

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
})


def _frame(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=20 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Kaplan-Meier
# ---------------------------------------------------------------------------

def km_curve(t, d):
    """Kaplan-Meier estimate. Returns (times, survival), both starting at (0, 1)."""
    sf = SurvfuncRight(t, d)
    times = np.concatenate([[0.0], sf.surv_times])
    surv = np.concatenate([[1.0], sf.surv_prob])
    return times, surv


def rmst(times, surv, horizon):
    """Restricted mean survival time: the area under S(t) out to `horizon`.

    This is the honest summary when the tail is not observed. The unrestricted
    mean requires extrapolating past the data, and under a heavy-tailed frailty
    distribution that extrapolation is where all the mass is.
    """
    t = np.concatenate([times[times < horizon], [horizon]])
    s = np.concatenate([surv[times < horizon], [np.interp(horizon, times, surv)]])
    return float(np.sum(np.diff(t) * s[:-1]))


def km_median(times, surv):
    below = np.where(surv <= 0.5)[0]
    return float(times[below[0]]) if len(below) else float("nan")


def part1_kaplan_meier(df):
    t, d = df.tenure_months.values, df.churned.values
    times, surv = km_curve(t, d)

    naive_median = float(np.median(t))
    km_med = km_median(times, surv)
    overall_rmst = rmst(times, surv, HORIZON)

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.step(times, surv, where="post", color=BLUE, linewidth=2)
    ax.fill_between(times, surv, step="post", color=BLUE, alpha=0.07)
    _frame(ax, "How long students stay",
           f"Kaplan-Meier survival, n={len(df):,}, "
           f"{(1-d.mean())*100:.0f}% right censored")
    ax.set_xlabel("months since enrollment")
    ax.set_ylabel("share still enrolled")
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.plot([km_med, km_med], [0, 0.5], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"median {km_med:.1f} months",
                (km_med, 0.5), xytext=(12, 10), textcoords="offset points",
                fontsize=9.5, color=INK)
    ax.annotate(f"naive median of the tenure column: {naive_median:.1f} months\n"
                f"(wrong: it treats a censored spell as a completed one)",
                (HORIZON * 0.42, 0.80), fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(CHARTS / "km_overall.png", bbox_inches="tight")
    plt.close(fig)
    return {"km_median": km_med, "naive_median": naive_median, "rmst": overall_rmst}


def part1b_by_channel(df):
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    colors = {"organic": BLUE, "paid": ORANGE, "referral": AQUA, "walk_in": YELLOW}
    out = {}
    for ch, color in colors.items():
        sub = df[df.channel == ch]
        times, surv = km_curve(sub.tenure_months.values, sub.churned.values)
        ax.step(times, surv, where="post", color=color, linewidth=2, label=ch)
        r = rmst(times, surv, HORIZON)
        out[ch] = {"n": int(len(sub)), "rmst": r, "km_median": km_median(times, surv)}
        # Direct label. Required: slots 3 and 4 warn on surface contrast,
        # so identity must not rest on colour alone.
        y = np.interp(HORIZON * 0.97, times, surv)
        ax.annotate(ch.replace("_", " "), (HORIZON * 0.97, y), xytext=(6, -3),
                    textcoords="offset points", fontsize=9.5, color=INK, va="center")

    chi2, p = survdiff(df.tenure_months.values, df.churned.values, df.channel.values)
    _frame(ax, "Retention is a channel property",
           f"Kaplan-Meier by acquisition channel · log-rank chi2={chi2:.1f}, p={p:.1e}")
    ax.set_xlabel("months since enrollment")
    ax.set_ylabel("share still enrolled")
    ax.set_xlim(0, HORIZON * 1.28)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS / "km_by_channel.png", bbox_inches="tight")
    plt.close(fig)
    out["_logrank"] = {"chi2": float(chi2), "p": float(p)}
    return out


# ---------------------------------------------------------------------------
# 2. Cox proportional hazards
# ---------------------------------------------------------------------------

def part2_cox(df):
    X = pd.DataFrame({
        "paid": (df.channel == "paid").astype(float),
        "referral": (df.channel == "referral").astype(float),
        "walk_in": (df.channel == "walk_in").astype(float),
        "two_sessions": (df.sessions_per_week == 2).astype(float),
        "assessment_gap": df.assessment_gap.astype(float),
        "first8_attendance": (df.first8_attendance - df.first8_attendance.mean()).astype(float),
        "high_school": (df.grade_band == "high").astype(float),
    })
    mod = PHReg(df.tenure_months.values, X.values, status=df.churned.values)
    res = mod.fit()
    hr = pd.DataFrame({
        "term": X.columns,
        "coef": res.params,
        "hazard_ratio": np.exp(res.params),
        "se": res.bse,
        "p": 2 * (1 - stats.norm.cdf(np.abs(res.params / res.bse))),
    })
    hr["ci_lo"] = np.exp(res.params - 1.96 * res.bse)
    hr["ci_hi"] = np.exp(res.params + 1.96 * res.bse)
    return hr, res


# ---------------------------------------------------------------------------
# 3. Gamma-frailty MLE, written out by hand
# ---------------------------------------------------------------------------

def gamma_frailty_mle(t, d, X):
    """Maximum likelihood for the unconditional (marginal) gamma-frailty model.

        h_i(t) = z_i * lambda * exp(b'x_i),   z_i ~ Gamma(1/theta, theta), E[z]=1
        S(t|x) = (1 + theta * lambda_x * t) ** (-1/theta)
        f(t|x) = lambda_x * (1 + theta * lambda_x * t) ** (-1/theta - 1)

    so the log likelihood collapses to

        LL = sum_i [ d_i * log(lambda_x_i) - (1/theta + d_i) * log(1 + theta * lambda_x_i * t_i) ]

    theta is the frailty variance. theta -> 0 recovers the exponential model with
    a flat hazard, so the fitted theta is a direct test of whether the falling
    population hazard needs heterogeneity to explain it.
    """
    X = np.asarray(X, float)
    n, k = X.shape

    def nll(p):
        log_lam, log_theta, b = p[0], p[1], p[2:]
        lam_x = np.exp(log_lam + X @ b)
        theta = np.exp(log_theta)
        u = 1.0 + theta * lam_x * t
        if not np.all(np.isfinite(u)) or np.any(u <= 0):
            return 1e12
        ll = d * np.log(lam_x) - (1.0 / theta + d) * np.log(u)
        return -float(np.sum(ll))

    p0 = np.concatenate([[np.log(0.06), np.log(0.4)], np.zeros(k)])
    fit = optimize.minimize(nll, p0, method="L-BFGS-B")

    # Standard errors from a numerical Hessian of the negative log likelihood
    eps = 1e-5
    H = np.zeros((len(fit.x), len(fit.x)))
    for i in range(len(fit.x)):
        for j in range(len(fit.x)):
            a, b_, c_, e = fit.x.copy(), fit.x.copy(), fit.x.copy(), fit.x.copy()
            a[i] += eps; a[j] += eps
            b_[i] += eps; b_[j] -= eps
            c_[i] -= eps; c_[j] += eps
            e[i] -= eps; e[j] -= eps
            H[i, j] = (nll(a) - nll(b_) - nll(c_) + nll(e)) / (4 * eps ** 2)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    return {
        "lambda": float(np.exp(fit.x[0])),
        "theta": float(np.exp(fit.x[1])),
        "theta_se_logscale": float(se[1]),
        "beta": fit.x[2:],
        "beta_se": se[2:],
        "loglik": float(-fit.fun),
        "converged": bool(fit.success),
    }


def exponential_mle(t, d, X):
    """Nested model with theta pinned at 0, for the likelihood ratio test."""
    X = np.asarray(X, float)

    def nll(p):
        lam_x = np.exp(p[0] + X @ p[1:])
        return -float(np.sum(d * np.log(lam_x) - lam_x * t))

    fit = optimize.minimize(nll, np.concatenate([[np.log(0.06)], np.zeros(X.shape[1])]),
                            method="L-BFGS-B")
    return {"lambda": float(np.exp(fit.x[0])), "beta": fit.x[1:], "loglik": float(-fit.fun)}


def part3_frailty(df, truth):
    X = pd.DataFrame({
        "paid": (df.channel == "paid").astype(float),
        "referral": (df.channel == "referral").astype(float),
        "walk_in": (df.channel == "walk_in").astype(float),
        "two_sessions": (df.sessions_per_week == 2).astype(float),
        "assessment_gap": df.assessment_gap.astype(float),
        "first8_attendance": (df.first8_attendance - df.first8_attendance.mean()).astype(float),
        "high_school": (df.grade_band == "high").astype(float),
    })
    t, d = df.tenure_months.values, df.churned.values

    fr = gamma_frailty_mle(t, d, X)
    ex = exponential_mle(t, d, X)

    # Likelihood ratio test of H0: theta = 0. The null sits on the boundary of the
    # parameter space, so the reference distribution is the 50:50 mixture of a
    # point mass at 0 and chi2(1), and the naive chi2(1) p-value is halved.
    lr = 2 * (fr["loglik"] - ex["loglik"])
    p_boundary = 0.5 * (1 - stats.chi2.cdf(lr, 1))

    # Empirical hazard, binned, against the two fitted population hazards
    edges = np.arange(0, HORIZON + 1.5, 1.5)
    mid, emp = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        at_risk = np.sum(t >= lo)
        if at_risk < 25:
            continue
        events = np.sum((t >= lo) & (t < hi) & (d == 1))
        exposure = np.sum(np.clip(t - lo, 0, hi - lo))
        if exposure > 0:
            mid.append((lo + hi) / 2)
            emp.append(events / exposure)
    mid, emp = np.array(mid), np.array(emp)

    grid = np.linspace(0.01, HORIZON, 300)
    lam_bar = fr["lambda"] * np.exp(float(np.mean(X.values @ fr["beta"])))
    h_frailty = lam_bar / (1 + fr["theta"] * lam_bar * grid)
    h_exp = np.full_like(grid, ex["lambda"] * np.exp(float(np.mean(X.values @ ex["beta"]))))

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.plot(mid, emp, "o", color=MUTED, markersize=6, label="observed")
    ax.plot(grid, h_frailty, color=BLUE, linewidth=2, label="gamma frailty")
    ax.plot(grid, h_exp, color=ORANGE, linewidth=2, linestyle=(0, (5, 3)), label="exponential")
    _frame(ax, "The churn rate falls, and no student changes",
           "Monthly hazard of leaving, by months since enrollment")
    ax.set_xlabel("months since enrollment")
    ax.set_ylabel("monthly hazard")
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, max(emp.max(), h_frailty.max()) * 1.28)
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper right", handlelength=1.4)
    ax.annotate(f"observed hazard falls {(emp[0]-emp[-1])/emp[0]*100:.0f}% across the window\n"
                f"while every individual hazard stays flat",
                (grid[110], h_exp[0] * 0.32), fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(CHARTS / "hazard_decline.png", bbox_inches="tight")
    plt.close(fig)

    return {"frailty": fr, "exponential": ex, "lr_stat": float(lr),
            "p_boundary": float(p_boundary), "terms": list(X.columns)}


# ---------------------------------------------------------------------------
# 4. Lifetime value, done twice
# ---------------------------------------------------------------------------

def part4_ltv(df, by_channel):
    """Naive LTV against survival-based LTV, per channel, against CAC.

    Three numbers for the same student, each one a correction of the last.

      naive     LTV = ARPU / pooled_churn
                Exact if and only if the hazard is constant. It is not, and this
                is the number on every marketing dashboard.

      rmst      LTV = margin * integral_0^H S(t) dt
                No constant-hazard assumption, no extrapolation past the data.

      npv       LTV = margin * integral_0^H S(t) v^t dt,  v = (1+r)^(-1/12)
                Tuition collected in month 26 is not worth what tuition collected
                next month is worth. Discounting penalises exactly the channels
                whose value sits furthest out.

    Gross revenue is a vanity metric here. An hour of tutoring has an instructor
    behind it, so the figure that pays back acquisition is contribution margin.
    """
    v = (1 + DISCOUNT_ANNUAL) ** (-1 / 12)
    rows = []
    for ch, stats_ in by_channel.items():
        if ch.startswith("_"):
            continue
        sub = df[df.channel == ch]
        times, surv = km_curve(sub.tenure_months.values, sub.churned.values)

        grid = np.arange(0.0, HORIZON, 1 / 12)
        s_grid = np.interp(grid, times, surv)
        rmst_ = float(np.sum(s_grid) / 12)
        npv_months = float(np.sum(s_grid * v ** grid) / 12)

        exposure = sub.tenure_months.sum()
        pooled_churn = sub.churned.sum() / exposure
        cac = float(sub.cac.iloc[0])

        # CAC payback: first month at which cumulative discounted contribution
        # margin covers acquisition cost. Undefined if it never does.
        cum = np.cumsum(s_grid * v ** grid) / 12 * TUITION * MARGIN
        hit = np.where(cum >= cac)[0]
        payback = float(grid[hit[0]]) if len(hit) else float("nan")

        rows.append({
            "channel": ch, "n": stats_["n"], "cac": cac,
            "pooled_monthly_churn": pooled_churn,
            "rmst_months": rmst_,
            "naive_ltv": TUITION * MARGIN / pooled_churn,
            "rmst_ltv": TUITION * MARGIN * rmst_,
            "npv_ltv": TUITION * MARGIN * npv_months,
            "payback_months": payback,
        })
    t = pd.DataFrame(rows)
    t["naive_ratio"] = t.naive_ltv / t.cac
    t["npv_ratio"] = t.npv_ltv / t.cac
    t = t.sort_values("npv_ltv", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    x = np.arange(len(t))
    w = 0.27
    ax.bar(x - w, t.naive_ltv, w * 0.92, color=ORANGE, label="naive: ARPU / churn")
    ax.bar(x, t.rmst_ltv, w * 0.92, color=BLUE, label="from the survival curve")
    ax.bar(x + w, t.npv_ltv, w * 0.92, color=AQUA, label="survival curve, discounted")
    for i, r in t.iterrows():
        for dx, val in [(-w, r.naive_ltv), (0, r.rmst_ltv), (w, r.npv_ltv)]:
            ax.text(i + dx, val, f"{val:,.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK, rotation=90)
    _frame(ax, "One student, three valuations",
           f"Contribution margin per enrolled student · {MARGIN:.0%} margin, "
           f"{HORIZON:.0f}-month horizon")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in t.channel])
    ax.set_ylabel("dollars per student")
    ax.set_ylim(0, t.naive_ltv.max() * 1.34)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="upper right",
              ncols=1, handlelength=1.1)

    ax = axes[1]
    ax.barh(x, t.payback_months, 0.52, color=BLUE)
    for i, r in t.iterrows():
        ax.text(r.payback_months, i, f"  {r.payback_months:.1f} mo",
                va="center", fontsize=9.5, color=INK)
    _frame(ax, "How long until a student pays for themselves",
           "Months to recover acquisition cost")
    ax.set_yticks(x)
    ax.set_yticklabels([c.replace("_", " ") for c in t.channel])
    ax.invert_yaxis()
    ax.set_xlabel("months")
    ax.set_xlim(0, t.payback_months.max() * 1.35)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(CHARTS / "ltv_by_channel.png", bbox_inches="tight")
    plt.close(fig)
    return t


# ---------------------------------------------------------------------------
# 5. What the frailty does to the Cox estimates
# ---------------------------------------------------------------------------

def part5_attenuation(df, fr, truth):
    """Unobserved heterogeneity biases Cox coefficients toward zero.

    This is not a bug in the estimator. The Cox model estimates a *population*
    hazard ratio, and under frailty the population is a moving target: at any
    time t the high-hazard group has already been thinned of its frailest
    members more aggressively than the low-hazard group, so the two groups look
    more alike as t grows. The estimand shrinks toward 1 with time, and the
    partial likelihood averages over that.

    The marginal frailty model estimates the *individual* coefficient instead,
    so it should recover the generating beta. Here the truth is known, so both
    claims are checkable rather than assertable.
    """
    hr, _ = part2_cox(df)
    name_map = {
        "paid": "channel_paid", "referral": "channel_referral",
        "two_sessions": "sessions_2_per_week", "assessment_gap": "assessment_gap",
        "first8_attendance": "first8_attendance", "high_school": "grade_high",
    }
    rows = []
    for i, term in enumerate(fr["terms"]):
        if term not in name_map:
            continue
        rows.append({
            "term": term,
            "true_beta": truth["beta"][name_map[term]],
            "cox_beta": float(hr.loc[hr.term == term, "coef"].iloc[0]),
            "frailty_beta": float(fr["frailty"]["beta"][i]),
        })
    a = pd.DataFrame(rows)
    a["cox_shrinkage"] = 1 - a.cox_beta / a.true_beta
    a["frailty_shrinkage"] = 1 - a.frailty_beta / a.true_beta

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(a))
    ax.plot(a.true_beta, y, "o", color=INK, markersize=9, label="true", zorder=3)
    ax.plot(a.cox_beta, y, "o", color=ORANGE, markersize=8, label="Cox", zorder=3)
    ax.plot(a.frailty_beta, y, "o", color=BLUE, markersize=8, label="frailty MLE", zorder=3)
    for i, r in a.iterrows():
        lo, hi = sorted([r.true_beta, r.cox_beta])
        ax.plot([lo, hi], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    _frame(ax, "Frailty pulls the Cox coefficients toward zero",
           "Log hazard ratios: the generating truth, and what each model recovered")
    ax.set_yticks(y)
    ax.set_yticklabels(a.term)
    ax.invert_yaxis()
    ax.set_xlabel("log hazard ratio")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=MUTED, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "attenuation.png", bbox_inches="tight")
    plt.close(fig)
    return a


def main():
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "students.csv")
    truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())

    km = part1_kaplan_meier(df)
    by_ch = part1b_by_channel(df)
    hr, _ = part2_cox(df)
    fr = part3_frailty(df, truth)
    ltv = part4_ltv(df, by_ch)
    att = part5_attenuation(df, fr, truth)

    out = {"km": km, "by_channel": by_ch, "cox": hr.to_dict("records"),
           "frailty": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in fr["frailty"].items()},
           "lr_stat": fr["lr_stat"], "p_boundary": fr["p_boundary"],
           "terms": fr["terms"], "ltv": ltv.to_dict("records"), "attenuation": att.to_dict("records"),
           "truth": truth}
    (ROOT / "results_survival.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"KM median {km['km_median']:.1f} mo (naive median {km['naive_median']:.1f}) "
          f"| RMST {km['rmst']:.1f} mo")
    print(f"log-rank across channels: p={by_ch['_logrank']['p']:.2e}")
    print("\nCox hazard ratios:")
    print(hr[["term", "hazard_ratio", "ci_lo", "ci_hi", "p"]].round(3).to_string(index=False))
    print(f"\nfrailty theta = {fr['frailty']['theta']:.3f} (true {truth['theta']}), "
          f"lambda = {fr['frailty']['lambda']:.4f} (true {truth['lambda']})")
    print(f"LR vs exponential: {fr['lr_stat']:.1f}, boundary p = {fr['p_boundary']:.2e}")
    print("\nLTV (contribution margin):")
    print(ltv[["channel", "cac", "naive_ltv", "rmst_ltv", "npv_ltv",
               "naive_ratio", "npv_ratio", "payback_months"]].round(2).to_string(index=False))
    print("\nCoefficient attenuation:")
    print(att.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
