"""
Which centers are actually underperforming, and why the league table is noise.

A franchise owner with fourteen centers wants a ranking. The obvious one is
churn rate per center, sorted. It is almost entirely useless, and the reason is
sample size: the smallest center on this list has 38 students, so a difference
of three departures moves it eight places.

Ranking on a noisy estimate ranks the noise. The smallest units have the widest
sampling distributions, so they occupy both ends of every league table ever
built, and the owner then flies out to fix the one at the bottom, which was
average all along.

The fix is partial pooling. Model each center's true churn rate as a draw from a
population distribution, fit that distribution from all fourteen at once, and
report each center's posterior rather than its raw rate. Small centers get
pulled hard toward the population mean because their own data says little; large
centers barely move because theirs says a lot. Nobody has to choose a shrinkage
factor: the data sets it.

With a Beta population and Binomial observations the posterior is closed form,

    y_j ~ Binomial(n_j, p_j),   p_j ~ Beta(a, b)
    p_j | y_j ~ Beta(a + y_j, b + n_j - y_j)

and the marginal likelihood of (a, b) after integrating the p_j out is the
Beta-Binomial, which is what gets maximised here. That is empirical Bayes; the
honest caveat is stated at the bottom.

Run:  python src/hierarchical.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

TUITION, MARGIN = 315.0, 0.45
VISIT_COST = 4200.0   # a week of the regional director's time and travel

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
})


def simulate(seed=17, n_centers=14):
    """Fourteen centers. True quarterly churn varies genuinely but modestly;
    roster sizes vary by a factor of six, which is what does the damage."""
    rng = np.random.default_rng(seed)
    n = rng.integers(38, 240, n_centers)
    # True rates from a Beta with mean about 0.14. The spread is real but small
    # relative to the binomial noise on a 45-student center, which is exactly the
    # regime where a league table stops meaning anything.
    a_true, b_true = 7.0, 43.0
    p = rng.beta(a_true, b_true, n_centers)
    y = rng.binomial(n, p)
    # a second independent quarter, used only to test whether a ranking holds up
    y2 = rng.binomial(n, p)
    names = [f"center {chr(65 + i)}" for i in range(n_centers)]
    return dict(names=names, n=n, y=y, y2=y2, p_true=p,
                a_true=a_true, b_true=b_true)


# ---------------------------------------------------------------------------
# Empirical Bayes
# ---------------------------------------------------------------------------

def fit_beta_binomial(y, n):
    """Maximise the Beta-Binomial marginal likelihood in (a, b).

        P(y | a, b) = C(n, y) B(y + a, n - y + b) / B(a, b)

    Parameterised as (log mu/(1-mu), log kappa) so the optimiser works on the
    real line and the mean and the concentration move independently.
    """
    def nll(theta):
        mu = 1 / (1 + np.exp(-theta[0]))
        kappa = np.exp(theta[1])
        a, b = mu * kappa, (1 - mu) * kappa
        ll = (stats.betabinom.logpmf(y, n, a, b)).sum()
        return -ll if np.isfinite(ll) else 1e12

    # A coarse grid first, then a local refine. Two parameters on 14 points give
    # a likelihood flat enough that an unconstrained simplex wanders into the
    # degenerate corner where kappa -> infinity and every center is the mean.
    # Grid, then refine, then take whichever is better.
    grid_mu = np.linspace(0.03, 0.45, 60)
    grid_k = np.exp(np.linspace(np.log(3), np.log(3000), 90))
    best, bx = np.inf, None
    for m in grid_mu:
        for k in grid_k:
            v = nll([np.log(m / (1 - m)), np.log(k)])
            if v < best:
                best, bx = v, [np.log(m / (1 - m)), np.log(k)]
    r = optimize.minimize(nll, bx, method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-6})
    x = r.x if r.fun < best else bx
    mu = 1 / (1 + np.exp(-x[0])); kappa = np.exp(x[1])
    return mu * kappa, (1 - mu) * kappa, mu, kappa


def rank_stability(y1, y2, n, a, b):
    """Rank the centers on quarter 1, then check the ranking against quarter 2.

    This is the empirical version of the argument. If a ranking is real it
    reproduces; if it is sampling noise it does not, and Spearman correlation
    between the two quarters says which. No appeal to theory required.
    """
    naive1, naive2 = y1 / n, y2 / n
    post1 = (a + y1) / (a + b + n)
    post2 = (a + y2) / (a + b + n)
    return {
        "naive": float(stats.spearmanr(naive1, naive2).statistic),
        "pooled": float(stats.spearmanr(post1, post2).statistic),
    }


def sample_size_sweep(sizes=(10, 14, 25, 50, 100, 200), reps=25, seed=5):
    """How many centers does empirical Bayes need before it stops over-shrinking?

    The hyperparameters are fitted from the same data and then treated as known,
    which ignores their own uncertainty. With few groups the marginal likelihood
    is biased toward a large concentration, so the shrinkage is too strong. This
    measures where that stops mattering instead of asserting that it does not.
    """
    rows = []
    for K in sizes:
        kap, mse_n, mse_p, mae_n, mae_p = [], [], [], [], []
        for r in range(reps):
            d = simulate(seed=seed + r * 101, n_centers=K)
            a, b, mu, kappa = fit_beta_binomial(d["y"], d["n"])
            post = (a + d["y"]) / (a + b + d["n"])
            naive = d["y"] / d["n"]
            kap.append(kappa)
            mse_n.append(np.mean((naive - d["p_true"]) ** 2))
            mse_p.append(np.mean((post - d["p_true"]) ** 2))
            mae_n.append(np.mean(np.abs(naive - d["p_true"])))
            mae_p.append(np.mean(np.abs(post - d["p_true"])))
        rows.append({"centers": K, "median_kappa": float(np.median(kap)),
                     "mse_naive": float(np.mean(mse_n)), "mse_pooled": float(np.mean(mse_p)),
                     "mae_naive": float(np.mean(mae_n)), "mae_pooled": float(np.mean(mae_p)),
                     "mse_gain": float(1 - np.mean(mse_p) / np.mean(mse_n))})
    return rows


def main():
    CHARTS.mkdir(exist_ok=True)
    d = simulate()
    names, n, y, y2, p_true = d["names"], d["n"], d["y"], d["y2"], d["p_true"]

    a, b, mu, kappa = fit_beta_binomial(y, n)
    naive = y / n
    post_mean = (a + y) / (a + b + n)
    lo = stats.beta.ppf(0.05, a + y, b + n - y)
    hi = stats.beta.ppf(0.95, a + y, b + n - y)

    # P(this center is genuinely worse than the population mean)
    p_worse = 1 - stats.beta.cdf(mu, a + y, b + n - y)

    err_naive = float(np.abs(naive - p_true).mean())
    err_pooled = float(np.abs(post_mean - p_true).mean())
    mse_naive = float(np.mean((naive - p_true) ** 2))
    mse_pooled = float(np.mean((post_mean - p_true) ** 2))
    sweep = sample_size_sweep()
    stab = rank_stability(y, y2, n, a, b)

    worst_naive = int(np.argmax(naive))
    worst_pooled = int(np.argmax(post_mean))
    confident = [i for i in range(len(n)) if p_worse[i] > 0.90]

    # ---- chart ----------------------------------------------------------
    order = np.argsort(-post_mean)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    yy = np.arange(len(order))
    for k, i in enumerate(order):
        ax.plot([lo[i] * 100, hi[i] * 100], [k, k], color=GRID, linewidth=6,
                solid_capstyle="butt", zorder=1)
    ax.scatter(naive[order] * 100, yy, s=52, color=ORANGE, zorder=3, label="raw rate")
    ax.scatter(post_mean[order] * 100, yy, s=52, color=BLUE, zorder=3, label="after pooling")
    for k, i in enumerate(order):
        ax.annotate("", xy=(post_mean[i] * 100, k), xytext=(naive[i] * 100, k),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, alpha=.65))
    ax.axvline(mu * 100, color=AQUA, linewidth=1.4, linestyle=(0, (5, 3)))
    ax.annotate(f"population mean {mu*100:.1f}%", (mu * 100, -1.0),
                fontsize=9, color=AQUA, ha="center")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{names[i]}  n={n[i]}" for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("quarterly churn, %")
    ax.set_title("Small centers move furthest, because they said least",
                 loc="left", fontsize=13, color=INK, pad=22)
    ax.text(0, 1.05, "Raw rate, posterior after partial pooling, and a 90% interval",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="lower right")

    ax = axes[1]
    bars = [("raw rate", stab["naive"], ORANGE), ("pooled", stab["pooled"], BLUE)]
    x = np.arange(2)
    ax.bar(x, [v for _, v, _ in bars], 0.5, color=[c for _, _, c in bars])
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rank correlation, quarter 1 vs quarter 2")
    ax.set_title("Does the league table reproduce?", loc="left",
                 fontsize=13, color=INK, pad=22)
    ax.text(0, 1.05, "Rank on one quarter, check it against the next",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(CHARTS / "hierarchical.png", bbox_inches="tight")
    plt.close(fig)

    out = {
        "a": float(a), "b": float(b), "mu": float(mu), "kappa": float(kappa),
        "mae_naive": err_naive, "mae_pooled": err_pooled,
        "mse_naive": mse_naive, "mse_pooled": mse_pooled,
        "mse_improvement": 1 - mse_pooled / mse_naive,
        "improvement": 1 - err_pooled / err_naive,
        "sweep": sweep, "kappa_true": 50.0,
        "rank_stability": stab,
        "worst_naive": names[worst_naive], "worst_naive_n": int(n[worst_naive]),
        "worst_pooled": names[worst_pooled], "worst_pooled_n": int(n[worst_pooled]),
        "confidently_worse": [names[i] for i in confident],
        "centers": [{"name": names[i], "n": int(n[i]), "events": int(y[i]),
                     "naive": float(naive[i]), "posterior": float(post_mean[i]),
                     "lo90": float(lo[i]), "hi90": float(hi[i]),
                     "p_worse_than_mean": float(p_worse[i]),
                     "true": float(p_true[i])} for i in order],
    }
    (ROOT / "results_hierarchical.json").write_text(json.dumps(out, indent=2))

    print(f"fitted population: mean {mu:.3f}, concentration {kappa:.1f} "
          f"(Beta({a:.1f}, {b:.1f}))")
    print(f"prior weight equals {a + b:.0f} students, so a center with n={n.min()} "
          f"is pulled {(a + b) / (a + b + n.min()):.0%} of the way to the mean\n")
    print(f"error against the true rate, this sample")
    print(f"  MAE  raw {err_naive:.4f}  pooled {err_pooled:.4f}")
    print(f"  MSE  raw {mse_naive:.6f}  pooled {mse_pooled:.6f} "
          f"({out['mse_improvement']:+.0%})\n")
    print(f"{'centers':>9}{'median kappa':>14}{'MSE raw':>11}{'MSE pooled':>13}{'gain':>8}")
    for r in sweep:
        print(f"{r['centers']:>9}{r['median_kappa']:>14.0f}{r['mse_naive']:>11.5f}"
              f"{r['mse_pooled']:>13.5f}{r['mse_gain']:>8.0%}")
    print(f"  (true kappa is 50; anything much above it is over-shrinkage)\n")
    print(f"rank correlation between quarter 1 and quarter 2")
    print(f"  raw rate : {stab['naive']:.2f}")
    print(f"  pooled   : {stab['pooled']:.2f}\n")
    print(f"worst center by raw rate  : {out['worst_naive']} (n={out['worst_naive_n']})")
    print(f"worst center after pooling: {out['worst_pooled']} (n={out['worst_pooled_n']})")
    print(f"\ncenters we can say are genuinely above the mean with 90% confidence: "
          f"{out['confidently_worse'] or 'none'}")
    print(f"cost of a director visit: ${VISIT_COST:,.0f}. "
          f"Number of visits the raw table justifies: 0.")


if __name__ == "__main__":
    main()
