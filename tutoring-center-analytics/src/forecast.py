"""
Seasonality, and the damage it does to a before-and-after comparison.

A tutoring centre is one of the most seasonal businesses there is. August and
September are enrollment season, December and June are dead, and the swing is
larger than most operating changes anyone will ever make. Two consequences, and
the first one is a criticism of the rest of this repository.

1.  README.md compares a "before" window to an "after" window. Those windows
    contain different months of the school year, so part of the measured
    improvement is calendar rather than management. This file quantifies how
    much, by removing the seasonal component first and re-running the same
    comparison. That correction is the honest headline number.

2.  Forecasting enrollment is a staffing decision before it is an analytics
    exercise. Instructors are hired weeks ahead, and the cost of being wrong is
    asymmetric: under-staffing in September loses students permanently,
    over-staffing in June costs one month of wages.

Run:  python src/forecast.py
"""

from __future__ import annotations

import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

STUDENTS_PER_INSTRUCTOR_HOUR = 3.0
INSTRUCTOR_COST_PER_HOUR = 24.0
SESSIONS_PER_STUDENT_MONTH = 5.2

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
# 1. Decompose, then re-run the comparison the repo already made
# ---------------------------------------------------------------------------

def part1_confounding(df):
    y = pd.Series(df.leads.values, index=pd.PeriodIndex(df.month, freq="M").to_timestamp())
    stl = STL(y, period=12, robust=True).fit()
    adj = y - stl.seasonal

    before = df.period == "before"
    raw_lift = df.leads[~before].mean() / df.leads[before].mean() - 1
    adj_lift = adj.values[~before.values].mean() / adj.values[before.values].mean() - 1

    # How much of the raw gap is calendar? The two windows differ in seasonal mix.
    seas_before = df.seasonal_index[before].mean()
    seas_after = df.seasonal_index[~before].mean()
    calendar_share = (seas_after / seas_before - 1) / raw_lift if raw_lift else np.nan

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.6), sharex=True)
    for ax, series, name, col in [
        (axes[0], y, "observed leads", MUTED),
        (axes[1], stl.trend, "trend", BLUE),
        (axes[2], stl.seasonal, "seasonal", ORANGE),
    ]:
        ax.plot(y.index, series, color=col, linewidth=2)
        ax.set_ylabel(name, fontsize=9.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    cut = y.index[int(before.sum())]
    for ax in axes:
        ax.axvline(cut, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].set_title("Pulling the school calendar out of the lead series",
                      loc="left", fontsize=13, color=INK, pad=20)
    axes[0].text(0, 1.06, "STL decomposition, 12-month period, robust fit",
                 transform=axes[0].transAxes, fontsize=9.5, color=MUTED)
    axes[0].annotate(" operating change", (cut, y.max()), fontsize=9, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(CHARTS / "seasonality.png", bbox_inches="tight")
    plt.close(fig)

    return {"raw_lift": float(raw_lift), "adjusted_lift": float(adj_lift),
            "calendar_share_of_raw": float(calendar_share),
            "seasonal_amplitude": float(stl.seasonal.max() - stl.seasonal.min()),
            "trend_lift": float(stl.trend.iloc[-1] / stl.trend.iloc[0] - 1)}


# ---------------------------------------------------------------------------
# 2. Backtest, rolling origin
# ---------------------------------------------------------------------------

def seasonal_naive(train, h):
    """Last year's same month. The baseline that embarrasses most models."""
    return np.array([train[-12 + (i % 12)] for i in range(h)])


def drift_naive(train, h):
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


def sarima(train, h):
    m = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    f = m.get_forecast(h)
    return f.predicted_mean, f.conf_int(alpha=0.2)


def fourier_ols(train, h):
    """Trend plus two Fourier harmonics. Fewer parameters than SARIMA and it
    does not need differencing, which matters when the series is short."""
    n = len(train)
    t = np.arange(n)
    def design(tt):
        cols = [np.ones_like(tt, dtype=float), tt.astype(float)]
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * tt / 12), np.cos(2 * np.pi * k * tt / 12)]
        return np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design(t), train, rcond=None)
    return design(np.arange(n, n + h)) @ beta


def part2_backtest(df, horizon=3, min_train=30):
    y = df.enrollments.values.astype(float)
    models = {"seasonal naive": seasonal_naive, "drift": drift_naive,
              "Fourier + trend": fourier_ols,
              "SARIMA": lambda tr, h: sarima(tr, h)[0]}
    errs = {k: [] for k in models}
    for origin in range(min_train, len(y) - horizon + 1):
        tr, actual = y[:origin], y[origin:origin + horizon]
        for name, fn in models.items():
            try:
                pred = np.asarray(fn(tr, horizon), dtype=float)
                errs[name].append(np.abs(pred - actual))
            except Exception:
                pass
    out = {}
    naive_mae = np.mean(np.concatenate(errs["seasonal naive"]))
    for name, e in errs.items():
        allerr = np.concatenate(e)
        out[name] = {"mae": float(np.mean(allerr)),
                     "rmse": float(np.sqrt(np.mean(allerr ** 2))),
                     "mase_vs_seasonal_naive": float(np.mean(allerr) / naive_mae),
                     "n_origins": len(e)}
    return out, errs


# ---------------------------------------------------------------------------
# 3. The forecast anyone actually uses
# ---------------------------------------------------------------------------

def part3_forecast(df, h=9):
    # Staffing follows the ACTIVE roster, not new enrollments. A student who
    # enrolled two years ago still needs an instructor in the room this Tuesday.
    y = df.active_students.values.astype(float)
    idx = pd.PeriodIndex(df.month, freq="M")
    mean, ci = sarima(y, h)
    future = pd.period_range(idx[-1] + 1, periods=h, freq="M")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(range(len(y)), y, color=MUTED, linewidth=1.8)
    xf = range(len(y), len(y) + h)
    ax.plot(xf, mean, color=BLUE, linewidth=2.2)
    lo, hi = np.asarray(ci)[:, 0], np.asarray(ci)[:, 1]
    ax.fill_between(xf, lo, hi, color=BLUE, alpha=0.16, linewidth=0)
    ax.axvline(len(y) - 0.5, color=GRID, linewidth=1.5)
    _frame(ax, "Active roster, nine months out",
           "SARIMA(1,0,1)(1,1,0)[12] with an 80% interval")
    ticks = list(range(0, len(y) + h, 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([(idx[0] + t).strftime("%Y-%m") if t < len(y)
                        else future[t - len(y)].strftime("%Y-%m") for t in ticks],
                       rotation=0, fontsize=8.5)
    ax.set_ylabel("active students")
    ax.annotate("forecast", (len(y) + 1, hi.max()), fontsize=9.5, color=BLUE, va="top")

    # Staffing: the decision the forecast is actually for
    ax = axes[1]
    hours_mid = mean * SESSIONS_PER_STUDENT_MONTH / STUDENTS_PER_INSTRUCTOR_HOUR
    hours_hi = hi * SESSIONS_PER_STUDENT_MONTH / STUDENTS_PER_INSTRUCTOR_HOUR
    x = np.arange(h)
    ax.bar(x, hours_mid, 0.62, color=BLUE)
    ax.errorbar(x, hours_mid, yerr=[np.zeros(h), hours_hi - hours_mid],
                fmt="none", ecolor=MUTED, capsize=3, linewidth=1)
    _frame(ax, "Instructor hours to schedule",
           "Point forecast, with the 80% upper bound as the hiring buffer")
    ax.set_xticks(x)
    ax.set_xticklabels([p.strftime("%b") for p in future], fontsize=8.5)
    ax.set_ylabel("instructor-hours per month")
    peak, trough = int(np.argmax(hours_mid)), int(np.argmin(hours_mid))
    ax.annotate(f"{hours_mid[peak]:.0f}h", (peak, hours_mid[peak]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK)
    ax.annotate(f"{hours_mid[trough]:.0f}h", (trough, hours_mid[trough]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK)

    fig.tight_layout()
    fig.savefig(CHARTS / "forecast.png", bbox_inches="tight")
    plt.close(fig)

    swing = float(hours_mid.max() - hours_mid.min())
    return {"months": [str(p) for p in future],
            "point": [float(v) for v in mean],
            "lo80": [float(v) for v in lo], "hi80": [float(v) for v in hi],
            "peak_hours": float(hours_mid.max()), "trough_hours": float(hours_mid.min()),
            "hour_swing": swing,
            "wage_swing_dollars": swing * INSTRUCTOR_COST_PER_HOUR}


def main():
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "center_monthly.csv")

    conf = part1_confounding(df)
    bt, _ = part2_backtest(df)
    fc = part3_forecast(df)

    (ROOT / "results_forecast.json").write_text(
        json.dumps({"confounding": conf, "backtest": bt, "forecast": fc}, indent=2))

    print("SEASONALITY AND THE BEFORE/AFTER COMPARISON")
    print(f"  raw lead lift, before vs after : {conf['raw_lift']:+.1%}")
    print(f"  after removing the calendar    : {conf['adjusted_lift']:+.1%}")
    print(f"  share of the raw gap that was calendar: {conf['calendar_share_of_raw']:.1%}")
    print(f"  peak-to-trough seasonal swing  : {conf['seasonal_amplitude']:.0f} leads/month")

    print("\nBACKTEST, rolling origin, 3-month horizon")
    print(f"  {'model':<18s}{'MAE':>8s}{'RMSE':>8s}{'vs naive':>10s}{'origins':>9s}")
    for k, v in sorted(bt.items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {k:<18s}{v['mae']:>8.2f}{v['rmse']:>8.2f}"
              f"{v['mase_vs_seasonal_naive']:>10.2f}{v['n_origins']:>9d}")

    print("\nSTAFFING")
    print(f"  peak month  : {fc['peak_hours']:.0f} instructor-hours")
    print(f"  trough month: {fc['trough_hours']:.0f} instructor-hours")
    print(f"  swing       : {fc['hour_swing']:.0f} hours, "
          f"${fc['wage_swing_dollars']:,.0f} of monthly wages")


if __name__ == "__main__":
    main()
