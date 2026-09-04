"""
Generate a synthetic 30-month operating history for a math tutoring center.

The data is SYNTHETIC. The structure is not: these are the tables a franchise
center actually keeps, at the grain it actually keeps them, and the analysis in
analyze.py is the analysis I ran against the real ones.

Everything you would want to change lives in CONFIG. Swap these for real figures
and every number in the README recomputes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG — the only block you need to edit
# --------------------------------------------------------------------------

CONFIG = {
    "seed": 11,
    "start_month": "2021-01",
    "n_months": 54,
    # Month index at which the analytics and organic-search work starts landing.
    # Before this the center buys leads; after it, it earns them.
    "intervention_month": 24,
    # Enrollment
    "enroll_start": 92,          # active students in month 0
    "enroll_end": 131,           # active students in the final month
    # Monthly tuition per active student, dollars
    "tuition": 315,
    # Paid acquisition spend per month, dollars
    "paid_spend_before": 4200,
    "paid_spend_after": 2100,
    # Lead volume by channel, per month
    "paid_leads_before": 34,
    "paid_leads_after": 19,
    "organic_leads_before": 9,
    "organic_leads_after": 41,
    # Funnel conversion rates
    "lead_to_trial_before": 0.44,
    "lead_to_trial_after": 0.52,
    "trial_to_enroll_before": 0.38,
    "trial_to_enroll_after": 0.49,
    # Monthly churn (share of active students who leave)
    "churn_before": 0.058,
    "churn_after": 0.037,
    # School-calendar seasonality, multiplicative, indexed by calendar month.
    # August and September are enrollment season; December and June are dead.
    # A tutoring centre lives and dies by this and every before-and-after
    # comparison in this repo is confounded by it until it is removed.
    "seasonal_leads": {
        1: 1.18, 2: 1.02, 3: 0.96, 4: 0.92, 5: 0.78, 6: 0.55,
        7: 0.72, 8: 1.62, 9: 1.55, 10: 1.05, 11: 0.88, 12: 0.62,
    },
    # Noise
    "noise": 0.09,
}


def _ramp(rng, n, before, after, cut, noise):
    """A step from `before` to `after` at month `cut`, smoothed over 4 months."""
    base = np.where(np.arange(n) < cut, before, after).astype(float)
    # smooth the step so it reads like an operating change, not a switch flip
    kernel = np.ones(4) / 4
    base = np.convolve(np.pad(base, (3, 0), mode="edge"), kernel, mode="valid")
    return base * (1 + rng.normal(0, noise, n))


def build() -> pd.DataFrame:
    c = CONFIG
    rng = np.random.default_rng(c["seed"])
    n = c["n_months"]
    cut = c["intervention_month"]

    months = pd.period_range(c["start_month"], periods=n, freq="M")

    paid_leads = _ramp(rng, n, c["paid_leads_before"], c["paid_leads_after"], cut, c["noise"])
    org_leads = _ramp(rng, n, c["organic_leads_before"], c["organic_leads_after"], cut, c["noise"])
    spend = _ramp(rng, n, c["paid_spend_before"], c["paid_spend_after"], cut, c["noise"] / 2)

    l2t = _ramp(rng, n, c["lead_to_trial_before"], c["lead_to_trial_after"], cut, c["noise"] / 3)
    t2e = _ramp(rng, n, c["trial_to_enroll_before"], c["trial_to_enroll_after"], cut, c["noise"] / 3)
    churn = _ramp(rng, n, c["churn_before"], c["churn_after"], cut, c["noise"] / 3)

    seas = np.array([c["seasonal_leads"][m.month] for m in months])
    paid_leads *= seas
    org_leads *= seas

    leads = paid_leads + org_leads
    trials = leads * l2t
    enrollments = trials * t2e

    # Active roster: start at enroll_start, add enrollments, lose churn.
    # Nudge toward enroll_end so the synthetic history lands where CONFIG says.
    active = np.empty(n)
    active[0] = c["enroll_start"]
    for i in range(1, n):
        active[i] = active[i - 1] * (1 - churn[i]) + enrollments[i]
    active *= np.linspace(1.0, c["enroll_end"] / active[-1], n)

    df = pd.DataFrame(
        {
            "month": months.astype(str),
            "calendar_month": months.month,
            "seasonal_index": seas.round(3),
            "paid_leads": paid_leads.round(0),
            "organic_leads": org_leads.round(0),
            "leads": leads.round(0),
            "trials": trials.round(0),
            "enrollments": enrollments.round(0),
            "paid_spend": spend.round(2),
            "active_students": active.round(0),
            "monthly_churn": churn.round(4),
            "period": np.where(np.arange(n) < cut, "before", "after"),
        }
    )

    df["revenue"] = (df["active_students"] * c["tuition"]).round(2)
    df["cost_per_lead"] = (df["paid_spend"] / df["paid_leads"]).round(2)
    df["cost_per_enrolled"] = (df["paid_spend"] / df["enrollments"]).round(2)
    df["organic_share"] = (df["organic_leads"] / df["leads"]).round(4)
    return df


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "center_monthly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(out, index=False)
    print(f"wrote {out}")
