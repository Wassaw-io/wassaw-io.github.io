"""
Student-level enrollment spells for the tutoring center, with right censoring.

SYNTHETIC. But generated from a specific and deliberately awkward truth, so that
the analysis in survival.py has something real to be wrong about.

The generating model
--------------------
Every student has a CONSTANT individual hazard of leaving. Students differ in that
hazard by a multiplicative frailty z drawn from a Gamma distribution with mean 1:

    h_i(t) = z_i * lambda * exp(beta' x_i),      z_i ~ Gamma(1/theta, theta)

Nothing about an individual student "settles in." No student's risk falls over
time. And yet the *population* hazard falls steeply, because the high-z students
leave first and the survivors are progressively selected toward low z. Integrating
the frailty out gives the unconditional survival function as the Laplace transform
of the Gamma density:

    S(t) = E_z[ exp(-z * Lambda(t)) ] = (1 + theta * Lambda(t)) ** (-1/theta)

with Lambda(t) = lambda * exp(beta' x) * t. The population hazard is then

    hbar(t) = lambda_x / (1 + theta * lambda_x * t)

which is strictly decreasing for any theta > 0.

This matters because it is the single most common misreading in retention work.
A falling churn curve is taken as evidence that students get more committed over
time, which implies an onboarding intervention. Here the curve falls for a reason
that no onboarding intervention would touch. The data cannot tell the two apart
without covariates or an experiment, and survival.py makes that argument in
numbers.

Mean tenure has a closed form under this model, for theta < 1:

    E[T] = 1 / (lambda_x * (1 - theta))

which is the benchmark the estimators in survival.py are graded against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFIG = {
    "seed": 7,
    "n_students": 1400,
    # Baseline monthly hazard for the reference student (lambda)
    "baseline_hazard": 0.075,
    # Frailty variance. 0 = every student identical = exponential = flat hazard.
    # 0.55 = substantial unobserved heterogeneity.
    "frailty_theta": 0.55,
    # Observation window, months. Students enrolling late are right censored.
    "window_months": 30,
    "tuition_per_month": 315,
    # Log-hazard coefficients (beta). Negative = stays longer.
    "beta": {
        "channel_paid": 0.44,        # paid-search students leave faster
        "channel_referral": -0.38,   # referred students stay
        "sessions_2_per_week": -0.29,
        "assessment_gap": 0.11,      # per grade level behind at intake
        "first8_attendance": -1.35,  # per unit of attendance share, 0 to 1
        "grade_high": 0.26,          # high schoolers age out and quit
    },
    # Acquisition cost per enrolled student, by channel
    "cac": {"organic": 95, "paid": 495, "referral": 60, "walk_in": 45},
}

CHANNELS = ["organic", "paid", "referral", "walk_in"]
CHANNEL_P = [0.38, 0.31, 0.19, 0.12]
GRADES = ["elementary", "middle", "high", "college"]
GRADE_P = [0.31, 0.34, 0.28, 0.07]


def build() -> pd.DataFrame:
    c = CONFIG
    rng = np.random.default_rng(c["seed"])
    n = c["n_students"]
    b = c["beta"]

    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    grade = rng.choice(GRADES, n, p=GRADE_P)
    sessions = rng.choice([1, 2], n, p=[0.62, 0.38])
    gap = np.clip(rng.normal(1.7, 0.9, n), 0, 4).round(1)

    # Attendance in the first eight weeks. Correlated with gap and sessions,
    # because the students furthest behind are the ones who start skipping.
    att = np.clip(0.93 - 0.045 * gap + 0.02 * (sessions - 1) + rng.normal(0, 0.11, n), 0.25, 1.0)

    log_hr = (
        b["channel_paid"] * (channel == "paid")
        + b["channel_referral"] * (channel == "referral")
        + b["sessions_2_per_week"] * (sessions == 2)
        + b["assessment_gap"] * gap
        + b["first8_attendance"] * (att - att.mean())
        + b["grade_high"] * (grade == "high")
    )
    lam = c["baseline_hazard"] * np.exp(log_hr)

    # Gamma frailty, mean 1, variance theta
    theta = c["frailty_theta"]
    z = rng.gamma(shape=1 / theta, scale=theta, size=n)

    # Exponential draw conditional on frailty
    true_tenure = rng.exponential(1 / (z * lam))
    individual_hazard = z * lam   # what a perfectly informed model would know

    # Staggered entry across the window, then administrative censoring at the end
    enroll_month = rng.integers(0, c["window_months"], n)
    time_available = c["window_months"] - enroll_month
    observed = np.minimum(true_tenure, time_available)
    event = (true_tenure <= time_available).astype(int)

    # A spell of zero months is not observable in a real system; floor at 0.5
    observed = np.maximum(observed, 0.5)

    df = pd.DataFrame({
        "student_id": [f"S{i:05d}" for i in range(n)],
        "enroll_month": enroll_month,
        "channel": channel,
        "grade_band": grade,
        "sessions_per_week": sessions,
        "assessment_gap": gap,
        "first8_attendance": att.round(3),
        "tenure_months": observed.round(2),
        "churned": event,
    })
    df["cac"] = df["channel"].map(c["cac"])
    df["revenue_to_date"] = (df["tenure_months"] * c["tuition_per_month"]).round(2)

    # Ground truth, kept so survival.py can grade its own estimates.
    df.attrs["truth"] = {
        "lambda": c["baseline_hazard"],
        "theta": theta,
        "individual_lambda": lam,
        "individual_hazard": individual_hazard,
        "mean_tenure_closed_form": float(np.mean(1 / (lam * (1 - theta)))),
        "beta": b,
    }
    return df


if __name__ == "__main__":
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    df = build()
    (root / "data").mkdir(exist_ok=True)
    df.to_csv(root / "data" / "students.csv", index=False)
    # The true per-student hazard. Not an input to any model: it exists so that
    # risk_model.py can compute how much of the remaining error is unlearnable.
    pd.DataFrame({"student_id": df.student_id,
                  "true_lambda": df.attrs["truth"]["individual_lambda"],
                  "true_hazard": df.attrs["truth"]["individual_hazard"]}
                 ).to_csv(root / "data" / "oracle_hazards.csv", index=False)
    truth = {k: v for k, v in df.attrs["truth"].items()
             if k not in ("individual_lambda", "individual_hazard")}
    truth["individual_lambda_mean"] = float(df.attrs["truth"]["individual_lambda"].mean())
    (root / "data" / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"{len(df)} students, {df.churned.mean():.1%} observed to churn, "
          f"{1 - df.churned.mean():.1%} right censored")
    print(f"true mean tenure (closed form): {truth['mean_tenure_closed_form']:.1f} months")
