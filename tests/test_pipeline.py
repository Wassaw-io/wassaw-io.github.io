"""Tests that would actually catch a wrong answer.

Not smoke tests. Each one checks a claim the write-ups make, against either a
closed form or a case small enough to compute by hand.

    pytest -q
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from survival import km_curve, rmst, gamma_frailty_mle, exponential_mle  # noqa: E402
from risk_model import expand, survival_from_hazard, FEATURES, fit_and_score  # noqa: E402
import generate_students  # noqa: E402


# --------------------------------------------------------------------------
# Kaplan-Meier, against a case you can do on paper
# --------------------------------------------------------------------------

def test_km_matches_hand_computation():
    # 5 subjects. Events at 2 and 5; censored at 3, 4, 6.
    t = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    d = np.array([1, 0, 0, 1, 0])
    times, surv = km_curve(t, d)
    # At t=2: 5 at risk, 1 event -> 4/5 = 0.8
    # At t=5: 2 at risk, 1 event -> 0.8 * 1/2 = 0.4
    assert surv[0] == pytest.approx(1.0)
    assert np.interp(2.0, times, surv) == pytest.approx(0.8)
    assert np.interp(5.0, times, surv) == pytest.approx(0.4)


def test_km_ignores_censoring_as_an_event():
    """Censored observations must not pull the curve down."""
    t = np.array([1.0, 2.0, 3.0])
    _, all_censored = km_curve(t, np.array([0, 0, 0]))
    assert np.all(all_censored == 1.0)


# --------------------------------------------------------------------------
# RMST, against the closed form for an exponential
# --------------------------------------------------------------------------

def test_rmst_matches_exponential_closed_form():
    lam, H = 0.08, 30.0
    t = np.linspace(0, H, 4001)
    S = np.exp(-lam * t)
    # integral_0^H exp(-lam t) dt = (1 - exp(-lam H)) / lam
    expected = (1 - np.exp(-lam * H)) / lam
    assert rmst(t, S, H) == pytest.approx(expected, rel=1e-3)


def test_rmst_is_bounded_by_the_horizon():
    t = np.linspace(0, 30, 500)
    assert rmst(t, np.ones_like(t), 30.0) == pytest.approx(30.0, rel=1e-6)


# --------------------------------------------------------------------------
# The frailty model has to recover what generated the data
# --------------------------------------------------------------------------

def test_frailty_mle_recovers_theta_on_a_large_sample():
    rng = np.random.default_rng(0)
    n, lam_true, theta_true = 20000, 0.07, 0.6
    z = rng.gamma(1 / theta_true, theta_true, n)
    x = rng.normal(0, 1, (n, 1))
    beta_true = 0.5
    lam_i = lam_true * np.exp(beta_true * x[:, 0])
    t = rng.exponential(1 / (z * lam_i))
    cens = rng.uniform(0, 60, n)
    obs, d = np.minimum(t, cens), (t <= cens).astype(int)

    fit = gamma_frailty_mle(obs, d, x)
    assert fit["converged"]
    assert fit["theta"] == pytest.approx(theta_true, rel=0.20)
    assert fit["lambda"] == pytest.approx(lam_true, rel=0.20)
    assert fit["beta"][0] == pytest.approx(beta_true, rel=0.20)


def test_frailty_beats_exponential_when_frailty_is_present():
    """The likelihood ratio must favour the richer model on frailty data."""
    rng = np.random.default_rng(1)
    n = 6000
    z = rng.gamma(1 / 0.7, 0.7, n)
    x = rng.normal(0, 1, (n, 1))
    t = rng.exponential(1 / (z * 0.07 * np.exp(0.4 * x[:, 0])))
    cens = rng.uniform(0, 60, n)
    obs, d = np.minimum(t, cens), (t <= cens).astype(int)
    fr, ex = gamma_frailty_mle(obs, d, x), exponential_mle(obs, d, x)
    assert fr["loglik"] > ex["loglik"]
    assert 2 * (fr["loglik"] - ex["loglik"]) > 10.83  # p < 0.001 on chi2(1)


def test_frailty_collapses_to_exponential_when_there_is_none():
    """theta should go near zero on genuinely exponential data. Guards against
    a model that always reports heterogeneity whether or not it is there."""
    rng = np.random.default_rng(2)
    n = 12000
    x = rng.normal(0, 1, (n, 1))
    t = rng.exponential(1 / (0.07 * np.exp(0.3 * x[:, 0])))
    cens = rng.uniform(0, 60, n)
    obs, d = np.minimum(t, cens), (t <= cens).astype(int)
    assert gamma_frailty_mle(obs, d, x)["theta"] < 0.15


# --------------------------------------------------------------------------
# Person-period expansion
# --------------------------------------------------------------------------

def _toy():
    return pd.DataFrame({
        "student_id": ["A", "B"],
        "channel": ["organic", "paid"],
        "grade_band": ["middle", "high"],
        "sessions_per_week": [1, 2],
        "assessment_gap": [1.0, 2.0],
        "first8_attendance": [0.9, 0.6],
        "tenure_months": [3.0, 2.4],
        "churned": [1, 0],
    })


def test_expansion_row_and_event_counts():
    pp = expand(_toy())
    # A: 3 months, churned -> 3 rows, one event on the last
    # B: ceil(2.4) = 3 months, censored -> 3 rows, no events
    assert len(pp) == 6
    assert pp.y.sum() == 1
    a = pp[pp.student_id == "A"]
    assert list(a.y) == [0, 0, 1]
    assert list(pp[pp.student_id == "B"].y) == [0, 0, 0]


def test_censored_student_contributes_no_event():
    pp = expand(_toy())
    assert pp[pp.student_id == "B"].y.sum() == 0


def test_month_index_starts_at_zero_and_is_contiguous():
    pp = expand(_toy())
    for sid in ["A", "B"]:
        m = pp[pp.student_id == sid].month.values
        assert m[0] == 0
        assert np.all(np.diff(m) == 1)


# --------------------------------------------------------------------------
# The bridge back to a survival curve
# --------------------------------------------------------------------------

def test_survival_from_hazard_is_a_valid_survival_function():
    h = np.array([0.1, 0.2, 0.05, 0.3])
    S = survival_from_hazard(h)
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S) <= 0)
    assert S[0] == pytest.approx(0.9)
    assert S[1] == pytest.approx(0.9 * 0.8)


def test_zero_hazard_means_everyone_survives():
    assert np.allclose(survival_from_hazard(np.zeros(10)), 1.0)


# --------------------------------------------------------------------------
# The leakage guard. This is the test that matters most.
# --------------------------------------------------------------------------

def test_no_student_appears_in_both_train_and_test():
    df = generate_students.build().head(400)
    _, _, te, _, _, m = fit_and_score(df, seed=5)
    # Reconstruct the split the same way and assert disjointness
    rng = np.random.default_rng(5)
    ids = np.array(df.student_id.unique(), dtype=object)
    rng.shuffle(ids)
    cut = int(0.75 * len(ids))
    train_ids, test_ids = set(ids[:cut]), set(ids[cut:])
    assert train_ids.isdisjoint(test_ids)
    assert set(te.student_id).issubset(test_ids)
    assert m["train_students"] + m["test_students"] == len(ids)


def test_features_are_ordered_consistently():
    """The call list builds a frame by hand; column order must match training."""
    pp = expand(_toy())
    assert list(pp.columns) == ["student_id"] + FEATURES + ["y"]


# --------------------------------------------------------------------------
# The generator has to produce what it documents
# --------------------------------------------------------------------------

def test_generator_produces_censoring_and_events():
    df = generate_students.build()
    assert 0.2 < df.churned.mean() < 0.9, "need both events and censored spells"
    assert df.tenure_months.min() >= 0.5
    assert set(df.channel.unique()) == {"organic", "paid", "referral", "walk_in"}


def test_paid_channel_churns_faster_as_designed():
    """The generator claims paid students leave sooner. Hold it to that."""
    df = generate_students.build()
    paid = df[df.channel == "paid"]
    ref = df[df.channel == "referral"]
    _, s_paid = km_curve(paid.tenure_months.values, paid.churned.values)
    _, s_ref = km_curve(ref.tenure_months.values, ref.churned.values)
    assert rmst(*km_curve(paid.tenure_months.values, paid.churned.values), 30) < \
           rmst(*km_curve(ref.tenure_months.values, ref.churned.values), 30)


# --------------------------------------------------------------------------
# Uplift
# --------------------------------------------------------------------------

def test_qini_of_a_random_score_is_near_zero():
    """A score carrying no information must not look like it targets."""
    import uplift as U
    rng = np.random.default_rng(4)
    n = 4000
    treated = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.4, n)
    x, q = U.qini_curve(rng.normal(size=n), treated, y)
    assert abs(U.qini_coefficient(x, q)) < 12


def test_qini_rewards_a_score_that_finds_the_persuadables():
    """A score equal to the true effect must beat a random one."""
    import uplift as U
    rng = np.random.default_rng(6)
    n = 4000
    tau = rng.uniform(0, 0.5, n)
    treated = rng.integers(0, 2, n)
    y = rng.binomial(1, np.where(treated == 1, 0.3 + tau, 0.3))
    x_good, q_good = U.qini_curve(tau, treated, y)
    x_rand, q_rand = U.qini_curve(rng.normal(size=n), treated, y)
    assert U.qini_coefficient(x_good, q_good) > U.qini_coefficient(x_rand, q_rand)


def test_experiment_arms_are_balanced_and_effect_is_positive():
    import uplift as U
    df = U.simulate_experiment(seed=9, n=3000)
    assert 0.45 < df.treated.mean() < 0.55
    ate = df[df.treated == 1].retained.mean() - df[df.treated == 0].retained.mean()
    assert ate > 0.03


def test_true_effect_is_non_monotone_in_attendance():
    """The whole argument depends on the effect peaking in the middle."""
    import uplift as U
    att = np.array([0.40, 0.72, 0.99])
    tau = U.true_tau_multiplier(att, np.array([1.0, 1.0, 1.0]))
    assert tau[1] < tau[0] and tau[1] < tau[2]   # most negative = biggest benefit


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

def test_seasonal_naive_repeats_last_year():
    import forecast as F
    train = np.arange(24, dtype=float)
    out = F.seasonal_naive(train, 3)
    assert list(out) == [12.0, 13.0, 14.0]


def test_fourier_recovers_a_clean_seasonal_signal():
    import forecast as F
    t = np.arange(48)
    y = 100 + 2 * t + 20 * np.sin(2 * np.pi * t / 12)
    pred = F.fourier_ols(y, 6)
    truth = 100 + 2 * np.arange(48, 54) + 20 * np.sin(2 * np.pi * np.arange(48, 54) / 12)
    assert np.allclose(pred, truth, atol=1.0)


def test_backtest_never_uses_future_data():
    """Every origin must forecast strictly forward. A model that peeks would
    score near-perfectly, so a suspiciously low error is the tell."""
    import forecast as F
    df = pd.read_csv(ROOT / "data" / "center_monthly.csv")
    out, _ = F.part2_backtest(df, horizon=3, min_train=30)
    assert out["seasonal naive"]["mae"] > 0.5
    assert all(v["n_origins"] == out["seasonal naive"]["n_origins"] for v in out.values())


# --------------------------------------------------------------------------
# Drift monitoring
# --------------------------------------------------------------------------

def test_psi_is_zero_for_identical_distributions():
    from monitoring import psi
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_grows_with_the_size_of_the_shift():
    from monitoring import psi
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    small = psi(ref, rng.normal(0.2, 1, 20000))
    large = psi(ref, rng.normal(1.5, 1, 20000))
    assert 0 < small < large
    assert large > 0.25


def test_psi_uses_frozen_reference_bins():
    """If the edges were recomputed on the current sample, a pure location
    shift would score near zero. It must not."""
    from monitoring import psi
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 20000)
    shifted = rng.normal(3.0, 1, 20000)
    assert psi(ref, shifted) > 1.0


# --------------------------------------------------------------------------
# The CLI is a contract
# --------------------------------------------------------------------------

def test_score_cli_round_trip(tmp_path):
    import score as S
    data = ROOT / "data" / "students.csv"
    model = tmp_path / "m.pkl"
    out = tmp_path / "calls.csv"
    assert S.main(["train", "--data", str(data), "--out", str(model)]) == 0
    assert model.exists()
    assert S.main(["score", "--model", str(model), "--data", str(data),
                   "--out", str(out), "--top", "10"]) == 0
    got = pd.read_csv(out)
    assert len(got) == 10
    assert got.value_at_risk.is_monotonic_decreasing
    assert out.with_suffix(".meta.json").exists()


def test_score_cli_rejects_unknown_channel(tmp_path):
    import score as S
    df = pd.read_csv(ROOT / "data" / "students.csv").head(200)
    df.loc[df.index[0], "channel"] = "tiktok"
    bad = tmp_path / "bad.csv"; df.to_csv(bad, index=False)
    model = tmp_path / "m.pkl"
    S.main(["train", "--data", str(ROOT / "data" / "students.csv"), "--out", str(model)])
    with pytest.raises(SystemExit):
        S.main(["score", "--model", str(model), "--data", str(bad),
                "--out", str(tmp_path / "x.csv")])


def test_score_cli_rejects_out_of_range_values(tmp_path):
    import score as S
    df = pd.read_csv(ROOT / "data" / "students.csv").head(200)
    df.loc[df.index[0], "first8_attendance"] = 4.2
    bad = tmp_path / "bad.csv"; df.to_csv(bad, index=False)
    model = tmp_path / "m.pkl"
    S.main(["train", "--data", str(ROOT / "data" / "students.csv"), "--out", str(model)])
    with pytest.raises(SystemExit):
        S.main(["score", "--model", str(model), "--data", str(bad),
                "--out", str(tmp_path / "x.csv")])


# --------------------------------------------------------------------------
# Partial pooling
# --------------------------------------------------------------------------

def test_beta_binomial_recovers_known_hyperparameters():
    """With enough groups the marginal likelihood must find the truth."""
    from hierarchical import fit_beta_binomial
    rng = np.random.default_rng(3)
    a_t, b_t, K = 7.0, 43.0, 400
    n = rng.integers(60, 400, K)
    p = rng.beta(a_t, b_t, K)
    y = rng.binomial(n, p)
    a, b, mu, kappa = fit_beta_binomial(y, n)
    assert mu == pytest.approx(a_t / (a_t + b_t), rel=0.15)
    assert kappa == pytest.approx(a_t + b_t, rel=0.5)


def test_pooling_shrinks_small_groups_more_than_large_ones():
    """The whole point: the weight is n, not a knob."""
    from hierarchical import fit_beta_binomial
    rng = np.random.default_rng(4)
    n = np.array([20, 20, 800, 800] * 25)
    p = rng.beta(7.0, 43.0, len(n))
    y = rng.binomial(n, p)
    a, b, mu, _ = fit_beta_binomial(y, n)
    naive = y / n
    post = (a + y) / (a + b + n)
    pull = np.abs(post - naive) / np.maximum(np.abs(naive - mu), 1e-9)
    assert pull[n == 20].mean() > pull[n == 800].mean()


def test_pooling_reduces_risk_on_average():
    """Averaged over draws, shrinkage must beat the raw rate on squared error."""
    from hierarchical import fit_beta_binomial, simulate
    gains = []
    for r in range(15):
        d = simulate(seed=900 + r * 7, n_centers=60)
        a, b, _, _ = fit_beta_binomial(d["y"], d["n"])
        post = (a + d["y"]) / (a + b + d["n"])
        naive = d["y"] / d["n"]
        gains.append(np.mean((post - d["p_true"]) ** 2) <
                     np.mean((naive - d["p_true"]) ** 2))
    assert sum(gains) >= 12, "shrinkage should win the large majority of draws"


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def test_bm25_ranks_the_obvious_document_first():
    from retrieval import BM25, DOCS
    names, texts = list(DOCS), list(DOCS.values())
    bm = BM25(texts)
    s = bm.score("refund pro rata written notice")
    assert names[int(np.argmax(s))] == "refunds"


def test_bm25_length_normalisation_is_active():
    """Doubling a document's length must not double its score."""
    from retrieval import BM25
    short = "referral credit for a referred family"
    long = short + " " + " ".join(["unrelated filler words here"] * 40)
    a = BM25([short, "something else entirely about capacity"]).score("referral credit")
    b = BM25([long, "something else entirely about capacity"]).score("referral credit")
    assert b[0] < a[0]


def test_every_labelled_gold_document_exists():
    from retrieval import DOCS, QUERIES
    for q, gold, _ in QUERIES:
        for g in gold:
            assert g in DOCS, f"query '{q}' points at a missing document {g}"


def test_chunking_preserves_all_content_words():
    from retrieval import chunk_corpus, DOCS, tok
    chunks, owner = chunk_corpus(DOCS, 20)
    for name, text in DOCS.items():
        mine = {w for c, o in zip(chunks, owner) if o == name for w in tok(c)}
        assert set(tok(text)) <= mine, f"{name} lost words to chunking"


def test_rrf_is_insensitive_to_score_scale():
    """The point of rank fusion: only order matters."""
    from retrieval import rrf
    a = np.array([10.0, 5.0, 1.0])
    b = np.array([0.001, 0.0005, 0.0001])
    assert np.allclose(rrf([a, b]), rrf([a * 1000, b]))


def test_bootstrap_interval_brackets_the_point_estimate():
    from retrieval import bootstrap_ci
    pq = [{"recall": v} for v in [1, 1, 1, 0, 1, 0.5, 1, 1]]
    lo, hi = bootstrap_ci(pq)
    mean = np.mean([p["recall"] for p in pq])
    assert lo <= mean <= hi
