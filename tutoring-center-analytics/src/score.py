"""
Train once, score many. The command-line interface an owner's laptop can run.

The difference between an analysis and a system is that a system can be run by
someone who did not write it, on data it has not seen, without a notebook. This
is that boundary.

    python src/score.py train --data data/students.csv --out models/retention.pkl
    python src/score.py score --model models/retention.pkl --data data/students.csv \
        --out data/call_list.csv --top 40
    python src/score.py check --model models/retention.pkl --data data/students.csv

`train` fits, evaluates on held-out students, and writes a single artifact that
carries the model, the feature order, the training distribution needed for drift
checks, and a fingerprint of the data it was fitted on.

`score` refuses to run if the incoming columns do not match what the model was
trained on, because a silent column reorder is the failure that produces
confident, wrong, unnoticed numbers for six months.

`check` runs the drift monitor from monitoring.py against the stored training
distribution and exits non-zero when it says retrain, so it can sit in a cron
job or a CI step and actually block something.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from monitoring import psi, PSI_ALERT, CALIB_ALERT           # noqa: E402
from risk_model import (FEATURES, expand, fit_and_score,     # noqa: E402
                        call_list, survival_from_hazard)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2


def _fingerprint(df: pd.DataFrame) -> str:
    """Content hash of the training data, so a scored file can always be traced
    back to the exact rows the model saw."""
    h = hashlib.sha256()
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def cmd_train(args):
    df = pd.read_csv(args.data)
    gbm, logit, te, p_gbm, p_log, metrics = fit_and_score(df, seed=args.seed)
    winner = logit if metrics["logit_auc"] >= metrics["gbm_auc"] else gbm
    name = "logistic" if winner is logit else "gradient_boosting"

    pp = expand(df)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model": winner,
        "model_name": name,
        "features": list(FEATURES),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_students": int(df.student_id.nunique()),
        "data_fingerprint": _fingerprint(df),
        "metrics": {k: v for k, v in metrics.items() if k != "importance"},
        # Reference distribution, kept so `check` can compute drift later without
        # needing the original training file to still exist.
        "reference": {f: pp[f].to_numpy() for f in FEATURES},
        "reference_event_rate": float(pp.y.mean()),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(artifact, fh)

    print(f"trained  : {name}")
    print(f"held out : AUC {metrics['logit_auc' if name == 'logistic' else 'gbm_auc']:.3f} "
          f"on {metrics['test_students']} students")
    print(f"fingerprint: {artifact['data_fingerprint']}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


def _load(path):
    with open(path, "rb") as fh:
        a = pickle.load(fh)
    if a.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"artifact schema v{a.get('schema_version')} "
                         f"but this code expects v{SCHEMA_VERSION}. Retrain.")
    return a


def _validate(df, artifact):
    """Fail loudly on a schema mismatch instead of scoring nonsense."""
    needed = {"student_id", "channel", "grade_band", "sessions_per_week",
              "assessment_gap", "first8_attendance", "tenure_months", "churned"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"input is missing required columns: {sorted(missing)}")
    for col, lo, hi in [("first8_attendance", 0.0, 1.0),
                        ("assessment_gap", 0.0, 12.0),
                        ("tenure_months", 0.0, 600.0)]:
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad):
            raise SystemExit(f"{len(bad)} rows have {col} outside [{lo}, {hi}]")
    unknown = set(df.channel.unique()) - {"organic", "paid", "referral", "walk_in"}
    if unknown:
        raise SystemExit(f"unknown channel values: {sorted(unknown)}. "
                         f"The model has no coefficient for these.")


def cmd_score(args):
    a = _load(args.model)
    df = pd.read_csv(args.data)
    _validate(df, a)
    top, full = call_list(df, a["model"], top_n=args.top)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out, index=False)

    meta = {"scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": a["model_name"], "trained_at": a["trained_at"],
            "training_fingerprint": a["data_fingerprint"],
            "scoring_fingerprint": _fingerprint(df),
            "active_students": int(len(full)),
            "total_value_at_risk": float(full.value_at_risk.sum())}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"scored {len(full)} active students with {a['model_name']} "
          f"(trained {a['trained_at']})")
    print(f"${meta['total_value_at_risk']:,.0f} of contribution margin at risk")
    print(f"wrote {out} (top {len(top)}) and {out.with_suffix('.meta.json').name}")
    return 0


def cmd_check(args):
    a = _load(args.model)
    df = pd.read_csv(args.data)
    _validate(df, a)
    pp = expand(df)

    drift = {f: psi(a["reference"][f], pp[f].to_numpy()) for f in FEATURES}
    worst_f = max(drift, key=drift.get)
    p = a["model"].predict_proba(pp[FEATURES])[:, 1]
    predicted, observed = float(p.mean()), float(pp.y.mean())
    calib = abs(predicted - observed) / max(observed, 1e-9)

    signals = {"psi": drift[worst_f] >= PSI_ALERT, "calibration": calib >= CALIB_ALERT}
    verdict = "RETRAIN" if sum(signals.values()) >= 2 else \
              ("WATCH" if any(signals.values()) else "OK")

    print(f"model      : {a['model_name']}, trained {a['trained_at']}")
    print(f"worst drift: {worst_f} PSI {drift[worst_f]:.3f} (alert at {PSI_ALERT})")
    print(f"calibration: predicted {predicted:.4f} vs observed {observed:.4f} "
          f"({calib:.1%} off, alert at {CALIB_ALERT:.0%})")
    print("\nPSI by feature:")
    for f, v in sorted(drift.items(), key=lambda kv: -kv[1]):
        flag = "  <-- alert" if v >= PSI_ALERT else ("  <- watch" if v >= 0.10 else "")
        print(f"  {f:<20s}{v:>7.3f}{flag}")
    print(f"\nverdict: {verdict}")
    if verdict == "RETRAIN":
        print("Two independent signals fired. Retrain before the next scoring run.")
        return 2
    if verdict == "WATCH":
        print("One signal fired. Feature drift without calibration loss is usually "
              "covariate shift, not concept drift, and usually does not need a retrain.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="score", description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit a model and write an artifact")
    t.add_argument("--data", default=str(ROOT / "data" / "students.csv"))
    t.add_argument("--out", default=str(ROOT / "models" / "retention.pkl"))
    t.add_argument("--seed", type=int, default=3)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("score", help="write a ranked call list")
    s.add_argument("--model", default=str(ROOT / "models" / "retention.pkl"))
    s.add_argument("--data", default=str(ROOT / "data" / "students.csv"))
    s.add_argument("--out", default=str(ROOT / "data" / "call_list.csv"))
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("check", help="drift check; exits 2 if a retrain is needed")
    c.add_argument("--model", default=str(ROOT / "models" / "retention.pkl"))
    c.add_argument("--data", default=str(ROOT / "data" / "students.csv"))
    c.set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
