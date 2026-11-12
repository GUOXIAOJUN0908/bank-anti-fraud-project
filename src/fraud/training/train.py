"""Train, evaluate and publish the fraud model (only if it passes the quality gate).

Steps
  1. Time-based split: train (oldest 70%) / validation (15%) / test (newest 15%).
     A random split would leak future behaviour into training.
  2. Fit a Logistic Regression baseline and a gradient-boosted tree candidate.
  3. Choose decision thresholds on validation (analyst capacity + decline precision).
  4. Report on the untouched test period.
  5. Quality gate: candidate must beat the baseline and the PR-AUC floor.
  6. Publish model.joblib + metadata.json (thresholds, drift reference, reason-code reference).

Usage:
  python -m fraud.training.train                   # HistGradientBoosting (default)
  python -m fraud.training.train --model xgboost   # needs `brew install libomp` on macOS
  python -m fraud.training.train --no-gate         # publish even if the gate fails (dev only)
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from fraud import config
from fraud.features.build import load_features
from fraud.features.schema import CATEGORICAL_FEATURES, DRIFT_EXCLUDE, FEATURES, LABEL, NUMERIC_FEATURES
from fraud.monitoring.psi import build_reference, numeric_profile
from fraud.training.evaluate import choose_thresholds, evaluate


def time_split(df: pd.DataFrame, valid_frac: float = 0.15, test_frac: float = 0.15):
    t = df.unix_time
    t_valid, t_test = t.quantile(1 - valid_frac - test_frac), t.quantile(1 - test_frac)
    return df[t < t_valid], df[(t >= t_valid) & (t < t_test)], df[t >= t_test]


def make_baseline() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def make_candidate(kind: str) -> Pipeline:
    # Categories -> integer codes; unseen categories become NaN (handled natively by the trees).
    prep = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )
    if kind == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", eval_metric="aucpr", random_state=42,
        )
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
            categorical_features=[0], early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42,
        )
    return Pipeline([("prep", prep), ("clf", clf)])


def _period(df: pd.DataFrame) -> dict:
    return {
        "from": str(df.ts.min()), "to": str(df.ts.max()),
        "n": int(len(df)), "fraud_rate": round(float(df[LABEL].mean()), 5),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def quiet_fp(fn):
    """numpy 2.x + Apple Accelerate emits spurious floating-point warnings inside matmul
    (numpy issue #26669). Results are unaffected, so silence them for training only."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return fn(*args, **kwargs)

    return wrapper


@quiet_fp
def train(kind: str = "hgb", gate: bool = True, model_dir: Path = config.MODEL_DIR,
          report_dir: Path = config.REPORT_DIR, db_path: Path = config.DB_PATH) -> dict:
    df = load_features(config.TRAIN_BATCH, db_path)
    tr, va, te = time_split(df)
    print(f"train {len(tr):,} | valid {len(va):,} | test {len(te):,} rows; fraud rate {df[LABEL].mean():.3%}")

    results, models = {}, {}
    for name, model in [("baseline_logreg", make_baseline()), ("candidate", make_candidate(kind))]:
        model.fit(tr[FEATURES], tr[LABEL])
        s_va = model.predict_proba(va[FEATURES])[:, 1]
        s_te = model.predict_proba(te[FEATURES])[:, 1]
        thr = choose_thresholds(va[LABEL].to_numpy(), s_va, config.ALERT_BUDGET, config.DECLINE_MIN_PRECISION)
        results[name] = {
            "thresholds": thr,
            "valid": evaluate(va[LABEL].to_numpy(), s_va, thr, va.amt.to_numpy()),
            "test": evaluate(te[LABEL].to_numpy(), s_te, thr, te.amt.to_numpy()),
        }
        models[name] = (model, s_va)
        r = results[name]["test"]
        print(f"{name:16s} test PR-AUC {r['pr_auc']:.3f}  recall@review {r['recall_at_review']:.3f}  "
              f"precision@review {r['precision_at_review']:.3f}")

    cand, base = results["candidate"]["test"], results["baseline_logreg"]["test"]
    checks = {
        f"PR-AUC >= {config.GATE_MIN_PR_AUC}": cand["pr_auc"] >= config.GATE_MIN_PR_AUC,
        "PR-AUC >= baseline": cand["pr_auc"] >= base["pr_auc"],
    }
    gate_result = {"passed": all(checks.values()), "checks": checks}

    model, s_va = models["candidate"]
    sample = va.sample(min(len(va), 30_000), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample[LABEL], scoring="average_precision",
                                 n_repeats=3, random_state=0)
    importance = sorted(
        ({"feature": f, "importance": round(float(m), 4)} for f, m in zip(FEATURES, imp.importances_mean)),
        key=lambda d: d["importance"], reverse=True,
    )

    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": f"{kind}-{now:%Y%m%dT%H%M%SZ}",
        "trained_at": now.isoformat(timespec="seconds"),
        "algorithm": type(model.named_steps["clf"]).__name__,
        "features": FEATURES,
        "thresholds": results["candidate"]["thresholds"],
        "policy": {"alert_budget": config.ALERT_BUDGET, "decline_min_precision": config.DECLINE_MIN_PRECISION},
        "data": {"train": _period(tr), "valid": _period(va), "test": _period(te)},
        "metrics": results,
        "gate": gate_result,
        "permutation_importance": importance,
        "reason_reference": {
            **{f: float(tr[f].median()) for f in NUMERIC_FEATURES},
            **{f: str(tr[f].mode().iloc[0]) for f in CATEGORICAL_FEATURES},
        },
        "drift_reference": {
            "features": build_reference(
                tr, [f for f in NUMERIC_FEATURES if f not in DRIFT_EXCLUDE], CATEGORICAL_FEATURES
            ),
            "score": {"type": "numeric", **numeric_profile(s_va)},
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_report.md").write_text(render_report(metadata))
    (report_dir / "metrics.json").write_text(json.dumps(
        {"model_version": metadata["model_version"], "metrics": results, "gate": gate_result},
        indent=2, default=_json_default))

    print(f"Quality gate: {'PASSED' if gate_result['passed'] else 'FAILED'} {checks}")
    if gate_result["passed"] or not gate:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))
        print(f"Published {metadata['model_version']} to {model_dir}")
        _log_mlflow(kind, metadata, report_dir)
    return metadata


def _log_mlflow(kind: str, metadata: dict, report_dir: Path) -> None:
    """Optional experiment tracking: active only if mlflow is installed."""
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=metadata["model_version"]):
        mlflow.log_param("model", kind)
        mlflow.log_params(metadata["policy"])
        for split in ("valid", "test"):
            for k, v in metadata["metrics"]["candidate"][split].items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(f"{split}_{k}", v)
        mlflow.log_artifact(str(report_dir / "training_report.md"))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def render_report(m: dict) -> str:
    keys = ["n", "fraud_rate", "pr_auc", "roc_auc", "alert_rate", "recall_at_review",
            "precision_at_review", "decline_count", "decline_precision", "fraud_amount_caught_share"]
    lines = [
        f"# Training report: `{m['model_version']}`",
        "",
        f"Trained at {m['trained_at']} with **{m['algorithm']}**.",
        "",
        "## Data (time-based split)",
        "",
        "| Split | From | To | Rows | Fraud rate |",
        "|---|---|---|---|---|",
    ]
    for split, d in m["data"].items():
        lines.append(f"| {split} | {d['from']} | {d['to']} | {d['n']:,} | {d['fraud_rate']:.3%} |")
    lines += [
        "",
        "## Results",
        "",
        "Thresholds are chosen on the validation period and applied unchanged to the test period.",
        "",
        "| Metric | Baseline valid | Baseline test | Candidate valid | Candidate test |",
        "|---|---|---|---|---|",
    ]
    b, c = m["metrics"]["baseline_logreg"], m["metrics"]["candidate"]
    for k in keys:
        lines.append(f"| {k} | {_fmt(b['valid'].get(k))} | {_fmt(b['test'].get(k))} | "
                     f"{_fmt(c['valid'].get(k))} | {_fmt(c['test'].get(k))} |")
    t = m["thresholds"]
    lines += [
        "",
        "## Decision policy",
        "",
        f"- REVIEW if score >= **{t['review']:.4f}** (analysts can review {m['policy']['alert_budget']:.0%} "
        "of traffic)",
        f"- DECLINE if score >= **{t['decline']:.4f}** (validation precision >= "
        f"{m['policy']['decline_min_precision']:.0%}; 1.01 means auto-decline disabled)",
        "",
        "## Quality gate",
        "",
    ]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in m["gate"]["checks"].items()]
    lines += [
        "",
        f"**Result: {'PASSED' if m['gate']['passed'] else 'FAILED'}**",
        "",
        "## Permutation importance (drop in validation PR-AUC)",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    lines += [f"| {d['feature']} | {d['importance']:.4f} |" for d in m["permutation_importance"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["hgb", "xgboost"], default="hgb")
    ap.add_argument("--no-gate", action="store_true", help="publish even if the quality gate fails")
    a = ap.parse_args()
    meta = train(a.model, gate=not a.no_gate)
    if not meta["gate"]["passed"] and not a.no_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
