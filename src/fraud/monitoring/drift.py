"""Drift and performance monitoring for the live model.

Two modes:
  --from-log logs/predictions.jsonl     what the live service actually saw (no labels needed)
             [--labels data/raw/transactions_new.csv]   add late-arriving fraud labels (chargebacks)
  --batch transactions_new             offline check on a warehouse batch (labels available)

Writes reports/drift_report.md and reports/drift_report.json.
Exit code 2 with --fail-on-alert if any PSI is in ALERT, so it can gate a scheduled job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud import config
from fraud.features.build import load_features
from fraud.features.schema import FEATURES, LABEL, NUMERIC_FEATURES
from fraud.monitoring.psi import PSI_ALERT, PSI_WARN, compare
from fraud.training.evaluate import evaluate

RECALL_DROP_ALERT = 0.10  # absolute drop in recall vs. the model's test period


def _from_log(path: Path, labels: Path | None) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.json_normalize(records)
    df.columns = [c.replace("features.", "") for c in df.columns]
    if labels is not None:
        lab = pd.read_csv(labels, usecols=["trans_num", LABEL])
        df = df.merge(lab, on="trans_num", how="left")
    return df


def _from_batch(batch: str, model_dir: Path) -> pd.DataFrame:
    df = load_features(batch)
    model = joblib.load(model_dir / "model.joblib")
    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    return df


def missed_fraud_profile(df: pd.DataFrame, thresholds: dict) -> dict:
    """Compare frauds the model caught vs. missed: the starting point of any RCA."""
    fraud = df[df[LABEL] == 1]
    missed = fraud.score < thresholds["review"]
    def describe(part: pd.DataFrame) -> dict:
        if part.empty:
            return {}
        return {
            "count": int(len(part)),
            "median_amt": round(float(part.amt.median()), 2),
            "median_txn_count_1h": float(part.txn_count_1h.median()),
            "median_distance_km": round(float(part.distance_km.median()), 1),
            "top_categories": part.category.value_counts().head(3).to_dict(),
        }
    return {"caught": describe(fraud[~missed]), "missed": describe(fraud[missed])}


def run(df: pd.DataFrame, meta: dict, source: str) -> dict:
    ref = meta["drift_reference"]
    feature_psi = compare(ref["features"], df)
    score_psi = compare({"score": ref["score"]}, df)[0]
    report = {
        "source": source,
        "model_version": meta["model_version"],
        "rows": int(len(df)),
        "feature_psi": feature_psi,
        "score_psi": score_psi,
        "alerts": [r["feature"] for r in feature_psi + [score_psi] if r["status"] == "ALERT"],
    }
    # Labels (chargebacks) arrive late, so only part of the traffic may be labelled yet.
    labelled = df[df[LABEL].notna()] if LABEL in df else df.iloc[0:0]
    if len(labelled) and labelled[LABEL].sum() > 0:
        y = labelled[LABEL].to_numpy().astype(int)
        perf = evaluate(y, labelled.score.to_numpy(), meta["thresholds"], labelled.amt.to_numpy())
        baseline = meta["metrics"]["candidate"]["test"]
        drop = baseline["recall_at_review"] - perf["recall_at_review"]
        report["label_coverage"] = round(len(labelled) / len(df), 4)
        report["performance"] = {"current": perf, "at_training_test": baseline, "recall_drop": round(drop, 4)}
        report["performance_alert"] = bool(drop > RECALL_DROP_ALERT)
        report["fraud_profile"] = missed_fraud_profile(labelled, meta["thresholds"])
        if report["performance_alert"]:
            report["alerts"].append("recall_at_review")
    return report


def render(r: dict) -> str:
    lines = [
        "# Drift & performance report",
        "",
        f"- Source: `{r['source']}` ({r['rows']:,} transactions)",
        f"- Model: `{r['model_version']}`",
        f"- PSI bands: OK < {PSI_WARN} | WARN {PSI_WARN}-{PSI_ALERT} | ALERT > {PSI_ALERT}",
        "",
        "## Population stability",
        "",
        "| Feature | PSI | Status |",
        "|---|---|---|",
    ]
    for row in [r["score_psi"]] + r["feature_psi"]:
        name = "**model score**" if row["feature"] == "score" else row["feature"]
        lines.append(f"| {name} | {row['psi']:.4f} | {row['status']} |")
    if "performance" in r:
        p, b = r["performance"]["current"], r["performance"]["at_training_test"]
        n_fraud = int(round(p["n"] * p["fraud_rate"]))
        lines += [
            "",
            "## Performance (labels available)",
            "",
            f"Label coverage: {r['label_coverage']:.1%} of transactions ({p['n']:,} labelled, {n_fraud} frauds).",
            "",
            "| Metric | At training (test period) | Now |",
            "|---|---|---|",
        ]
        for k in ["fraud_rate", "pr_auc", "alert_rate", "recall_at_review", "precision_at_review",
                  "fraud_amount_caught_share"]:
            lines.append(f"| {k} | {b.get(k, float('nan')):.4f} | {p.get(k, float('nan')):.4f} |")
        status = "ALERT" if r["performance_alert"] else "OK"
        lines += ["", f"Recall drop: **{r['performance']['recall_drop']:+.1%}** -> {status} "
                      f"(alert if > {RECALL_DROP_ALERT:.0%})", "", "## Caught vs. missed fraud", ""]
        lines += ["| | Caught | Missed |", "|---|---|---|"]
        c, m = r["fraud_profile"]["caught"], r["fraud_profile"]["missed"]
        for k in ["count", "median_amt", "median_txn_count_1h", "median_distance_km", "top_categories"]:
            lines.append(f"| {k} | {c.get(k, '-')} | {m.get(k, '-')} |")
    lines += ["", f"**Alerts:** {', '.join(r['alerts']) if r['alerts'] else 'none'}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-log", type=Path)
    src.add_argument("--batch")
    ap.add_argument("--labels", type=Path, help="CSV with trans_num,is_fraud (late labels)")
    ap.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    ap.add_argument("--out", type=Path, default=config.REPORT_DIR / "drift_report.md")
    ap.add_argument("--fail-on-alert", action="store_true")
    a = ap.parse_args()

    meta = json.loads((a.model_dir / "metadata.json").read_text())
    if a.from_log:
        df, source = _from_log(a.from_log, a.labels), str(a.from_log)
    else:
        df, source = _from_batch(a.batch, a.model_dir), f"warehouse batch {a.batch}"
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)

    report = run(df, meta, source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(report))
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=lambda o: o.item()
                                                     if isinstance(o, np.generic) else str(o)))
    print(render(report))
    if a.fail_on_alert and report["alerts"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
