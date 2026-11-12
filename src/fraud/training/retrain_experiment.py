"""RCA-001 corrective action: does retraining on recent labelled data restore recall?

Champion   = the published model (trained on Jan-Apr only).
Challenger = same pipeline, retrained on Jan-Apr + the first 10 days of the new period,
             thresholds re-chosen on new-period days 10-15.
Both are compared on new-period days 15-30, which neither model has seen.

Usage:
  python -m fraud.training.retrain_experiment
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from fraud import config
from fraud.features.build import load_features
from fraud.features.schema import FEATURES, LABEL
from fraud.training.evaluate import choose_thresholds, evaluate
from fraud.training.train import make_candidate, quiet_fp


@quiet_fp
def main() -> None:
    hist = load_features(config.TRAIN_BATCH)
    new = load_features("transactions_new")
    t0 = new.unix_time.min()
    d10, d15 = t0 + 10 * 86_400, t0 + 15 * 86_400
    new_train, new_valid, new_test = new[new.unix_time < d10], new[(new.unix_time >= d10) & (new.unix_time < d15)], \
        new[new.unix_time >= d15]

    meta = json.loads((config.MODEL_DIR / "metadata.json").read_text())
    champion = joblib.load(config.MODEL_DIR / "model.joblib")

    challenger = make_candidate("hgb")
    train = pd.concat([hist, new_train], ignore_index=True)
    challenger.fit(train[FEATURES], train[LABEL])
    thr = choose_thresholds(new_valid[LABEL].to_numpy(), challenger.predict_proba(new_valid[FEATURES])[:, 1],
                            config.ALERT_BUDGET, config.DECLINE_MIN_PRECISION)

    y, amt = new_test[LABEL].to_numpy(), new_test.amt.to_numpy()
    rows = {
        "champion (Jan-Apr)": evaluate(y, champion.predict_proba(new_test[FEATURES])[:, 1], meta["thresholds"], amt),
        "challenger (+ new days 0-10)": evaluate(y, challenger.predict_proba(new_test[FEATURES])[:, 1], thr, amt),
    }
    keys = ["n", "fraud_rate", "pr_auc", "alert_rate", "recall_at_review", "precision_at_review",
            "fraud_amount_caught_share"]
    lines = [
        "# RCA-001 corrective action: champion vs. challenger",
        "",
        f"Evaluation window: new-period days 15-30 ({len(new_test):,} transactions, "
        f"{int(y.sum())} frauds). Neither model trained on it.",
        "",
        "| Metric | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for k in keys:
        cells = [f"{r[k]:,}" if isinstance(r[k], int) else f"{r[k]:.4f}" for r in rows.values()]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "rca_experiment.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
