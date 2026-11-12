"""Central paths and tunable settings. Every path can be overridden by an env var."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default)))


RAW_DIR = _path("FRAUD_RAW_DIR", ROOT / "data" / "raw")
DB_PATH = _path("FRAUD_DB_PATH", ROOT / "data" / "warehouse.duckdb")
SQL_DIR = ROOT / "sql"
MODEL_DIR = _path("MODEL_DIR", ROOT / "models")
REPORT_DIR = _path("FRAUD_REPORT_DIR", ROOT / "reports")
PREDICTION_LOG = _path("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl")

# Name of the batch (raw CSV file stem) used for training.
TRAIN_BATCH = "transactions"

# Business settings used to pick decision thresholds on the validation set.
ALERT_BUDGET = 0.01  # fraud analysts can review ~1% of transactions
DECLINE_MIN_PRECISION = 0.90  # auto-decline only when we are this sure

# Model quality gate (CI fails if the candidate model does not pass).
GATE_MIN_PR_AUC = 0.60

# Feature windows in seconds. Must match sql/features.sql.
WINDOW_1H = 3_600
WINDOW_24H = 86_400
WINDOW_30D = 2_592_000
