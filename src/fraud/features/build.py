"""Load raw transaction extracts into the warehouse and build the feature table.

DuckDB is used as a local stand-in for the bank's PostgreSQL warehouse; the
feature SQL in sql/features.sql is portable to both.

Usage:
  python -m fraud.features.build                 # all data/raw/transactions*.csv
  python -m fraud.features.build --raw-dir other/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd

from fraud import config

RAW_TYPES = {
    "trans_num": "VARCHAR",
    "trans_date_trans_time": "TIMESTAMP",
    "unix_time": "BIGINT",
    "cc_num": "VARCHAR",
    "category": "VARCHAR",
    "amt": "DOUBLE",
    "lat": "DOUBLE",
    "long": "DOUBLE",
    "city_pop": "BIGINT",
    "dob": "DATE",
    "merch_lat": "DOUBLE",
    "merch_long": "DOUBLE",
    "is_fraud": "INTEGER",
}
FEATURE_SQL = config.SQL_DIR / "features.sql"


def build(raw_dir: Path = config.RAW_DIR, db_path: Path = config.DB_PATH) -> int:
    files = sorted(raw_dir.glob("transactions*.csv"))
    if not files:
        raise SystemExit(f"No transactions*.csv found in {raw_dir}. Run `make data` first.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        selects = []
        for f in files:
            if not re.fullmatch(r"[A-Za-z0-9_]+", f.stem):
                raise SystemExit(f"Refusing unexpected file name: {f.name}")
            path = str(f).replace("'", "''")  # file names are validated/escaped; no user input reaches SQL
            selects.append(
                f"SELECT *, '{f.stem}' AS batch FROM read_csv('{path}', header=true, "  # noqa: S608
                f"auto_detect=true, types={RAW_TYPES!r})"
            )
        con.execute("CREATE OR REPLACE TABLE transactions AS " + " UNION ALL BY NAME ".join(selects))
        con.execute(f"CREATE OR REPLACE TABLE features AS {FEATURE_SQL.read_text()}")
        summary = con.execute(
            "SELECT batch, COUNT(*) AS n_rows, AVG(is_fraud) AS fraud_rate, "
            "MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM features GROUP BY batch ORDER BY batch"
        ).df()
    finally:
        con.close()
    with pd.option_context("display.width", 140):
        print(summary.to_string(index=False))
    return int(summary.n_rows.sum())


def load_features(batch: str = config.TRAIN_BATCH, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM features WHERE batch = ? ORDER BY unix_time, trans_num", [batch]).df()
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    a = ap.parse_args()
    build(a.raw_dir, a.db)


if __name__ == "__main__":
    main()
