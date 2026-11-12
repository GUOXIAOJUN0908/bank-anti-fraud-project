"""Replay card transactions against the scoring API, the way a payment switch would.

The fraud label in the CSV is NOT sent to the API; it is only used afterwards to
report how many frauds the live service caught (like chargebacks arriving later).

Usage:
  python -m fraud.simulator.replay --csv data/raw/transactions_new.csv --limit 5000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from fraud import config

SEND_COLUMNS = ["trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
                "lat", "long", "merch_lat", "merch_long", "city_pop", "dob"]


def replay(csv: Path, url: str, api_key: str, limit: int, offset: int = 0) -> dict:
    df = pd.read_csv(csv, dtype={"cc_num": str}).sort_values("unix_time").iloc[offset: offset + limit]
    decisions, client_ms, server_ms = [], [], []
    with httpx.Client(base_url=url, headers={"X-API-Key": api_key}, timeout=10) as client:
        client.get("/health").raise_for_status()
        started = time.perf_counter()
        for rec in df[SEND_COLUMNS].to_dict(orient="records"):
            t0 = time.perf_counter()
            resp = client.post("/v1/score", json=rec)
            client_ms.append((time.perf_counter() - t0) * 1000)
            resp.raise_for_status()
            body = resp.json()
            decisions.append(body["decision"])
            server_ms.append(body["latency_ms"])
        elapsed = time.perf_counter() - started

    dec = np.array(decisions)
    y = df.is_fraud.to_numpy()
    flagged = dec != "APPROVE"
    stats = {
        "requests": len(dec),
        "throughput_rps": round(len(dec) / elapsed, 1),
        "approve": int((dec == "APPROVE").sum()),
        "review": int((dec == "REVIEW").sum()),
        "decline": int((dec == "DECLINE").sum()),
        "fraud_in_sample": int(y.sum()),
        "fraud_caught": int((flagged & (y == 1)).sum()),
        "recall": round(float((flagged & (y == 1)).sum() / max(y.sum(), 1)), 3),
        "precision": round(float((flagged & (y == 1)).sum() / max(flagged.sum(), 1)), 3),
        "server_p50_ms": round(float(np.percentile(server_ms, 50)), 2),
        "server_p95_ms": round(float(np.percentile(server_ms, 95)), 2),
        "client_p95_ms": round(float(np.percentile(client_ms, 95)), 2),
        "client_p99_ms": round(float(np.percentile(client_ms, 99)), 2),
    }
    for k, v in stats.items():
        print(f"{k:>18}: {v}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=config.RAW_DIR / "transactions_new.csv")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "dev-key"))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    replay(a.csv, a.url, a.api_key, a.limit, a.offset)


if __name__ == "__main__":
    main()
