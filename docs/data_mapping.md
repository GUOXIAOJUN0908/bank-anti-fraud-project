# Data Mapping Specification

| | |
|---|---|
| Document ID | DMS-FRAUD-001 |
| Source | Card transaction extract (`data/raw/transactions*.csv`), schema of the public Kaggle "Credit Card Transactions Fraud Detection" dataset (Sparkov) |
| Target | `features` table (warehouse) and the online feature vector (API) |
| Logic | [`sql/features.sql`](../sql/features.sql) (batch) = [`src/fraud/features/online.py`](../src/fraud/features/online.py) (real time), parity tested in `tests/unit/test_feature_consistency.py` |

## 1. Source fields

| Source field | Type | Example | Classification | Used for |
|---|---|---|---|---|
| trans_num | VARCHAR | T000123456 | Internal id | Join key, audit |
| trans_date_trans_time | TIMESTAMP (UTC) | 2026-05-01 14:03:22 | Internal | Time features, ordering |
| unix_time | BIGINT | 1777644202 | Internal | Ignored by the feature SQL, which re-derives epoch seconds from `trans_date_trans_time` (in the public Kaggle extract this column does not line up with the timestamp) |
| cc_num | VARCHAR | 4xxxxxxxxxxx1234 | **PCI / personal** | Partition key only; masked in logs |
| merchant | VARCHAR | merchant_grocery_pos_07 | Internal | Not used (too sparse); kept for case review |
| category | VARCHAR | grocery_pos | Internal | Feature |
| amt | DOUBLE | 42.50 | Internal | Feature + aggregates |
| gender | VARCHAR | F | **Personal, protected** | **Not used**, deliberately excluded (fairness) |
| city | VARCHAR | Singapore | Personal | Not used |
| lat, long | DOUBLE | 1.3521, 103.8198 | **Personal (home location)** | Distance only; never logged |
| city_pop | BIGINT | 5600000 | Internal | Feature |
| dob | DATE | 1988-07-14 | **Personal** | Age only; never logged |
| merch_lat, merch_long | DOUBLE | 1.36, 103.83 | Internal | Distance |
| is_fraud | INTEGER | 0/1 | Label (arrives late via chargebacks) | Training target, monitoring |

## 2. Feature mapping

All windows are per card (`PARTITION BY cc_num`), ordered by `unix_time`, and include only the current and earlier
transactions, so no feature can see the future.

| Feature | Source fields | Transformation | Missing / first-transaction rule |
|---|---|---|---|
| category | category | As-is, native categorical | Unseen category -> treated as unknown (NaN) |
| amt | amt | As-is | - (validated > 0) |
| hour_of_day | trans_date_trans_time | `EXTRACT(HOUR)` (UTC) | - |
| day_of_week | trans_date_trans_time | `EXTRACT(DOW)`, 0 = Sunday | - |
| is_night | trans_date_trans_time | 1 if hour >= 22 or < 6 | - |
| age | dob, trans_date_trans_time | year(ts) - year(dob) | - |
| city_pop | city_pop | As-is | - |
| distance_km | lat, long, merch_lat, merch_long | Haversine, earth radius 6,371 km | - |
| txn_count_1h | cc_num, unix_time | COUNT over `RANGE 3600 PRECEDING` (incl. current) | 1 |
| txn_count_24h | cc_num, unix_time | COUNT over `RANGE 86400 PRECEDING` | 1 |
| amt_sum_24h | amt | SUM over `RANGE 86400 PRECEDING` | = amt |
| amt_ratio_to_avg_30d | amt | amt / average of the PREVIOUS 30 days (current excluded) | 1.0 |
| secs_since_last_txn | unix_time | unix_time - LAG(unix_time), capped at 30 days | 2,592,000 (cap) |

## 3. Data quality rules

| Rule | Where enforced | Action on failure |
|---|---|---|
| Typed columns (timestamps, numbers) | `read_csv(types=...)` in build.py | Load fails loudly |
| amt > 0, coordinates in range, PAN 12-19 digits | API schema (`schemas.py`) | HTTP 422 with field detail |
| One timestamp per card per second | Generator; documented assumption | Same-second events are treated as simultaneous (both counted) |
| Raw file names match `[A-Za-z0-9_]+` | build.py | Refuse to load |

## 4. Outputs

| Output | Location | Retention (proposed) |
|---|---|---|
| `transactions`, `features` tables | warehouse | 13 months (regulatory look-back) |
| Model artefact + metadata | `models/` | Every released version kept |
| Audit log (masked) | `logs/predictions.jsonl` | 5 years (MAS record keeping) |
