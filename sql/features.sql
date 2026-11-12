-- Point-in-time feature engineering for card transactions.
--
-- Every feature for a transaction uses only that transaction and EARLIER
-- transactions of the same card, so there is no look-ahead leakage and the
-- online scoring service (src/fraud/features/online.py) can reproduce the
-- exact same values in real time.
--
-- Portable SQL: runs on DuckDB (local) and PostgreSQL 11+ (RANGE offsets on
-- integer columns, standard window functions, no vendor-specific date maths).
-- Source table: transactions (one row per card transaction, see docs/data_mapping.md).

WITH base AS (
    SELECT
        trans_num,
        batch,
        cc_num,
        -- Derived from the timestamp, not the source unix_time column: in the public Kaggle extract that
        -- column does not line up with trans_date_trans_time. Naive timestamps are UTC in both engines.
        CAST(EXTRACT(EPOCH FROM trans_date_trans_time) AS BIGINT) AS unix_time,
        trans_date_trans_time AS ts,
        category,
        amt,
        lat,
        "long" AS lon,
        merch_lat,
        merch_long,
        city_pop,
        dob,
        is_fraud
    FROM transactions
),

windowed AS (
    SELECT
        *,
        COUNT(*) OVER w_1h  AS txn_count_1h,
        COUNT(*) OVER w_24h AS txn_count_24h,
        SUM(amt) OVER w_24h AS amt_sum_24h,
        COUNT(*) OVER w_30d AS cnt_30d,
        SUM(amt) OVER w_30d AS sum_30d,
        unix_time - LAG(unix_time) OVER (PARTITION BY cc_num ORDER BY unix_time) AS gap_secs
    FROM base
    WINDOW
        w_1h  AS (PARTITION BY cc_num ORDER BY unix_time RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW),
        w_24h AS (PARTITION BY cc_num ORDER BY unix_time RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW),
        w_30d AS (PARTITION BY cc_num ORDER BY unix_time RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW)
)

SELECT
    trans_num,
    batch,
    cc_num,
    unix_time,
    ts,
    is_fraud,
    category,
    amt,
    CAST(EXTRACT(HOUR FROM ts) AS INTEGER) AS hour_of_day,
    CAST(EXTRACT(DOW FROM ts) AS INTEGER) AS day_of_week,           -- 0 = Sunday
    CASE WHEN EXTRACT(HOUR FROM ts) >= 22 OR EXTRACT(HOUR FROM ts) < 6 THEN 1 ELSE 0 END AS is_night,
    CAST(EXTRACT(YEAR FROM ts) - EXTRACT(YEAR FROM dob) AS INTEGER) AS age,
    city_pop,
    -- Haversine distance between cardholder home and merchant, in km.
    2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(merch_lat - lat) / 2), 2)
        + COS(RADIANS(lat)) * COS(RADIANS(merch_lat)) * POWER(SIN(RADIANS(merch_long - lon) / 2), 2)
    )) AS distance_km,
    txn_count_1h,
    txn_count_24h,
    amt_sum_24h,
    -- Current amount vs. the card's average over the PREVIOUS 30 days (current row excluded).
    CASE WHEN cnt_30d > 1 THEN amt / ((sum_30d - amt) / (cnt_30d - 1)) ELSE 1.0 END AS amt_ratio_to_avg_30d,
    -- Capped at 30 days; first-ever transaction gets the cap.
    LEAST(COALESCE(gap_secs, 2592000), 2592000) AS secs_since_last_txn
FROM windowed
ORDER BY unix_time, trans_num
