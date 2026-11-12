"""Generate synthetic card transactions.

The output uses the same column names as the public Kaggle dataset
"Credit Card Transactions Fraud Detection" (Sparkov simulator), so the real
data can be dropped into data/raw/ later without code changes.

Two files are written:
  data/raw/transactions.csv      history used for training (fraud patterns A + B)
  data/raw/transactions_new.csv  a later period where fraudsters change tactics
                                 (pattern C "card testing") and normal spend rises,
                                 used to exercise drift monitoring and the RCA.

Usage:
  python -m fraud.data.generate --customers 2000 --days 120 --new-days 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud import config

# category: (share of normal transactions, median amount, lognormal sigma)
CATEGORIES = {
    "grocery_pos": (0.16, 55, 0.6),
    "gas_transport": (0.12, 45, 0.5),
    "food_dining": (0.10, 28, 0.6),
    "shopping_pos": (0.10, 70, 0.8),
    "shopping_net": (0.08, 65, 0.9),
    "home": (0.08, 60, 0.8),
    "entertainment": (0.07, 45, 0.7),
    "kids_pets": (0.07, 40, 0.7),
    "personal_care": (0.06, 25, 0.7),
    "health_fitness": (0.06, 40, 0.6),
    "misc_pos": (0.04, 35, 0.9),
    "misc_net": (0.03, 40, 1.0),
    "grocery_net": (0.02, 50, 0.6),
    "travel": (0.01, 220, 0.9),
}
CAT_NAMES = list(CATEGORIES)
CAT_SHARES = np.array([v[0] for v in CATEGORIES.values()])

# (city, lat, lon, population, share of customers)
HOME_CITIES = [
    ("Singapore", 1.3521, 103.8198, 5_600_000, 0.70),
    ("Johor Bahru", 1.4927, 103.7414, 500_000, 0.10),
    ("Kuala Lumpur", 3.1390, 101.6869, 1_800_000, 0.10),
    ("Penang", 5.4141, 100.3288, 700_000, 0.05),
    ("Batam", 1.0456, 104.0305, 1_200_000, 0.05),
]

# Relative likelihood of a genuine transaction in each hour of the day.
HOUR_WEIGHTS = np.array(
    [0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 1.0, 2.0, 3.5, 4.0, 4.5, 5.0,
     5.5, 5.0, 4.5, 4.5, 4.5, 5.0, 5.5, 5.5, 5.0, 4.0, 2.5, 1.2]
)
HOUR_PROBS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

COMPROMISE_RATE_PER_30D = 0.03  # share of cards compromised in any 30-day window
START = np.datetime64("2026-01-01T00:00:00")


def _far_location(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Random location somewhere else in the world (cross-border use)."""
    return rng.uniform(-35, 55, n), rng.uniform(-120, 150, n)


def _make_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    city_idx = rng.choice(len(HOME_CITIES), size=n, p=[c[4] for c in HOME_CITIES])
    cities = [HOME_CITIES[i] for i in city_idx]
    ages = rng.integers(18, 80, n)
    return pd.DataFrame(
        {
            "cc_num": [f"4{x:015d}" for x in rng.integers(0, 10**15, n)],
            "gender": rng.choice(["F", "M"], n),
            "city": [c[0] for c in cities],
            "lat": np.array([c[1] for c in cities]) + rng.normal(0, 0.03, n),
            "long": np.array([c[2] for c in cities]) + rng.normal(0, 0.03, n),
            "city_pop": [c[3] for c in cities],
            "dob": [f"{2026 - a}-{m:02d}-{d:02d}" for a, m, d in
                    zip(ages, rng.integers(1, 13, n), rng.integers(1, 29, n))],
            "rate": np.clip(np.exp(rng.normal(np.log(1.2), 0.5, n)), 0.2, 6.0),
            "spend": np.exp(rng.normal(0, 0.35, n)),
        }
    )


def _normal_txns(rng, cust, prefs, t0: int, t1: int, inflation: float) -> dict:
    days = (t1 - t0) / 86_400
    n = rng.poisson(cust.rate * days)
    day = rng.integers(0, int(days), n)
    hour = rng.choice(24, n, p=HOUR_PROBS)
    secs = t0 + day * 86_400 + hour * 3_600 + rng.integers(0, 3_600, n)
    cat = rng.choice(len(CAT_NAMES), n, p=prefs)
    med = np.array([CATEGORIES[CAT_NAMES[c]][1] for c in cat])
    sig = np.array([CATEGORIES[CAT_NAMES[c]][2] for c in cat])
    amt = np.exp(rng.normal(np.log(med), sig)) * cust.spend * inflation
    m_lat = cust.lat + rng.normal(0, 0.05, n)
    m_lon = cust.long + rng.normal(0, 0.05, n)
    travel = rng.random(n) < 0.015  # genuine overseas spend, so distance alone is not a giveaway
    f_lat, f_lon = _far_location(rng, int(travel.sum()))
    m_lat[travel], m_lon[travel] = f_lat, f_lon
    return {"secs": secs, "cat": cat, "amt": amt, "m_lat": m_lat, "m_lon": m_lon, "fraud": np.zeros(n, int)}


def _fraud_txns(rng, cust, t0: int, t1: int, pattern: str) -> dict:
    """One fraud episode on a compromised card."""
    day = rng.integers(0, max(1, (t1 - t0) // 86_400 - 1))
    if pattern == "A":  # card-not-present spree: online, big amounts, often at night, overseas
        k, span_h = rng.integers(3, 11), rng.uniform(1, 6)
        hour = rng.choice([22, 23, 0, 1, 2, 3, 4]) if rng.random() < 0.6 else rng.integers(8, 22)
        cats = rng.choice(["shopping_net", "misc_net", "travel", "grocery_net"], k, p=[0.45, 0.3, 0.15, 0.1])
        amt = np.clip(np.exp(rng.normal(np.log(350), 0.7, k)), 20, 3_000)
        if rng.random() < 0.6:
            m_lat, m_lon = _far_location(rng, k)
        else:
            m_lat, m_lon = cust.lat + rng.normal(0, 0.05, k), cust.long + rng.normal(0, 0.05, k)
    elif pattern == "B":  # skimmed card used in shops in another city
        k, span_h = rng.integers(2, 6), rng.uniform(1, 12)
        hour = rng.integers(9, 21)
        cats = rng.choice(["grocery_pos", "gas_transport", "shopping_pos", "misc_pos"], k)
        amt = np.clip(np.exp(rng.normal(np.log(250), 0.6, k)), 20, 2_000)
        angle, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 4.0)
        m_lat = cust.lat + dist * np.sin(angle) + rng.normal(0, 0.05, k)
        m_lon = cust.long + dist * np.cos(angle) + rng.normal(0, 0.05, k)
    else:  # "C" card testing: many tiny online payments near home, daytime
        k, span_h = rng.integers(5, 16), rng.uniform(0.5, 1.5)
        hour = rng.integers(9, 21)
        cats = rng.choice(["misc_net", "shopping_net", "food_dining"], k, p=[0.5, 0.3, 0.2])
        amt = rng.uniform(1, 15, k)
        m_lat, m_lon = cust.lat + rng.normal(0, 0.05, k), cust.long + rng.normal(0, 0.05, k)
    start = t0 + day * 86_400 + hour * 3_600
    secs = start + np.sort(rng.integers(0, int(span_h * 3_600), k))
    cat = np.array([CAT_NAMES.index(c) for c in cats])
    return {"secs": secs, "cat": cat, "amt": amt, "m_lat": m_lat, "m_lon": m_lon, "fraud": np.ones(k, int)}


def _period(rng, customers, prefs, t0, t1, patterns, pattern_p, inflation) -> pd.DataFrame:
    p_compromise = COMPROMISE_RATE_PER_30D * (t1 - t0) / (30 * 86_400)
    parts = []
    for i, cust in enumerate(customers.itertuples()):
        chunks = [_normal_txns(rng, cust, prefs[i], t0, t1, inflation)]
        if rng.random() < p_compromise:
            chunks.append(_fraud_txns(rng, cust, t0, t1, rng.choice(patterns, p=pattern_p)))
        for ch in chunks:
            ch["cust"] = np.full(len(ch["secs"]), i)
            parts.append(ch)
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    df = pd.DataFrame(cols)
    df = df[df.secs < t1].drop_duplicates(subset=["cust", "secs"])  # unique timestamp per card
    return df


def _to_frame(df: pd.DataFrame, customers: pd.DataFrame, rng) -> pd.DataFrame:
    c = customers.iloc[df.cust.to_numpy()].reset_index(drop=True)
    ts = START + df.secs.to_numpy().astype("timedelta64[s]")
    cat = np.array(CAT_NAMES)[df.cat.to_numpy()]
    out = pd.DataFrame(
        {
            "trans_date_trans_time": np.char.replace(np.datetime_as_string(ts, unit="s"), "T", " "),
            "unix_time": ts.astype("datetime64[s]").astype(np.int64),
            "cc_num": c.cc_num,
            "merchant": [f"merchant_{k}_{j:02d}" for k, j in zip(cat, rng.integers(0, 40, len(df)))],
            "category": cat,
            "amt": np.round(np.clip(df.amt.to_numpy(), 1.0, None), 2),
            "gender": c.gender,
            "city": c.city,
            "lat": c.lat.round(6),
            "long": c.long.round(6),
            "city_pop": c.city_pop,
            "dob": c.dob,
            "merch_lat": np.round(df.m_lat.to_numpy(), 6),
            "merch_long": np.round(df.m_lon.to_numpy(), 6),
            "is_fraud": df.fraud.to_numpy(),
        }
    )
    return out.sort_values(["unix_time", "cc_num"]).reset_index(drop=True)


def generate(customers: int, days: int, new_days: int, seed: int, out_dir: Path) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    cust = _make_customers(rng, customers)
    prefs = rng.dirichlet(CAT_SHARES * 20, customers)
    t_hist = days * 86_400

    hist = _to_frame(_period(rng, cust, prefs, 0, t_hist, ["A", "B"], [0.6, 0.4], 1.0), cust, rng)
    frames = {"transactions": hist}
    if new_days > 0:
        new = _period(rng, cust, prefs, t_hist, t_hist + new_days * 86_400,
                      ["A", "B", "C"], [0.2, 0.1, 0.7], 1.15)
        frames["transactions_new"] = _to_frame(new, cust, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    offset, paths = 0, {}
    for name, frame in frames.items():
        frame.insert(0, "trans_num", [f"T{i:09d}" for i in range(offset, offset + len(frame))])
        offset += len(frame)
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
        print(f"{path}: {len(frame):,} rows, fraud rate {frame.is_fraud.mean():.3%}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customers", type=int, default=2000)
    ap.add_argument("--days", type=int, default=120, help="length of the training history")
    ap.add_argument("--new-days", type=int, default=30, help="length of the later 'new' period (0 = skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=config.RAW_DIR)
    a = ap.parse_args()
    generate(a.customers, a.days, a.new_days, a.seed, a.out_dir)


if __name__ == "__main__":
    main()
