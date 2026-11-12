"""Real-time feature computation for the scoring service.

Reproduces sql/features.sql for one incoming transaction, using an in-memory
per-card history. tests/unit/test_feature_consistency.py proves both paths give
the same numbers (no training/serving skew).

Production note: the in-memory store would be replaced by Redis (or the
bank's feature store) so several API replicas share card history.
"""

from __future__ import annotations

import calendar
import math
import threading
from collections import defaultdict, deque
from datetime import date, datetime

from fraud.config import WINDOW_1H, WINDOW_24H, WINDOW_30D


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def to_unix(ts: datetime) -> int:
    """Timestamps are naive UTC, same convention as the source extract."""
    return calendar.timegm(ts.timetuple())


class OnlineFeatureStore:
    """Keeps the last 30 days of (unix_time, amount) per card."""

    def __init__(self) -> None:
        self._history: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def compute_and_update(
        self,
        *,
        cc_num: str,
        ts: datetime,
        category: str,
        amt: float,
        lat: float,
        lon: float,
        merch_lat: float,
        merch_long: float,
        city_pop: int,
        dob: date,
    ) -> dict:
        t = to_unix(ts)
        with self._lock:
            hist = self._history[cc_num]
            while hist and hist[0][0] < t - WINDOW_30D:
                hist.popleft()
            prev_t = hist[-1][0] if hist else None
            in_1h = [a for (pt, a) in hist if pt >= t - WINDOW_1H]
            in_24h = [a for (pt, a) in hist if pt >= t - WINDOW_24H]
            prior_30d = [a for (_, a) in hist]
            hist.append((t, amt))

        avg_30d = sum(prior_30d) / len(prior_30d) if prior_30d else None
        gap = WINDOW_30D if prev_t is None else min(t - prev_t, WINDOW_30D)
        return {
            "category": category,
            "amt": amt,
            "hour_of_day": ts.hour,
            "day_of_week": (ts.weekday() + 1) % 7,  # 0 = Sunday, like SQL EXTRACT(DOW)
            "is_night": int(ts.hour >= 22 or ts.hour < 6),
            "age": ts.year - dob.year,
            "city_pop": city_pop,
            "distance_km": haversine_km(lat, lon, merch_lat, merch_long),
            "txn_count_1h": len(in_1h) + 1,
            "txn_count_24h": len(in_24h) + 1,
            "amt_sum_24h": sum(in_24h) + amt,
            "amt_ratio_to_avg_30d": amt / avg_30d if avg_30d else 1.0,
            "secs_since_last_txn": gap,
        }

    def bootstrap(self, df) -> int:
        """Hydrate card history from a warehouse extract (no scoring), oldest first."""
        df = df.sort_values("unix_time")
        for r in df.itertuples(index=False):
            self.compute_and_update(
                cc_num=str(r.cc_num), ts=r.trans_date_trans_time.to_pydatetime(), category=r.category,
                amt=float(r.amt), lat=r.lat, lon=r.long, merch_lat=r.merch_lat, merch_long=r.merch_long,
                city_pop=int(r.city_pop), dob=r.dob,
            )
        return len(df)

    def __len__(self) -> int:
        return len(self._history)
