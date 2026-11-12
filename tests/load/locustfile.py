"""Performance test: sustained load on /v1/score.

  pip install locust
  API_KEY=dev-key locust -f tests/load/locustfile.py --host http://localhost:8000 \
      --users 50 --spawn-rate 10 --run-time 2m --headless

Pass criteria (docs/test_plan.md, PT-01): p95 < 100 ms, error rate < 0.1%.
"""

import os
import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task

CATEGORIES = ["grocery_pos", "gas_transport", "food_dining", "shopping_pos", "shopping_net", "misc_net"]


class PaymentSwitch(HttpUser):
    wait_time = between(0.01, 0.05)

    def on_start(self):
        self.client.headers["X-API-Key"] = os.environ.get("API_KEY", "dev-key")
        self.card = f"4{random.randint(0, 10**15 - 1):015d}"
        self.clock = datetime(2026, 6, 1, 8, 0)
        self.n = 0

    @task
    def score(self):
        self.n += 1
        self.clock += timedelta(minutes=random.randint(1, 240))
        self.client.post("/v1/score", name="/v1/score", json={
            "trans_num": f"LOAD-{self.card[-6:]}-{self.n}",
            "trans_date_trans_time": self.clock.isoformat(sep=" "),
            "cc_num": self.card,
            "merchant": "merchant_load_test",
            "category": random.choice(CATEGORIES),
            "amt": round(random.lognormvariate(3.8, 0.8), 2),
            "lat": 1.3521, "long": 103.8198,
            "merch_lat": 1.3521 + random.gauss(0, 0.05), "merch_long": 103.8198 + random.gauss(0, 0.05),
            "city_pop": 5_600_000, "dob": "1990-01-01",
        })
