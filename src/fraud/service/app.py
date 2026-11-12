"""Real-time fraud scoring API.

Run locally:
  API_KEY=dev-key uvicorn fraud.service.app:app --port 8000
Interactive API docs (OpenAPI): http://localhost:8000/docs

Environment variables
  API_KEY                 shared secret expected in the X-API-Key header (from a secret manager in prod)
  MODEL_DIR               folder with model.joblib + metadata.json
  PREDICTION_LOG          JSONL audit log of every decision (card numbers masked)
  FEATURE_BOOTSTRAP_CSV   optional extract used to hydrate the last 30 days of card history at startup
"""

import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from fraud import config
from fraud.features.online import OnlineFeatureStore
from fraud.features.schema import FEATURES, NUMERIC_FEATURES
from fraud.security import mask_card
from fraud.service.schemas import ScoreOut, TransactionIn
from fraud.training.explain import reason_codes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("fraud.service")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ScoringEngine:
    def __init__(self, model_dir: Path, log_path: Path, bootstrap_csv: Optional[Path] = None) -> None:
        self.model = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.thresholds = self.meta["thresholds"]
        self.version = self.meta["model_version"]
        self.features = OnlineFeatureStore()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()
        if bootstrap_csv and bootstrap_csv.exists():
            self._bootstrap(bootstrap_csv)

    def _bootstrap(self, path: Path) -> None:
        df = pd.read_csv(path, dtype={"cc_num": str}, parse_dates=["trans_date_trans_time", "dob"])
        df = df[df.unix_time >= df.unix_time.max() - config.WINDOW_30D]
        df["dob"] = df["dob"].dt.date
        n = self.features.bootstrap(df)
        log.info("Feature store hydrated with %d transactions for %d cards", n, len(self.features))

    def score(self, tx: TransactionIn) -> ScoreOut:
        start = time.perf_counter()
        ts = tx.trans_date_trans_time
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        feats = self.features.compute_and_update(
            cc_num=tx.cc_num, ts=ts, category=tx.category, amt=tx.amt, lat=tx.lat, lon=tx.long,
            merch_lat=tx.merch_lat, merch_long=tx.merch_long, city_pop=tx.city_pop, dob=tx.dob,
        )
        row = pd.DataFrame([feats], columns=FEATURES).astype({f: float for f in NUMERIC_FEATURES})
        score = float(self.model.predict_proba(row)[0, 1])
        if score >= self.thresholds["decline"]:
            decision = "DECLINE"
        elif score >= self.thresholds["review"]:
            decision = "REVIEW"
        else:
            decision = "APPROVE"
        reasons = reason_codes(self.model, row, self.meta["reason_reference"], score) if decision != "APPROVE" else []
        out = ScoreOut(
            trans_num=tx.trans_num, score=round(score, 6), decision=decision, reasons=reasons,
            model_version=self.version, latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        self._audit(tx, feats, out, score)
        return out

    def _audit(self, tx: TransactionIn, feats: dict, out: ScoreOut, raw_score: float) -> None:
        """Append-only decision log. No full card number, name, DOB or home location is written."""
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trans_num": tx.trans_num,
            "card": mask_card(tx.cc_num),
            "model_version": out.model_version,
            # Unrounded: most scores are < 1e-4, rounding would break score-drift monitoring.
            "score": raw_score,
            "decision": out.decision,
            "reasons": [r.feature for r in out.reasons],
            "latency_ms": out.latency_ms,
            "features": feats,
        }
        line = json.dumps(record)
        with self._log_lock, self.log_path.open("a") as fh:
            fh.write(line + "\n")


def create_app() -> FastAPI:
    model_dir = Path(os.environ.get("MODEL_DIR", str(config.MODEL_DIR)))
    log_path = Path(os.environ.get("PREDICTION_LOG", str(config.PREDICTION_LOG)))
    bootstrap = os.environ.get("FEATURE_BOOTSTRAP_CSV")
    api_key = os.environ.get("API_KEY")
    if not api_key:
        api_key = "dev-key"
        log.warning("API_KEY not set; using the development key. Never do this outside a laptop.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = ScoringEngine(model_dir, log_path, Path(bootstrap) if bootstrap else None)
        log.info("Loaded model %s (thresholds %s)", app.state.engine.version, app.state.engine.thresholds)
        yield

    app = FastAPI(
        title="Real-time Fraud Scoring API",
        version="1.0.0",
        description="Scores card transactions and returns APPROVE / REVIEW / DECLINE with reason codes.",
        lifespan=lifespan,
    )

    def require_api_key(key: Optional[str] = Security(api_key_header)) -> None:
        if not key or not secrets.compare_digest(key, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health", tags=["ops"])
    def health(request: Request) -> dict:
        engine = getattr(request.app.state, "engine", None)
        return {"status": "ok" if engine else "starting", "model_version": engine.version if engine else None}

    @app.get("/v1/model-info", tags=["ops"], dependencies=[Depends(require_api_key)])
    def model_info(request: Request) -> dict:
        m = request.app.state.engine.meta
        return {
            "model_version": m["model_version"],
            "trained_at": m["trained_at"],
            "algorithm": m["algorithm"],
            "features": m["features"],
            "thresholds": m["thresholds"],
            "policy": m["policy"],
            "test_metrics": m["metrics"]["candidate"]["test"],
        }

    @app.post("/v1/score", response_model=ScoreOut, tags=["scoring"], dependencies=[Depends(require_api_key)])
    def score(tx: TransactionIn, request: Request) -> ScoreOut:
        return request.app.state.engine.score(tx)

    return app


app = create_app()
