"""Shared fixtures: build a small but complete pipeline once per test session."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fraud.data.generate import generate
from fraud.features.build import build
from fraud.training.train import train

API_KEY = "test-key"


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    ns = SimpleNamespace(raw=root / "raw", db=root / "warehouse.duckdb", models=root / "models",
                         reports=root / "reports")
    generate(customers=400, days=60, new_days=0, seed=7, out_dir=ns.raw)
    build(ns.raw, ns.db)
    ns.meta = train("hgb", gate=False, model_dir=ns.models, report_dir=ns.reports, db_path=ns.db)
    return ns


@pytest.fixture()
def api(pipeline, tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("MODEL_DIR", str(pipeline.models))
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("PREDICTION_LOG", str(log_path))
    monkeypatch.delenv("FEATURE_BOOTSTRAP_CSV", raising=False)
    from fraud.service.app import create_app

    with TestClient(create_app(), headers={"X-API-Key": API_KEY}) as client:
        yield SimpleNamespace(client=client, log_path=log_path, meta=pipeline.meta)


def make_txn(**overrides) -> dict:
    """A normal daytime grocery purchase in Singapore; override fields per scenario."""
    txn = {
        "trans_num": "T-TEST-0001",
        "trans_date_trans_time": "2026-03-02 12:15:00",
        "cc_num": "4000111122223333",
        "merchant": "merchant_grocery_pos_03",
        "category": "grocery_pos",
        "amt": 48.20,
        "lat": 1.3521,
        "long": 103.8198,
        "merch_lat": 1.3600,
        "merch_long": 103.8300,
        "city_pop": 5_600_000,
        "dob": "1985-06-15",
    }
    txn.update(overrides)
    return txn
