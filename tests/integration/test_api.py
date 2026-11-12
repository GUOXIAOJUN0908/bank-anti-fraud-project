"""System integration tests (SIT) for the scoring API. IDs match docs/test_plan.md."""

import json

import pytest

from tests.conftest import make_txn


def _log(api) -> list[dict]:
    return [json.loads(line) for line in api.log_path.read_text().splitlines()]


def test_sit01_health_reports_loaded_model(api):
    body = api.client.get("/health").json()
    assert body == {"status": "ok", "model_version": api.meta["model_version"]}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key"}])
def test_sit02_rejects_missing_or_wrong_api_key(api, headers):
    client = api.client
    client.headers.pop("X-API-Key", None)
    resp = client.post("/v1/score", json=make_txn(), headers=headers)
    assert resp.status_code == 401
    assert client.get("/v1/model-info", headers=headers).status_code == 401


@pytest.mark.parametrize(
    "override",
    [{"cc_num": "4000-abc"}, {"amt": -10}, {"amt": 0}, {"merch_lat": 123.0}, {"trans_date_trans_time": "yesterday"}],
)
def test_sit03_invalid_payload_returns_422(api, override):
    assert api.client.post("/v1/score", json=make_txn(**override)).status_code == 422


def test_sit03_missing_field_returns_422(api):
    txn = make_txn()
    del txn["category"]
    assert api.client.post("/v1/score", json=txn).status_code == 422


def test_sit04_valid_transaction_returns_contract(api):
    resp = api.client.post("/v1/score", json=make_txn())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"trans_num", "score", "decision", "reasons", "model_version", "latency_ms"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert body["model_version"] == api.meta["model_version"]
    assert body["latency_ms"] > 0


def test_sit05_audit_log_masks_card_number(api):
    api.client.post("/v1/score", json=make_txn(cc_num="4000999988887777"))
    raw = api.log_path.read_text()
    assert "4000999988887777" not in raw
    assert "1985-06-15" not in raw  # date of birth is not logged either
    record = _log(api)[-1]
    assert record["card"] == "************7777"
    assert set(record["features"]) == set(api.meta["features"])


def test_sit06_card_history_drives_velocity_features(api):
    for i in range(5):
        api.client.post("/v1/score", json=make_txn(
            trans_num=f"T-VEL-{i}", cc_num="4000555566667777", trans_date_trans_time=f"2026-03-02 12:0{i}:00"))
    counts = [r["features"]["txn_count_1h"] for r in _log(api)]
    assert counts == [1, 2, 3, 4, 5]


def test_sit07_timezone_aware_timestamps_are_normalised_to_utc(api):
    api.client.post("/v1/score", json=make_txn(trans_date_trans_time="2026-03-02T22:00:00+08:00"))
    assert _log(api)[-1]["features"]["hour_of_day"] == 14


def test_sit08_model_info_exposes_thresholds(api):
    body = api.client.get("/v1/model-info").json()
    assert body["thresholds"] == api.meta["thresholds"]
    assert "pr_auc" in body["test_metrics"]
