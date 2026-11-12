"""User acceptance tests written as business scenarios. IDs match docs/test_plan.md."""

from datetime import datetime, timedelta

import pytest

from tests.conftest import make_txn

pytestmark = pytest.mark.uat
CARD = "4000777788889999"


def _history(api, card=CARD, days=5):
    """Give the card a few normal days of history, as a real customer would have."""
    start = datetime(2026, 2, 20, 12, 0)
    for d in range(days):
        ts = start + timedelta(days=d, hours=d % 3)
        api.client.post("/v1/score", json=make_txn(
            trans_num=f"T-HIST-{card[-4:]}-{d}", cc_num=card, trans_date_trans_time=ts.isoformat(sep=" "),
            amt=35 + 5 * d, category=["grocery_pos", "food_dining", "gas_transport"][d % 3]))


def test_uat01_everyday_purchase_near_home_is_approved(api):
    _history(api)
    body = api.client.post("/v1/score", json=make_txn(
        trans_num="T-UAT01", cc_num=CARD, trans_date_trans_time="2026-02-26 13:10:00")).json()
    assert body["decision"] == "APPROVE"
    assert body["reasons"] == []


def test_uat02_overseas_online_spree_at_night_is_stopped(api):
    _history(api)
    decisions = []
    for i in range(6):
        body = api.client.post("/v1/score", json=make_txn(
            trans_num=f"T-UAT02-{i}", cc_num=CARD, trans_date_trans_time=f"2026-02-27 01:{10 * i:02d}:00",
            category="shopping_net", amt=420 + 37 * i, merch_lat=51.5072, merch_long=-0.1276)).json()
        decisions.append(body["decision"])
    assert decisions[-1] in {"REVIEW", "DECLINE"}
    assert decisions.count("APPROVE") <= 2, decisions


def test_uat03_every_alert_explains_itself_in_plain_english(api):
    _history(api)
    body = api.client.post("/v1/score", json=make_txn(
        trans_num="T-UAT03", cc_num=CARD, trans_date_trans_time="2026-02-27 02:00:00",
        category="shopping_net", amt=1800, merch_lat=51.5072, merch_long=-0.1276)).json()
    assert body["decision"] in {"REVIEW", "DECLINE"}
    assert 1 <= len(body["reasons"]) <= 3
    for reason in body["reasons"]:
        assert reason["impact"] > 0
        assert len(reason["description"]) > 10
