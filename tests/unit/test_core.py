"""Unit tests for security, metrics, thresholds and drift helpers."""

import numpy as np
import pandas as pd
import pytest

from fraud.monitoring.psi import build_reference, compare, psi, status
from fraud.security import mask_card
from fraud.training.evaluate import choose_thresholds, decide, evaluate


@pytest.mark.parametrize(
    "raw, masked",
    [("4000123412345678", "************5678"), ("4000-1234-1234-5678", "************5678"), ("123", "***")],
)
def test_mask_card_keeps_only_last_four(raw, masked):
    assert mask_card(raw) == masked


def test_thresholds_respect_alert_budget_and_order():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.01).astype(int)
    scores = np.clip(rng.normal(0.1, 0.05, y.size) + y * rng.uniform(0.2, 0.8, y.size), 0, 1)
    thr = choose_thresholds(y, scores, alert_budget=0.02, decline_min_precision=0.9)
    assert thr["decline"] >= thr["review"]
    alert_rate = (decide(scores, thr) != "APPROVE").mean()
    assert alert_rate == pytest.approx(0.02, abs=0.002)


def test_evaluate_perfect_model():
    y = np.array([0] * 98 + [1] * 2)
    scores = y.astype(float)
    m = evaluate(y, scores, {"review": 0.5, "decline": 0.9}, amt=np.ones(100))
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["recall_at_review"] == 1.0
    assert m["precision_at_review"] == 1.0
    assert m["fraud_amount_caught_share"] == 1.0


def test_psi_identical_is_zero_and_shift_is_alert():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"amt": rng.lognormal(3, 0.5, 50_000), "category": rng.choice(["a", "b"], 50_000)})
    reference = build_reference(ref, ["amt"], ["category"])

    same = compare(reference, ref)
    assert all(r["status"] == "OK" for r in same)

    shifted = pd.DataFrame({"amt": ref.amt * 3, "category": ["c"] * len(ref)})
    result = {r["feature"]: r for r in compare(reference, shifted)}
    assert result["amt"]["status"] == "ALERT"
    assert result["category"]["status"] == "ALERT"  # unseen category is counted, not ignored


def test_psi_bands():
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert status(0.05) == "OK" and status(0.15) == "WARN" and status(0.25) == "ALERT"
