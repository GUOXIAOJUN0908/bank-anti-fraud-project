"""Metrics and threshold selection.

Accuracy is meaningless at a ~0.4% fraud rate (predicting "never fraud" scores
99.6%), so we report PR-AUC and recall at the analysts' review capacity.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def choose_thresholds(y: np.ndarray, scores: np.ndarray, alert_budget: float, decline_min_precision: float) -> dict:
    """Pick decision thresholds on the VALIDATION set.

    review  : top `alert_budget` share of transactions go to a fraud analyst.
    decline : lowest score where precision >= `decline_min_precision`
              (auto-decline only when we are very sure; never below review).
    """
    review = float(np.quantile(scores, 1 - alert_budget))
    precision, _, thr = precision_recall_curve(y, scores)
    ok = np.where(precision[:-1] >= decline_min_precision)[0]
    decline = float(thr[ok[0]]) if len(ok) else 1.01  # 1.01 = never auto-decline
    return {"review": review, "decline": max(decline, review)}


def decide(scores: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(
        scores >= thresholds["decline"], "DECLINE", np.where(scores >= thresholds["review"], "REVIEW", "APPROVE")
    )


def evaluate(y: np.ndarray, scores: np.ndarray, thresholds: dict | None = None, amt: np.ndarray | None = None) -> dict:
    y = np.asarray(y)
    out = {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan"),
    }
    if thresholds:
        dec = decide(scores, thresholds)
        flagged = dec != "APPROVE"
        tp = int((flagged & (y == 1)).sum())
        out.update(
            {
                "alert_rate": float(flagged.mean()),
                "recall_at_review": tp / max(int(y.sum()), 1),
                "precision_at_review": tp / max(int(flagged.sum()), 1),
                "decline_count": int((dec == "DECLINE").sum()),
                "decline_precision": float(y[dec == "DECLINE"].mean()) if (dec == "DECLINE").any() else None,
            }
        )
        if amt is not None:
            fraud_amt = float(amt[y == 1].sum())
            out["fraud_amount_caught_share"] = float(amt[(y == 1) & flagged].sum()) / fraud_amt if fraud_amt else None
    return out
