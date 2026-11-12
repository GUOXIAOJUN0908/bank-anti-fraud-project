"""Reason codes: why did this transaction get a high score?

Method (model-agnostic, no extra dependency): for each feature, swap the value
for a "typical" reference value (training median / most common category) and
re-score. The bigger the drop in score, the more that feature pushed the score
up. All variants are scored in one batch, so it costs a single model call.
"""

from __future__ import annotations

import pandas as pd

from fraud.features.schema import DESCRIPTIONS, FEATURES, NUMERIC_FEATURES


def reason_codes(model, row: pd.DataFrame, reference: dict, base_score: float, top_k: int = 3) -> list[dict]:
    variants = pd.concat([row] * len(FEATURES), ignore_index=True)
    variants = variants.astype({f: float for f in NUMERIC_FEATURES})
    for i, feat in enumerate(FEATURES):
        variants.at[i, feat] = reference[feat]
    scores = model.predict_proba(variants[FEATURES])[:, 1]
    impacts = sorted(
        ((base_score - s, feat) for s, feat in zip(scores, FEATURES)), reverse=True
    )
    reasons = []
    for impact, feat in impacts[:top_k]:
        if impact <= 0:
            break
        value = row.iloc[0][feat]
        reasons.append(
            {
                "feature": feat,
                "value": value if isinstance(value, str) else round(float(value), 2),
                "impact": round(float(impact), 4),
                "description": DESCRIPTIONS[feat],
            }
        )
    return reasons
