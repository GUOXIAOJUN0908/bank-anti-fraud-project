"""Population Stability Index (PSI), the standard drift measure in bank model risk.

PSI < 0.10 stable | 0.10 - 0.20 moderate shift, watch | > 0.20 significant shift, act.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
PSI_WARN = 0.10
PSI_ALERT = 0.20


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "ALERT" if value > PSI_ALERT else "WARN" if value > PSI_WARN else "OK"


def numeric_profile(values: np.ndarray, n_bins: int = 10) -> dict:
    """Decile bins from the reference data; returns inner edges + share per bin."""
    values = np.asarray(values, float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1))[1:-1])
    return {"edges": edges.tolist(), "props": _numeric_props(values, edges).tolist()}


def _numeric_props(values: np.ndarray, edges) -> np.ndarray:
    counts = np.bincount(np.digitize(values, edges, right=True), minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def categorical_profile(values: pd.Series) -> dict:
    return {"props": values.value_counts(normalize=True).to_dict()}


def build_reference(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    ref = {f: {"type": "numeric", **numeric_profile(df[f].to_numpy())} for f in numeric}
    ref.update({f: {"type": "categorical", **categorical_profile(df[f])} for f in categorical})
    return ref


def compare(reference: dict, df: pd.DataFrame) -> list[dict]:
    """PSI of every profiled column in `df` against the reference."""
    rows = []
    for col, ref in reference.items():
        if col not in df:
            continue
        if ref["type"] == "numeric":
            actual = _numeric_props(df[col].to_numpy(float), np.asarray(ref["edges"]))
            value = psi(np.asarray(ref["props"]), actual)
        else:
            cats = list(ref["props"])
            shares = df[col].value_counts(normalize=True)
            expected = [ref["props"][c] for c in cats] + [0.0]
            actual = [float(shares.get(c, 0.0)) for c in cats] + [float(shares[~shares.index.isin(cats)].sum())]
            value = psi(np.asarray(expected), np.asarray(actual))
        rows.append({"feature": col, "psi": round(value, 4), "status": status(value)})
    return sorted(rows, key=lambda r: r["psi"], reverse=True)
