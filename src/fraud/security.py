"""Data-protection helpers (PDPA / MAS TRM: no full card numbers in logs)."""

from __future__ import annotations


def mask_card(cc_num: str) -> str:
    """Keep only the last 4 digits: 4123456789012345 -> ************2345."""
    digits = "".join(ch for ch in str(cc_num) if ch.isdigit())
    if len(digits) <= 4:
        return "*" * len(digits)
    return "*" * (len(digits) - 4) + digits[-4:]
