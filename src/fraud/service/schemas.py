"""API contract (request/response). FastAPI publishes it as OpenAPI at /docs."""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Union

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    trans_num: str = Field(..., min_length=1, max_length=64, examples=["T000000001"])
    trans_date_trans_time: datetime = Field(..., examples=["2026-05-01 14:03:22"])
    cc_num: str = Field(..., pattern=r"^\d{12,19}$", examples=["4000123412341234"])
    merchant: str = Field(..., min_length=1, max_length=128, examples=["merchant_grocery_pos_07"])
    category: str = Field(..., min_length=1, max_length=64, examples=["grocery_pos"])
    amt: float = Field(..., gt=0, lt=1_000_000, examples=[42.5])
    lat: float = Field(..., ge=-90, le=90, description="Cardholder home latitude")
    long: float = Field(..., ge=-180, le=180, description="Cardholder home longitude")
    merch_lat: float = Field(..., ge=-90, le=90)
    merch_long: float = Field(..., ge=-180, le=180)
    city_pop: int = Field(..., ge=0)
    dob: date

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trans_num": "T000000001",
                    "trans_date_trans_time": "2026-05-01 14:03:22",
                    "cc_num": "4000123412341234",
                    "merchant": "merchant_grocery_pos_07",
                    "category": "grocery_pos",
                    "amt": 42.5,
                    "lat": 1.3521,
                    "long": 103.8198,
                    "merch_lat": 1.36,
                    "merch_long": 103.83,
                    "city_pop": 5600000,
                    "dob": "1988-07-14",
                }
            ]
        }
    }


class Reason(BaseModel):
    feature: str
    value: Union[float, str]
    impact: float = Field(..., description="Drop in score if this feature had a typical value")
    description: str


class ScoreOut(BaseModel):
    trans_num: str
    score: float = Field(..., ge=0, le=1, description="Fraud probability")
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    reasons: List[Reason] = Field(default_factory=list, description="Top drivers (REVIEW/DECLINE only)")
    model_version: str
    latency_ms: float
