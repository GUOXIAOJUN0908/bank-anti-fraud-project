"""Single source of truth for the model's feature contract."""

CATEGORICAL_FEATURES = ["category"]

NUMERIC_FEATURES = [
    "amt",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "age",
    "city_pop",
    "distance_km",
    "txn_count_1h",
    "txn_count_24h",
    "amt_sum_24h",
    "amt_ratio_to_avg_30d",
    "secs_since_last_txn",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LABEL = "is_fraud"

# Calendar features are excluded from drift checks: a monitoring window shorter
# than a week always "drifts" on day_of_week and would cause false alerts.
DRIFT_EXCLUDE = ["day_of_week"]

# Plain-English text used for reason codes shown to fraud analysts.
DESCRIPTIONS = {
    "category": "Merchant category",
    "amt": "Transaction amount",
    "hour_of_day": "Hour of day the transaction happened",
    "day_of_week": "Day of week",
    "is_night": "Transaction made late at night (22:00-06:00)",
    "age": "Cardholder age",
    "city_pop": "Population of cardholder's home city",
    "distance_km": "Distance between cardholder home and merchant (km)",
    "txn_count_1h": "Number of transactions on this card in the last hour",
    "txn_count_24h": "Number of transactions on this card in the last 24 hours",
    "amt_sum_24h": "Total amount spent on this card in the last 24 hours",
    "amt_ratio_to_avg_30d": "Amount compared with the card's 30-day average",
    "secs_since_last_txn": "Seconds since the card's previous transaction",
}
