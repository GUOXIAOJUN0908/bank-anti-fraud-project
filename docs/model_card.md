# Model Card: Card Transaction Fraud Scorer

| | |
|---|---|
| Model | `HistGradientBoostingClassifier` (scikit-learn) in a pipeline with categorical encoding |
| Reference version | `hgb-20261112T081747Z` (numbers below). The latest run is always in `reports/training_report.md` |
| Owner | AI/ML Solutions (portfolio project) |
| Intended use | Real-time risk score for card authorisations: route the riskiest 1% to fraud analysts, auto-decline only very-high-confidence cases |
| Not intended for | Credit decisions, AML/sanctions screening, any decision about a customer without a human review path |

> **Data disclaimer.** All results are on synthetic data produced by `src/fraud/data/generate.py`. They show that
> the pipeline works end to end. They do **not** predict performance on real bank data. Re-run on the public Kaggle
> dataset (same schema) or on internal data before quoting numbers externally.

## 1. Training data

| Split | Period | Rows | Fraud rate |
|---|---|---|---|
| Train | 2026-01-01 to 2026-03-25 | 228,414 | 0.373% |
| Validation (threshold selection) | 2026-03-25 to 2026-04-12 | 48,946 | 0.380% |
| Test (untouched) | 2026-04-12 to 2026-04-30 | 48,946 | 0.349% |

Fraud patterns present in training: card-not-present sprees (large online amounts, often at night, often overseas) and
skimmed cards used in shops in another city.

## 2. Features

13 features, all point-in-time: see the [data mapping](data_mapping.md). Gender, name, address and merchant id are
deliberately **not** used.

## 3. Performance (test period)

| Metric | Baseline (logistic regression) | This model |
|---|---|---|
| PR-AUC | 0.412 | **0.893** |
| ROC-AUC | 0.988 | 0.998 |
| Alert rate (REVIEW + DECLINE) | 0.93% | 0.96% |
| Recall at review threshold | 74.3% | **92.4%** |
| Precision at review threshold | 27.9% | 33.5% |
| Share of fraud amount caught | 84.1% | 95.8% |
| Auto-declines / their precision | 0 / - | 162 / 87.0% |

Decision thresholds (from validation): REVIEW at score >= 0.0027, DECLINE at score >= 0.2437.

Most important features (drop in validation PR-AUC when shuffled): amount 0.34, distance from home 0.30,
merchant category 0.25, 24-hour spend 0.16, time since last transaction 0.16.

## 4. Fairness check (test period, review threshold)

Gender is not a model input. Outcomes are still checked by group, because other features can act as proxies.

| Group | Transactions | Frauds | Recall | False-alert rate (genuine customers flagged) |
|---|---|---|---|---|
| Age 18-30 | 10,364 | 29 | 100.0% | 0.50% |
| Age 31-45 | 12,035 | 21 | 100.0% | 0.70% |
| Age 46-60 | 11,881 | 41 | 90.2% | 0.68% |
| Age 61+ | 14,666 | 80 | 88.8% | 0.67% |
| Female | 24,239 | 70 | 91.4% | 0.68% |
| Male | 24,707 | 101 | 93.1% | 0.61% |

False-alert rates are similar across groups (0.50-0.70%), so no group of genuine customers is disproportionately
inconvenienced. Recall per group rests on 21-101 frauds, so differences of a few points are within noise; re-check on
larger real data.

## 5. Limitations and known failure modes

- **New fraud tactics.** Measured in [RCA-001](rca/RCA-001-card-testing-drift.md): a switch to low-value card testing
  dropped recall to ~22% while feature PSI stayed normal. Mitigated by label-based monitoring and retraining.
- **Cold start.** A card's first transaction has no history (velocity = 1, ratio = 1); risk relies on amount,
  category and distance only.
- **Label delay.** Chargebacks arrive weeks later, so performance monitoring lags; PSI is the early (but incomplete)
  signal.
- **Probabilities are not calibrated.** Scores rank risk well but are not true probabilities; use the thresholds, not
  the raw score, for decisions.

## 6. Monitoring and retraining

| Signal | Frequency | Threshold | Action |
|---|---|---|---|
| Feature and score PSI | Daily | > 0.20 ALERT | Investigate per [runbook](runbook.md) |
| Recall at review threshold (with chargeback labels) | Weekly | Drop > 10 points vs test | Open an RCA, retrain challenger |
| Alert rate | Daily | Outside 0.5%-2% | Check thresholds and upstream data |
| p95 latency | Continuous | > 100 ms | Scale out / investigate |

Retraining: monthly, or when an alert fires. A new model must pass the CI quality gate (beat the baseline and
PR-AUC >= 0.60) and be compared with the live model on the most recent labelled weeks before release.
