# Test Plan: Fraud Scoring Service

| | |
|---|---|
| Document ID | TP-FRAUD-001 |
| Run all automated tests | `make test` (26 tests, ~5 s); `make lint` |
| CI | every push / pull request (`.github/workflows/ci.yml`) |

## 1. Test levels

| Level | Purpose | Location | Automated |
|---|---|---|---|
| Unit | Functions behave correctly in isolation | `tests/unit/` | Yes |
| Data / feature parity | Batch SQL and real-time features are identical | `tests/unit/test_feature_consistency.py` | Yes |
| Model quality gate | Candidate beats baseline and PR-AUC floor | `src/fraud/training/train.py` (CI `train` job) | Yes |
| SIT (system integration) | API contract, security, logging, state | `tests/integration/test_api.py` | Yes |
| UAT (user acceptance) | Business scenarios signed off by the fraud operations team | `tests/uat/` | Yes (scenarios agreed with business) |
| Performance | Latency and throughput under load | `tests/load/locustfile.py`, `make simulate` | Semi (run before release) |
| Container smoke test | Built image starts, scores, rejects unauthenticated calls | CI `image` job | Yes |

## 2. Entry and exit criteria

- **Entry:** code merged to a feature branch, lint clean, model artefact produced by `make train`.
- **Exit (release):** 100% of SIT and UAT cases pass; quality gate passed; PT-01 met; no open Severity 1-2 defects;
  model card updated.

## 3. Test cases

### Unit and data

| ID | Case | Expected | Test |
|---|---|---|---|
| UT-01 | SQL features vs online features on every training transaction | Identical to 1e-9 | `test_online_features_match_sql` |
| UT-02 | A card's first transaction | No history leaks in (counts = 1, ratio = 1, gap = cap) | `test_no_lookahead_first_transaction_of_card_has_no_history` |
| UT-03 | Card masking | Only last 4 digits remain | `test_mask_card_keeps_only_last_four` |
| UT-04 | Threshold selection | Alert rate = budget; decline >= review | `test_thresholds_respect_alert_budget_and_order` |
| UT-05 | Metrics on a perfect model | PR-AUC = recall = precision = 1 | `test_evaluate_perfect_model` |
| UT-06 | PSI | 0 for same data; ALERT for shifted data and unseen categories | `test_psi_identical_is_zero_and_shift_is_alert`, `test_psi_bands` |

### SIT

| ID | Case | Expected | Test |
|---|---|---|---|
| SIT-01 | GET /health after start-up | 200, status ok, loaded model version | `test_sit01_*` |
| SIT-02 | Missing or wrong API key | 401 on /v1/score and /v1/model-info | `test_sit02_*` |
| SIT-03 | Invalid PAN, amount <= 0, bad coordinates, bad timestamp, missing field | 422 with field detail | `test_sit03_*` |
| SIT-04 | Valid transaction | 200; response matches contract; decision in allowed set | `test_sit04_*` |
| SIT-05 | Audit log | PAN masked; DOB absent; all features recorded | `test_sit05_*` |
| SIT-06 | Five transactions on one card in five minutes | txn_count_1h = 1..5 (state kept across calls) | `test_sit06_*` |
| SIT-07 | Timestamp with +08:00 offset | Converted to UTC before features | `test_sit07_*` |
| SIT-08 | GET /v1/model-info | Returns live thresholds and metrics | `test_sit08_*` |

### UAT (business language)

| ID | Scenario | Acceptance criterion | Test |
|---|---|---|---|
| UAT-01 | Regular customer buys groceries near home at lunchtime | APPROVE, no reasons shown | `test_uat01_*` |
| UAT-02 | Six large online purchases from London between 01:00 and 01:50 on a Singapore card | Spree is stopped: last one REVIEW/DECLINE, at most 2 approved | `test_uat02_*` |
| UAT-03 | Any REVIEW or DECLINE | 1-3 reasons, each with plain-English description and positive impact | `test_uat03_*` |

### Performance

| ID | Case | Target | Result (reference run, laptop, 1 worker) |
|---|---|---|---|
| PT-01 | 6,000 sequential replays (`make simulate`) | p95 < 100 ms, 0 errors | server p95 12 ms, client p95 13 ms, 0 errors, ~112 req/s |
| PT-02 | 50 concurrent users for 2 min (Locust) | p95 < 100 ms, error rate < 0.1% | Run before release; record result here |

## 4. Defect severity

| Severity | Definition | Example |
|---|---|---|
| 1 | Wrong decisions at scale, data leak, outage | PAN written to logs |
| 2 | Contract broken for some inputs | 500 on unseen category |
| 3 | Degraded but workable | Reason text unclear |
| 4 | Cosmetic | Typo in docs |
