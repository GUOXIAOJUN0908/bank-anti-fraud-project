# Real-time Transaction Fraud Scoring

[![ci](https://github.com/GUOXIAOJUN0908/bank-aiti-fraud-project/actions/workflows/ci.yml/badge.svg)](https://github.com/GUOXIAOJUN0908/bank-aiti-fraud-project/actions/workflows/ci.yml)

An end-to-end machine-learning service that scores card transactions in real time and returns
**APPROVE / REVIEW / DECLINE** with plain-English reasons, built the way a bank would run it: tested, documented,
monitored and shipped through CI/CD.

```text
Payment switch ──REST──► FastAPI scoring service ──► decision + reason codes      (p95 ≈ 12 ms)
                              │        ▲
                              ▼        │ point-in-time features (same logic as SQL, parity-tested)
                         audit log   model + thresholds ◄── training pipeline + quality gate ◄── SQL features ◄── warehouse
                              │
                              ▼
                   drift & performance monitoring ──► runbook / RCA
```

## Results (reference run, synthetic data)

| | Logistic regression baseline | Gradient-boosted trees |
|---|---|---|
| PR-AUC (test period) | 0.412 | **0.893** |
| Fraud caught when analysts review 1% of traffic | 74.3% | **92.4%** |
| Precision of those alerts | 27.9% | 33.5% |
| Scoring latency p95 | | **~12 ms** |

**Simulated incident ([RCA-001](docs/rca/RCA-001-card-testing-drift.md)):** fraudsters switch to card testing. Recall falls
to 22% while **every drift metric stays green**. Root cause analysis plus a retrain with 10 days of new labels brings it
back to 85%.

> Data is synthetic (`src/fraud/data/generate.py`) in the schema of the public Kaggle *Credit Card Transactions Fraud
> Detection* dataset, so the real data can be dropped into `data/raw/` without code changes.

## Quick start

```bash
make setup            # Python 3.9+ virtualenv + dependencies
make all              # generate data -> build features -> train (with quality gate) -> 26 tests
make serve            # API on http://localhost:8000  (docs at /docs, dev key: dev-key)
make simulate         # in a 2nd terminal: replay 5,000 May transactions through the API
make drift            # drift + performance report from the live audit log
make drift-batch      # same check on the whole May batch in the warehouse
make rca              # RCA-001 corrective action: champion vs retrained challenger
```

The replayed May traffic deliberately contains a fraud pattern the model has never seen (card testing), so **live recall
is low on purpose**. The demo shows the service working *and* monitoring catching the failure; see RCA-001.

Score one transaction:

```bash
curl -s -X POST localhost:8000/v1/score -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
     -d @tests/fixtures/sample_txn.json
```

Docker: `make train && cp .env.example .env && docker compose up --build`.

## What is inside

| Area | Where | Highlights |
|---|---|---|
| Data preparation | `sql/features.sql`, `src/fraud/features/` | Point-in-time window features in portable SQL (DuckDB locally, PostgreSQL-compatible); identical real-time implementation proven by a parity test |
| Modelling | `src/fraud/training/` | Time-based split, baseline vs candidate, PR-AUC, thresholds from analyst capacity and decline precision, permutation importance, model-agnostic reason codes, quality gate |
| Integration | `src/fraud/service/` | FastAPI, OpenAPI contract, Pydantic validation, API-key auth, card masking, append-only audit log |
| Testing | `tests/` | Unit, feature parity, SIT, UAT in business language, Locust load test, container smoke test |
| Monitoring | `src/fraud/monitoring/` | PSI per feature and score, label-based recall tracking, caught-vs-missed analysis |
| Delivery | `Dockerfile`, `.github/workflows/ci.yml` | Lint → tests → train + gate → image + smoke test → (gated) deploy to AWS App Runner |
| Documentation | `docs/` | [Solution design](docs/solution_design.md), [data mapping](docs/data_mapping.md), [model card](docs/model_card.md), [test plan](docs/test_plan.md), [runbook](docs/runbook.md), [RCA-001](docs/rca/RCA-001-card-testing-drift.md) |

## Project layout

```text
sql/features.sql              point-in-time feature SQL
src/fraud/
  data/generate.py            synthetic transactions (Kaggle/Sparkov schema)
  features/                   build.py (warehouse), online.py (real time), schema.py (feature contract)
  training/                   train.py, evaluate.py, explain.py, retrain_experiment.py
  service/                    app.py (FastAPI), schemas.py (API contract)
  simulator/replay.py         payment-switch simulator
  monitoring/                 psi.py, drift.py
  security.py                 PAN masking
tests/                        unit/, integration/ (SIT), uat/, load/
docs/                         SDD, data mapping, model card, test plan, runbook, RCA
```

## Design choices worth discussing

- **Why not accuracy?** At 0.4% fraud, "approve everything" is 99.6% accurate. PR-AUC and recall at a fixed alert
  budget match how a fraud team actually works.
- **Why a time split?** A random split lets the model learn from the future and overstates performance.
- **Why one feature definition plus a parity test?** Training/serving skew silently breaks models in production.
- **Why monitor labels, not just PSI?** RCA-001: a real change in fraud behaviour hid inside normal-looking traffic.
- **Why HistGradientBoosting?** Same family as XGBoost/LightGBM with no native dependencies. `--model xgboost` is
  supported (`pip install xgboost`; on macOS also `brew install libomp`).
