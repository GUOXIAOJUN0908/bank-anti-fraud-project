# Runbook: Fraud Scoring Service

## 1. Service facts

| Item | Value |
|---|---|
| Endpoints | `GET /health` (no auth), `POST /v1/score`, `GET /v1/model-info` (X-API-Key) |
| Port | 8000 |
| Config | `API_KEY`, `MODEL_DIR`, `PREDICTION_LOG`, `FEATURE_BOOTSTRAP_CSV` |
| Artefacts | `models/model.joblib`, `models/metadata.json` (thresholds, references, metrics) |
| Logs | stdout (service events), `logs/predictions.jsonl` (audit, PAN masked) |
| Upstream fallback | Payment switch times out after 150 ms and uses its rule engine |

## 2. Routine operations

**Deploy a new model**
1. Merge to `main`; CI runs tests, trains, applies the quality gate and builds the image.
2. Compare the new model with the live one on the latest labelled weeks (`make rca` shows the pattern).
3. Approve the `production` environment in GitHub; App Runner rolls out.
4. Check `/health` shows the new `model_version` and the alert rate looks normal for 1 hour.

**Roll back**
Redeploy the previous image tag (images are tagged with the git SHA). The model and thresholds are inside the image, so
a rollback restores both together.

**Rotate the API key**
Add the new key to the secret manager, deploy, switch the payment switch to the new key, then remove the old one.

## 3. Alerts and what to do

### A. `/health` failing or container restarting
1. `docker logs <container>`: look for `FileNotFoundError` (model missing) or bootstrap errors.
2. Model missing means the image was built without artefacts: rebuild after `make train` / CI `train` job.
3. Payments continue on rules during the outage; record the incident start and end time.

### B. p95 latency > 100 ms
1. Check CPU on the host; one worker serves ~110 req/s sequentially.
2. Check for a few cards with huge history (bot traffic): `txn_count_24h` values in the audit log.
3. Short term: add replicas (requires shared card history, see SDD section 6). Long term: Redis feature store.

### C. Alert rate outside 0.5%-2%
1. Confirm the live `model_version` and thresholds (`/v1/model-info`).
2. Run `make drift` and look at the score PSI and the top-PSI features.
3. Check upstream changes: new categories, currency or timezone changes in the feed.

### D. Drift or performance alert
1. Run `make drift` (live log) or `make drift-batch` (warehouse).
2. If PSI is high but recall is fine: data shift only. Note it and keep watching.
3. If recall dropped (> 10 points): open an RCA using the "caught vs missed" table, see
   [RCA-001](rca/RCA-001-card-testing-drift.md) as the template.
4. Retrain a challenger with recent labels (`make rca` pattern), compare, release through CI.

## 4. Useful commands

```bash
curl -s localhost:8000/health
curl -s -H "X-API-Key: $API_KEY" localhost:8000/v1/model-info | python -m json.tool
tail -n 5 logs/predictions.jsonl | python -m json.tool --json-lines
make drift
```
