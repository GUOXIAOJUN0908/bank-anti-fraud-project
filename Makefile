PY ?= .venv/bin/python
LIMIT ?= 5000
export PYTHONPATH := src

.PHONY: setup data features train serve simulate drift drift-batch rca test lint all docker clean

setup:            ## create .venv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

data:             ## generate synthetic transactions (history + a later "new" period)
	$(PY) -m fraud.data.generate --customers 2000 --days 120 --new-days 30

features:         ## load raw data into the warehouse and build point-in-time features
	$(PY) -m fraud.features.build

train:            ## train baseline + candidate, choose thresholds, run the quality gate
	$(PY) -m fraud.training.train

serve:            ## run the scoring API on :8000 (card history hydrated from the warehouse extract)
	FEATURE_BOOTSTRAP_CSV=data/raw/transactions.csv $(PY) -m uvicorn fraud.service.app:app --port 8000

simulate:         ## replay the new period against the running API (make simulate LIMIT=1200)
	$(PY) -m fraud.simulator.replay --limit $(LIMIT)

drift:            ## drift + performance report from the live prediction log
	$(PY) -m fraud.monitoring.drift --from-log logs/predictions.jsonl --labels data/raw/transactions_new.csv \
		--out reports/drift_report_live.md

drift-batch:      ## offline drift + performance check on the new warehouse batch (exit 2 on alert)
	$(PY) -m fraud.monitoring.drift --batch transactions_new --fail-on-alert --out reports/drift_report_batch.md

rca:              ## RCA-001 corrective action: champion vs. retrained challenger
	$(PY) -m fraud.training.retrain_experiment

test:             ## unit + SIT + UAT tests
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

all: data features train test

docker:           ## build the API image (run `make train` first so models/ exists)
	docker build -t fraud-api:local .

clean:
	rm -rf data/warehouse.duckdb logs reports models/model.joblib models/metadata.json
