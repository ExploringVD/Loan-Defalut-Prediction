# Shortcuts for the usual commands. Run from the project folder, e.g. `make test`.
# The targets call .venv/bin/python directly, so the venv does not need to be activated first.
PY := .venv/bin/python

.PHONY: help setup clean-data leakage-check train tune test-model db api frontend test drift up down

help:            ## list the targets
	@grep -E '^[a-z-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

setup:           ## create .venv (Python 3.12) and install the pinned requirements
	test -d .venv || uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - change the passwords")

clean-data:      ## data/raw/credit_risk_dataset.csv -> data/processed/ (needs the Kaggle file)
	$(PY) src/clean_data.py

leakage-check:   ## single-feature AUC + quick baselines -> reports/leakage_check.md
	$(PY) src/leakage_check.py

train:           ## compare 4 models (CV + validation, MLflow) -> reports/model_comparison.*
	$(PY) src/train.py

tune:            ## Optuna + calibration + the ONE test evaluation -> models/, reports/final_model.md
	$(PY) src/tune.py

test-model:      ## test ROC-AUC of LR, RF and the saved XGBoost model, HCL target, decision bands
	$(PY) src/test_model.py

db:              ## start only PostgreSQL in Docker
	docker compose up -d db

api: db          ## local API on http://localhost:8000 (migrations + users first)
	$(PY) -m alembic upgrade head
	$(PY) -m app.seed
	$(PY) -m uvicorn app.main:app --reload

frontend:        ## local Streamlit app on http://localhost:8501 (the API must be running)
	$(PY) -m streamlit run frontend/app.py

test:            ## run all tests (SQLite, no Docker needed)
	$(PY) -m pytest -q

drift:           ## drift check of the simulated batches -> reports/drift_summary.md, reports/drift/
	$(PY) src/drift.py

up:              ## build and start db + api + frontend in Docker
	docker compose up --build -d
	@echo "App: http://localhost:8501   API docs: http://localhost:8000/docs"

down:            ## stop the Docker services (database data is kept)
	docker compose down
