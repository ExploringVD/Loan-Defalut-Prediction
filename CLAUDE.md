# CLAUDE.md - Loan Default Prediction (HCL training project)

Read this before every task. It is the shared context for all steps in `prompts/PROMPTS.md`.

## What the project is
An end-to-end credit-risk system: predict whether a loan will default, explain every prediction with SHAP,
serve it in real time (FastAPI backend + Streamlit frontend), log everything to PostgreSQL, and monitor data drift.
Synopsis targets: ROC-AUC >= 0.85 and a 30% NPA reduction (simulated). See "Honest metrics" below.

## Tech stack (everything is Python - trainer's requirement)
- Python 3.12 in `.venv` (managed with `uv`). Install with `uv pip install -r requirements.txt`.
- ML: pandas, scikit-learn, XGBoost, imbalanced-learn, Optuna, SHAP, joblib
- Tracking: MLflow, local only (`sqlite:///mlflow.db`, artifacts in `mlruns/`) - no MLflow server needed
- Drift: Evidently + APScheduler
- Backend: FastAPI, Uvicorn, Pydantic, SQLAlchemy, Alembic, PyJWT, passlib/bcrypt
- Database: PostgreSQL 16 in Docker (SQLite allowed for tests only)
- Frontend: Streamlit + Plotly
- Tests: pytest + httpx
- Packaging: Docker Compose, GitHub Actions

## Machine constraints (important)
- MacBook Air, Apple Silicon, **8 GB RAM**. Keep it light:
  - use `n_jobs=2` (not -1) for training; Optuna <= 50 trials unless asked
  - Docker runs only what is needed (Postgres; API/frontend containers only in the packaging step)
  - no MinIO, no Kafka, no Kubernetes; Prometheus/Grafana are optional
- The user runs things from the VS Code terminal. Always activate the venv first: `source .venv/bin/activate`.

## Folder layout
```
data/raw/            loan_train.csv (27,003 x 47, has loan_status), loan_test.csv (11,574, no labels), loan_sample_submission.csv
data/processed/      train_clean.parquet/.csv, test_clean.parquet/.csv, cleaning_report.json
src/clean_data.py    cleaning (DONE) - run: python src/clean_data.py
src/load_data.py     get_splits() -> 70/15/15 stratified train/val/test, NUMERIC_FEATURES, CATEGORICAL_FEATURES (DONE)
notebooks/01_eda.ipynb   EDA (DONE)
reports/             eda_summary.md, figures/*.png (DONE); later: model results, business simulation, drift reports
models/              saved model pipelines + metadata (to create)
app/                 FastAPI backend (to create)
frontend/            Streamlit app (to create)
tests/               pytest tests (to create)
docs/                final report, presentation outline, viva prep (to create)
prompts/PROMPTS.md   step-by-step prompts
```

## Dataset facts (Lending Club loans, June 2007 - Dec 2011)
- Target `loan_status`: 1 = defaulted (14.7%), 0 = repaid -> imbalanced. Use ROC-AUC, PR-AUC, recall, not accuracy.
- 27 model features: 22 numeric + 5 categorical, defined in `src/load_data.py`. Always load data with `get_splits()`.
- `issue_date` is kept for drift demos; it is NOT a model feature. `id` is not a feature.
- Remaining missing values (emp_length, mths_since_last_delinq, revol_util, pub_rec_bankruptcies) must be imputed
  INSIDE the sklearn pipeline, fitted on training data only.
- Highly correlated pairs: int_rate~sub_grade_num (0.96), loan_amnt~installment (0.93),
  loan_to_income~installment_to_income (0.93), pub_rec~pub_rec_bankruptcies (0.84).
- Strongest single features: int_rate, sub_grade_num, term_months, loan_to_income, revol_util.

## Rules that must never be broken
1. **No data leakage.** Never add back payment/recovery columns (total_pymnt, recoveries, last_pymnt_*, out_prncp,
   funded_amnt, ...). They were removed on purpose: with them AUC is a fake 0.998.
2. Fit every transformer (imputer, scaler, encoder, SMOTE) on the training fold only. SMOTE only inside an imblearn Pipeline.
3. The test split from `get_splits()` is used ONCE, for the final numbers. Tune on CV / validation.
4. `random_state=42` everywhere.
5. The same saved pipeline (preprocessing + model) is used by training, the API and drift checks - no duplicate preprocessing code.

## Honest metrics
A quick baseline gives ~0.70 validation AUC; the realistic ceiling on this data without leakage is ~0.72-0.76.
Do NOT chase the 0.85 target with leakage or by evaluating on training data. Report the real number and explain it.
The same applies to the 30% NPA reduction: compute it by simulation and report whatever it really is.

## Conventions
- Code, comments, file names in English; keep code simple and readable (this is a student project that must be explained in a viva).
- Put reusable logic in `src/` (ML) or `app/` (backend); notebooks only for exploration and showing results.
- Every script runs from the project root, e.g. `python src/train.py`.
- Save charts to `reports/figures/` (PNG, clear titles, labelled axes).
- Add/extend tests in `tests/` for new logic; run `pytest -q` before finishing a step.
- At the end of each step: update README.md (what was added + how to run), list files created/changed,
  show the key results, and explain in simple Hinglish what was done so the user can explain it to the teacher.
- Suggest a git commit message at the end of each step.
