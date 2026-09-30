# Claude Code prompts - one per step

How to use:
1. Open the project folder in VS Code and start Claude Code in it (it reads `CLAUDE.md` automatically).
2. Copy ONE prompt at a time, in order. Let it finish, check the result, then commit with git.
3. If something breaks, paste the error back to Claude Code and ask it to fix it.

Done already: cleaning (`src/clean_data.py`), loader (`src/load_data.py`), EDA (`notebooks/01_eda.ipynb`).

---

## Step 1 - Environment and project setup

```
Read CLAUDE.md first. Set up the project environment on this Mac:
1. Create a Python 3.12 virtual environment with uv in .venv and install requirements.txt.
   Add bcrypt and python-dotenv to requirements.txt if they are missing.
2. Verify the install: import every package in requirements.txt and print its version. Fix any failures.
3. Run python src/clean_data.py and python src/load_data.py and confirm the split sizes are
   18,901 / 4,051 / 4,051 with a 14.7% default rate in each.
4. Create a .gitignore (.venv, __pycache__, .DS_Store, mlruns/, mlflow.db, .env, *.log, reports/drift/*.html).
5. Create .env.example with DATABASE_URL, JWT_SECRET, JWT_EXPIRE_MINUTES, ADMIN_USERNAME, ADMIN_PASSWORD,
   MODEL_PATH, API_URL, and copy it to .env with safe local values.
6. Create docker-compose.yml with ONLY a postgres:16 service (named db), a named volume, port 5432,
   a healthcheck, and a memory limit of 512m. Start it and confirm I can connect.
7. Initialise git and make the first commit if the folder is not already a repo.
Finish with a short checklist of what works.
```

---

## Step 2 - Preprocessing pipeline

```
Read CLAUDE.md first. Build the preprocessing in src/features.py using scikit-learn ColumnTransformer + Pipeline:
- build_preprocessor(kind) where kind is "linear" or "tree".
- "linear": median imputation with missing-value indicators, log1p on skewed columns (annual_inc, revol_bal,
  loan_to_income, installment_to_income, revol_bal_to_income), StandardScaler, OneHotEncoder(handle_unknown="ignore",
  min_frequency=20) for categoricals. Drop one column of each highly correlated pair listed in CLAUDE.md
  (keep int_rate, loan_amnt, loan_to_income, pub_rec) - implement this as a configurable column list, not hard-coded deletes.
- "tree": median imputation (plus indicator for mths_since_last_delinq), no scaling, OneHotEncoder for categoricals.
- get_feature_names(preprocessor) that returns readable output feature names (needed later for SHAP).
Use NUMERIC_FEATURES / CATEGORICAL_FEATURES from src/load_data.py. Nothing may be fitted outside a Pipeline.
Add tests/test_features.py: output has no NaNs, same number of columns for train/val/test, unseen categories
do not crash, fitting only uses the data it is given. Run pytest.
```

---

## Step 3 - Train and compare three models

```
Read CLAUDE.md first. Create src/train.py that trains and compares three models on the training split
from get_splits(), each as a full Pipeline (preprocessor from src/features.py + model):
1. LogisticRegression (linear preprocessor, class_weight="balanced", max_iter=2000)
2. RandomForestClassifier (tree preprocessor, class_weight="balanced_subsample", n_estimators=300, n_jobs=2)
3. XGBClassifier (tree preprocessor, scale_pos_weight = negatives/positives, n_jobs=2, eval_metric="auc")
Also train one extra variant: XGBoost with SMOTE inside an imblearn Pipeline instead of scale_pos_weight.
For each model:
- 5-fold StratifiedKFold cross-validation ROC-AUC on the training split (mean and std)
- fit on the full training split, evaluate on the validation split: ROC-AUC, PR-AUC, KS statistic,
  and precision / recall / F1 at threshold 0.5
- log params, metrics, the fitted pipeline and plots to MLflow (tracking URI sqlite:///mlflow.db,
  experiment "loan-default")
Save reports/model_comparison.csv and a markdown table reports/model_comparison.md, plus
reports/figures/12_roc_curves_validation.png (all models on one chart) and 13_pr_curves_validation.png.
Print the comparison table and recommend which model to tune, with reasons.
Tell me how to open the MLflow UI (mlflow ui --backend-store-uri sqlite:///mlflow.db).
```

---

## Step 4 - Hyperparameter tuning and final evaluation

```
Read CLAUDE.md first. Create src/tune.py:
1. Tune the best model from reports/model_comparison.csv (expected XGBoost) with Optuna, 50 trials,
   objective = mean 5-fold stratified CV ROC-AUC on the training split. Search learning_rate, max_depth,
   min_child_weight, subsample, colsample_bytree, n_estimators, reg_lambda, gamma. Use n_jobs=2 and
   Optuna's MedianPruner. Log every trial to MLflow as nested runs.
2. Refit the best pipeline on train + validation.
3. Evaluate ONCE on the test split: ROC-AUC, PR-AUC, KS, Brier score, confusion matrix at 0.5.
   Also check calibration; if it is poor, wrap the model with CalibratedClassifierCV (isotonic, cv=3) and compare.
4. Save the final pipeline to models/loan_default_model.joblib and models/model_meta.json
   (model type, version, training date, feature lists, best params, test metrics, sklearn/xgboost versions).
5. Register it in the MLflow model registry as "loan_default_model" and set the alias "production".
6. Save charts: 14_test_roc.png, 15_test_pr.png, 16_confusion_matrix.png, 17_calibration.png, 18_optuna_history.png.
7. Write reports/final_model.md: chosen model, best params, test metrics, comparison with the baseline (~0.70),
   and an honest note on the 0.85 AUC target versus what this data allows without leakage.
```

---

## Step 5 - Decision threshold and NPA reduction simulation

```
Read CLAUDE.md first. Create src/business.py using the saved model and the splits from get_splits():
1. On the VALIDATION split, choose two probability cut-offs that create three decision bands:
   APPROVE (low risk), REVIEW (medium), REJECT (high). Pick them so that REJECT catches a meaningful share of
   defaulters while rejecting as few good borrowers as possible; show the trade-off table for several cut-offs
   (share of loans in each band, default rate per band, % of defaulters caught, % of good loans lost).
   Save the chosen cut-offs into models/model_meta.json.
2. On the TEST split, simulate the business impact versus the baseline "approve everyone" (which is what
   actually happened to these loans):
   - number and amount (sum of loan_amnt) of defaulted loans approved, baseline vs model
     (scenario A: REVIEW counted as approved; scenario B: REVIEW counted as rejected)
   - % reduction in defaulted loans / defaulted amount = the "NPA reduction"
   - good loans lost (opportunity cost)
3. Save reports/business_simulation.md with the tables and a plain-language explanation, and
   reports/figures/19_decision_bands.png and 20_npa_reduction.png.
Report the real NPA reduction number and compare it honestly with the 30% target.
```

---

## Step 6 - SHAP explainability

```
Read CLAUDE.md first. Create src/explain.py for the saved final pipeline:
1. Global explanations on a 2,000-row sample of the test split with shap.TreeExplainer (or the right explainer
   for the final model): save 21_shap_summary_beeswarm.png and 22_shap_importance_bar.png.
   Aggregate one-hot columns back to their original feature (e.g. all purpose_* -> purpose).
2. explain_one(applicant: dict) -> dict with probability, decision band, and the top 5 reasons, each with:
   feature, readable label (e.g. "Interest rate"), the applicant's value, SHAP contribution, and direction
   ("increases risk" / "decreases risk"). Keep a FEATURE_LABELS dict for readable names.
3. A reason_text() helper that turns the top reasons into short sentences, e.g.
   "Interest rate of 18.5% increases risk".
4. The explainer must be created once and reused (it will be loaded by the API).
5. Add a notebook notebooks/02_model_and_shap.ipynb that shows the model results and SHAP charts with short insights.
6. Add tests/test_explain.py (returns 5 reasons, contributions sum close to model output in log-odds, works with missing values).
Write reports/shap_insights.md with the top 10 features and what they mean for lending.
```

---

## Step 7 - FastAPI backend with PostgreSQL

```
Read CLAUDE.md first. Build the backend in app/ (FastAPI + SQLAlchemy 2.0 + PostgreSQL from docker-compose):
Structure: app/main.py, app/config.py (reads .env), app/db.py, app/models.py, app/schemas.py, app/auth.py,
app/services/scoring.py, app/routers/{auth,applications,model,monitoring}.py, app/seed.py.
1. Tables: users (id, username, hashed_password, role: admin/officer, created_at), loan_applications (all input
   fields + applicant name, created_by, created_at), predictions (application_id, probability, decision,
   top_reasons JSON, model_version, latency_ms, created_at), audit_log (user, action, details, timestamp),
   drift_reports (created_at, drift_share, dataset_drift, prediction_drift, report_path). Use Alembic migrations.
2. Auth: POST /auth/login returns a JWT (PyJWT, bcrypt password hashing). Roles: admin, officer.
   app/seed.py creates one admin and one officer from .env values.
3. Endpoints: GET /health; POST /applications (validate input with Pydantic ranges, derive the engineered
   features the same way src/clean_data.py does, score with the saved pipeline, explain with src/explain.py,
   save application + prediction + audit row, return probability, decision, top reasons);
   GET /applications (paginated, filter by decision); GET /applications/{id}; GET /model/info (from model_meta.json);
   GET /monitoring/stats (request count, avg latency, approve/review/reject counts, last 7 days) - admin only.
4. Load the model and explainer ONCE at startup. Target latency under 200 ms per request - measure and report it.
5. Swagger docs at /docs with example request bodies.
6. tests/test_api.py using httpx/TestClient and a SQLite test database: login, auth required, valid scoring,
   invalid input returns 422, officer cannot see admin stats.
Give me the exact commands to run migrations, seed users and start the API (uvicorn app.main:app --reload).
```

---

## Step 8 - Streamlit frontend

```
Read CLAUDE.md first. Build the Streamlit frontend in frontend/ that talks ONLY to the FastAPI backend (API_URL from .env):
1. Login page (JWT kept in st.session_state, logout button).
2. "New Application" page: a clear form grouped into Applicant, Loan and Credit history sections, with sensible
   defaults, min/max limits and help text; submit calls POST /applications.
3. Result view: default probability, decision band shown with both colour AND text/icon (APPROVE / REVIEW / REJECT),
   a horizontal Plotly bar chart of the top 5 SHAP reasons (red = increases risk, blue = decreases risk)
   and the reasons as sentences.
4. "History" page: table of past applications with filters (decision, date) and a detail view.
5. "Dashboard" page (admin only): model info (version, test AUC), requests per day, average latency,
   decision mix, and the latest drift status from the API.
6. Handle API errors nicely (show a message instead of a stack trace).
Keep the UI clean and simple (st.set_page_config with a title and wide layout).
Tell me how to run it: streamlit run frontend/app.py
```

---

## Step 9 - Drift monitoring and retraining

```
Read CLAUDE.md first. Add drift monitoring:
1. src/drift.py using Evidently (check the installed version first and use that version's API).
   - Reference data: training loans issued up to 2010-12. Current data: either loans issued in 2011
     (demo mode) or recent applications stored in PostgreSQL (live mode).
   - Data drift for all model features + prediction drift (distribution of predicted probability) +
     PSI of the predicted probability.
   - Save an HTML report and a JSON summary to reports/drift/ with a timestamp, and a row in drift_reports.
   - Alert rule: drift_share > 0.3 or prediction PSI > 0.2 -> status "DRIFT DETECTED", logged and shown on the dashboard.
2. Schedule it with APScheduler inside the FastAPI app (daily at 02:00) and add POST /monitoring/drift/run
   (admin only) to run it on demand, plus GET /monitoring/drift/latest.
3. src/retrain.py: retrain with the tuned parameters on data up to a cut-off date, evaluate on the newest data,
   and replace the production model (new version in MLflow + models/) ONLY if its AUC is not worse. Log the decision.
4. Demo in a notebook notebooks/03_drift_demo.ipynb: show that 2011 loans drift from 2007-2010 loans,
   which features drift most, and what retraining changes.
5. Tests for the alert rule and the PSI function.
```

---

## Step 10 - Docker, CI and README

```
Read CLAUDE.md first. Package the project:
1. Dockerfile for the API and Dockerfile.frontend for Streamlit (python:3.12-slim, non-root user, only needed files).
2. Extend docker-compose.yml with api and frontend services (depends_on db healthy, env from .env,
   models/ and reports/ mounted, memory limits so everything fits in 8 GB RAM: db 512m, api 1g, frontend 512m).
   On startup the api runs migrations and seeds users.
3. A Makefile (or scripts/*.sh) with: setup, clean-data, train, tune, api, frontend, test, drift, up, down.
4. .github/workflows/ci.yml: install requirements, run pytest (SQLite, no Docker) on every push.
5. Rewrite README.md properly: project overview, architecture diagram (mermaid), dataset, results table,
   screenshots placeholders, how to run locally and with Docker, project structure, team/credits.
Verify: docker compose up --build starts all three services and the Streamlit app can score a loan end to end.
```

---

## Step 11 - Report, presentation and viva preparation

```
Read CLAUDE.md first, then read everything in reports/ and the notebooks. Write the documentation in docs/:
1. docs/Project_Report.md - full project report with chapters: Abstract, Introduction and problem statement,
   Objectives, Literature review (short, with real references), Dataset description, Data cleaning and leakage,
   EDA (use the charts in reports/figures), Methodology, Models and tuning, Results (real numbers from reports/),
   Business impact (NPA simulation), Explainability (SHAP), System design (architecture, API, database schema),
   Drift monitoring, Testing, Limitations, Conclusion, Future scope, References, Appendix (how to run).
   Use ONLY numbers that exist in the reports/ files - no invented results.
2. docs/presentation_outline.md - 12-15 slides: title, one-line message per slide, bullet points, which chart to show.
3. docs/viva_questions.md - 40 likely viva questions with short, simple answers (ML basics, why AUC, imbalance,
   leakage, SHAP, drift, API design, security, why these models, limitations).
4. docs/demo_script.md - a 5-minute live demo script step by step.
Keep the language simple and clear.
```

---

## Step 12 (optional) - Kaggle submission

```
Read CLAUDE.md first. Create src/predict_kaggle.py that loads the production model, scores
data/processed/test_clean.parquet, and writes submissions/submission.csv in exactly the format of
data/raw/loan_sample_submission.csv (id, loan_status). Check the competition's evaluation metric: if it is
AUC, write probabilities; if it is accuracy/F1, write 0/1 using the threshold from model_meta.json.
Validate the file (same ids and row count as the sample) and tell me how to submit it.
```
