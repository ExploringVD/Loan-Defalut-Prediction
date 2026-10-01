# Loan Default Prediction

Predicts whether a loan will default, explains each prediction with SHAP, serves scores through FastAPI + Streamlit, and watches for data drift.

## Folder layout

```
data/
  raw/          loan_train.csv, loan_test.csv, loan_sample_submission.csv  (Kaggle, Lending Club 2007-2011)
  processed/    train_clean.*, test_clean.*, cleaning_report.json          (made by src/clean_data.py)
src/
  clean_data.py   step 1: clean raw data
  load_data.py    step 2: load cleaned data + 70/15/15 train/val/test split
  features.py     step 3: preprocessing (imputation, log, scaling, one-hot) for "linear" and "tree" models
notebooks/
  01_eda.ipynb    exploratory data analysis (charts + insights)
reports/
  eda_summary.md  key EDA findings table (for the project report)
  figures/        all EDA charts as PNG
tests/            pytest tests
requirements.txt
docker-compose.yml  PostgreSQL 16 (service "db")
.env.example        settings template (copy to .env)
```

## Setup (once)

```bash
cd ~/Desktop/"Loan Defalut Prediction"
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # then edit passwords / JWT_SECRET
```

Note: `bcrypt` is pinned to 4.0.1 because passlib 1.7.4 crashes with bcrypt >= 4.1.

### Database (PostgreSQL in Docker)

Start Docker Desktop, then:

```bash
docker compose up -d          # starts the "db" container (512 MB memory limit)
docker compose ps             # STATUS should show (healthy)
docker exec -it loan_db psql -U loan_user -d loan_db   # open a SQL prompt
docker compose down           # stop (data is kept in the "pgdata" volume)
```

If port 5432 is busy because Homebrew Postgres is running, stop it first:
`brew services stop postgresql@16`.

## Run

```bash
python src/clean_data.py   # re-creates data/processed from data/raw
python src/load_data.py    # prints the train/val/test sizes
pytest -q                  # run the tests
```

## Preprocessing (src/features.py)

`build_preprocessor(kind)` returns an **unfitted** `ColumnTransformer`. It always goes inside a
`Pipeline` with the model, so imputation medians, scaler means and categories are learned from the
training rows only.

| | `"linear"` (Logistic Regression) | `"tree"` (Random Forest / XGBoost) |
|---|---|---|
| Missing numbers | median + `<col>_missing` flag | median (+ flag for `mths_since_last_delinq`) |
| Skewed columns | `log1p` (annual_inc, revol_bal, loan_to_income, revol_bal_to_income) | unchanged |
| Scaling | StandardScaler | none |
| Categoricals | one-hot, categories with < 20 rows -> `<col>_other`, unseen -> all zeros | same |
| Correlated pairs | drops sub_grade_num, installment, installment_to_income, pub_rec_bankruptcies (`LINEAR_DROP_FEATURES`) | keeps all |
| Output columns | 93 | 95 |

`get_feature_names(fitted)` gives readable names (e.g. `emp_length_missing`, `addr_state_other`) for SHAP.

## Dataset summary

| | |
|---|---|
| Source | Kaggle competition "Loan Default Prediction" (Lending Club loans, 2007-2011) |
| Train | 27,003 loans, 14.7% defaulted |
| Test | 11,574 loans, no labels (for Kaggle submission only) |
| Features used | 27 (22 numeric + 5 categorical) |
| Removed | 14 leakage columns (known only after the loan was given), IDs, free text |
