# Loan Default Prediction

Predicts whether a loan will default, explains each prediction with SHAP, serves scores through FastAPI + Streamlit, and watches for data drift.

## Dataset

**Kaggle "Credit Risk Dataset"** - https://www.kaggle.com/datasets/laotse/credit-risk-dataset

32,581 loan applicants, 12 columns, target `loan_status` (1 = defaulted, 21.8% in the raw file). Every column is
known at application time (no payment history), and there is no date column.

| Column | Meaning |
|---|---|
| person_age, person_income, person_emp_length | Applicant age, yearly income, years employed |
| person_home_ownership | RENT, MORTGAGE, OWN, OTHER |
| loan_intent | EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT |
| loan_grade | Lender's risk grade A (safest) ... G |
| loan_amnt, loan_int_rate | Loan amount and interest rate (%) |
| loan_percent_income | Loan amount / yearly income |
| cb_person_default_on_file | Credit bureau: defaulted before? (Y/N) |
| cb_person_cred_hist_length | Credit history length (years) |
| **loan_status** | **Target: 1 = defaulted, 0 = repaid** |

The raw file is not in git. Put it at `data/raw/credit_risk_dataset.csv`:

- **Browser:** open the Kaggle link, click *Download*, unzip, and move `credit_risk_dataset.csv` into `data/raw/`.
- **Kaggle CLI** (needs a Kaggle API token in `~/.kaggle/kaggle.json`):
  ```bash
  pip install kaggle
  kaggle datasets download -d laotse/credit-risk-dataset -p data/raw --unzip
  ```

The cleaned file `data/processed/credit_clean.parquet` *is* in git, so training and the tests work without the raw file.

### Cleaning (`src/clean_data.py`)

| | Rows |
|---|---|
| Raw file | 32,581 |
| Exact duplicates removed | 165 |
| person_age > 100 removed (ages 123, 144) | 5 |
| person_emp_length > person_age - 14 removed (e.g. 123 years employed) | 2 |
| **Clean** | **32,409** (21.9% default) |

Cleaning also adds `loan_id` (row number in the raw file) and `loan_grade_num` (A=1 ... G=7), and turns
`cb_person_default_on_file` Y/N into 1/0. Missing values (`loan_int_rate` 9.5%, `person_emp_length` 2.7%) are
**not** filled here - they are imputed inside the model pipeline using the training split only.

### Model features (`src/load_data.py`)

11 features: 9 numeric (`person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`,
`loan_percent_income`, `loan_grade_num`, `cb_person_default_on_file`, `cb_person_cred_hist_length`) and
2 categorical (`person_home_ownership`, `loan_intent`). `get_splits()` gives a stratified 70/15/15 split
(22,685 / 4,862 / 4,862 rows, 21.9% default in each, `random_state=42`).

### Earlier version

The project first used Lending Club data (honest AUC ~0.70; a fake 0.998 when leakage columns were kept).
That work is archived locally in `archive/lending_club/` (git-ignored), with notes in `archive/lending_club/NOTES.md`.

## Folder layout

```
data/
  raw/            credit_risk_dataset.csv                       (Kaggle download, not in git)
  processed/      credit_clean.parquet, cleaning_report.json    (made by src/clean_data.py; .csv copy not in git)
src/
  clean_data.py     clean the raw data (duplicates, impossible rows, grade -> number, Y/N -> 1/0)
  load_data.py      load cleaned data + 70/15/15 train/val/test split
  features.py       preprocessing (imputation, log, scaling, one-hot) for "linear" and "tree" models
  leakage_check.py  single-feature AUC + quick Logistic Regression / Gradient Boosting baselines
notebooks/
  01_eda.ipynb    exploratory data analysis (charts + insights, executed)
reports/
  eda_summary.md    key EDA findings table (for the project report)
  leakage_check.md  leakage check and baseline AUCs
  figures/          all charts as PNG
tests/            pytest tests
archive/          old Lending Club work (not in git)
pytest.ini        pytest settings (only runs tests/)
requirements.txt
docker-compose.yml  PostgreSQL 16 (service "db")
.env.example        settings template (copy to .env)
```

## Setup (once)

```bash
git clone https://github.com/ExploringVD/Loan-Defalut-Prediction.git
cd Loan-Defalut-Prediction
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
source .venv/bin/activate
python src/clean_data.py      # re-creates data/processed from data/raw (needs the Kaggle file)
python src/load_data.py       # prints the train/val/test sizes and default rates
python src/leakage_check.py   # single-feature AUC + quick baselines -> reports/leakage_check.md
pytest -q                     # run the tests
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # re-run the EDA
```

## Preprocessing (src/features.py)

`build_preprocessor(kind)` returns an **unfitted** `ColumnTransformer`. It always goes inside a
`Pipeline` with the model, so imputation medians, scaler means and categories are learned from the
training rows only.

| | `"linear"` (Logistic Regression) | `"tree"` (Random Forest / XGBoost) |
|---|---|---|
| Missing numbers | median + `<col>_missing` flag | median (+ flag for `loan_int_rate`) |
| Skewed columns | `log1p` (person_income, loan_amnt) | unchanged |
| Scaling | StandardScaler | none |
| Categoricals | one-hot, categories with < 20 rows -> `<col>_other`, unseen -> all zeros | same |
| Correlated pairs | drops loan_int_rate, cb_person_cred_hist_length (`LINEAR_DROP_FEATURES`) | keeps all |
| Output columns | 18 | 20 |

`get_feature_names(fitted)` gives readable names (e.g. `person_emp_length_missing`, `loan_intent_MEDICAL`) for SHAP.

## Results so far

| | Validation ROC-AUC |
|---|---|
| Best single feature (loan_grade_num) | 0.725 |
| Logistic Regression (untuned) | 0.878 |
| Gradient Boosting (untuned) | 0.936 |

No feature passes the 0.90 single-feature leakage alarm and no model passes 0.97 (`reports/leakage_check.md`).
Key EDA findings: `reports/eda_summary.md`.
