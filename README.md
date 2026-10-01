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
  clean_data.py     clean the raw data (duplicates, impossible rows, grade -> number, Y/N -> 1/0);
                    to_model_input() converts the 11 raw applicant fields for the model (used by SHAP + API)
  load_data.py      load cleaned data + 70/15/15 train/val/test split
  features.py       preprocessing (imputation, log, scaling, one-hot) for "linear" and "tree" models
  leakage_check.py  single-feature AUC + quick Logistic Regression / Gradient Boosting baselines + perfect-rule check
  train.py          train + compare 4 models (CV + validation), log everything to MLflow
  tune.py           Optuna tuning of XGBoost, calibration, ONE test evaluation, save + register the final model
  business.py       APPROVE / REVIEW / REJECT cut-offs (validation) + NPA reduction simulation (test)
  decision.py       the band rule (APPROVE / REVIEW / REJECT), shared by business.py, explain.py and the API
  explain.py        SHAP: explain_one(applicant) -> probability, band, top 5 reasons; global charts
models/
  loan_default_model.joblib   final pipeline (preprocessing + calibrated XGBoost) used by the API
  model_meta.json             version, features, best params, test metrics, decision cut-offs, library versions
notebooks/
  01_eda.ipynb    exploratory data analysis (charts + insights, executed)
  02_model_and_shap.ipynb   model results (Steps 3-5) + SHAP explanations (executed)
reports/
  eda_summary.md    key EDA findings table (for the project report)
  leakage_check.md  leakage check and baseline AUCs
  model_comparison.csv / .md   Step 3 model comparison
  final_model.md    Step 4 final model: best params, calibration, test metrics
  business_simulation.md / .csv   Step 5 decision bands and NPA reduction
  shap_insights.md / shap_importance.csv   Step 6 features ranked by SHAP and what they mean for lending
  figures/          all charts as PNG (01-12 EDA, 13-14 models, 15-19 final model, 20-21 business, 22-25 SHAP)
mlflow.db, mlruns/  local MLflow tracking (made by src/train.py, not in git)
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
python src/train.py           # train + compare 4 models (~35 s) -> reports/model_comparison.*, MLflow
python src/tune.py            # Optuna 50 trials + final model (~3 min) -> models/, reports/final_model.md
python src/business.py        # decision cut-offs + NPA simulation (~10 s) -> reports/business_simulation.md
python src/explain.py         # SHAP global charts (~6 s) -> reports/shap_insights.md
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_and_shap.ipynb   # model + SHAP notebook
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

## Model comparison (src/train.py)

Four pipelines (preprocessing + model), trained on the training split. 5-fold stratified CV ROC-AUC on the
training split, then fit on the whole training split and scored on the validation split. The test split is not used.

| Model | Imbalance handling | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val KS | Recall @0.5 |
|---|---|---|---|---|---|---|
| **XGBoost** | scale_pos_weight | 0.942 +/- 0.005 | **0.955** | 0.912 | 0.772 | 0.802 |
| XGBoost + SMOTE | SMOTE in an imblearn Pipeline | 0.943 +/- 0.004 | 0.952 | 0.912 | 0.771 | 0.751 |
| Random Forest | class_weight="balanced_subsample" | 0.930 +/- 0.006 | 0.940 | 0.896 | 0.748 | 0.712 |
| Logistic Regression | class_weight="balanced" | 0.873 +/- 0.005 | 0.879 | 0.726 | 0.601 | 0.791 |

**XGBoost is the model to tune (Step 4):** best validation AUC; SMOTE adds no real gain (+0.001 CV, within noise);
trees beat Logistic Regression by 0.076 because they learn step patterns (grade C->D, loan > 30% of income, renters).
No model passes the 0.97 leakage alarm. Full table and reasons: `reports/model_comparison.md`.
Charts: `reports/figures/13_roc_curves_validation.png`, `14_pr_curves_validation.png`.

**Honest limitation:** on the training split, *every* renter whose loan is more than 30% of income defaulted
(1,650 of 1,650). That is not leakage (both are known at application time), but such a clean rule is unusual
for real loans, so AUCs on this dataset are probably higher than on real lending data.

### MLflow (local, no server needed)

`src/train.py` logs one parent run (`model_comparison`) with four child runs: parameters, CV and validation
metrics, a validation plot (ROC, PR, confusion matrix) and the fitted pipeline (saved with `skops`, a safer
format than pickle). To browse the runs:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db     # takes ~20 s to start
```

Open http://127.0.0.1:5000, choose the experiment **loan-default**, expand **model_comparison** to see the four
models, tick them and press **Compare**. Stop the UI with Ctrl+C.

## Final model (src/tune.py)

1. **Optuna** (TPE sampler, MedianPruner, 50 trials: 47 completed, 3 pruned) tunes 8 XGBoost hyperparameters on the
   mean 5-fold CV ROC-AUC of the training split. Every trial is a nested MLflow run.
2. The best pipeline is fitted on the training split and checked on the validation split (overfitting check).
3. **Calibration check on the validation split:** `scale_pos_weight` made the probabilities too high
   (expected calibration error 0.078). Isotonic calibration (`CalibratedClassifierCV`, cv=3) brought it to 0.010
   and lowered the Brier score (0.0621 -> 0.0509) with the same AUC, so the final model is calibrated.
4. Refit on train + validation (27,547 loans), then **one** evaluation on the test split (4,862 loans).
5. Saved to `models/loan_default_model.joblib` + `models/model_meta.json` and registered in MLflow as
   `loan_default_model` (version 1, alias `production`).

| | CV ROC-AUC | Train ROC-AUC | Validation ROC-AUC | Train - validation gap |
|---|---|---|---|---|
| XGBoost untuned (Step 3) | 0.942 | 0.995 | 0.955 | 0.040 |
| XGBoost tuned | 0.946 | 0.986 | 0.957 | 0.028 |

**Test split (used once):** ROC-AUC **0.952**, PR-AUC 0.909, KS 0.767, Brier 0.0514, calibration error 0.004.
At threshold 0.5: precision 0.975, recall 0.726 (772 of 1,063 defaults caught, 20 of 3,799 good loans wrongly flagged).
The 0.85 AUC target is met without leakage. Best parameters, the calibration decision and all charts
(15-19): `reports/final_model.md`.

Load the model in Python:

```python
import joblib
model = joblib.load("models/loan_default_model.joblib")      # full pipeline: takes the 11 raw feature columns
model.predict_proba(X)[:, 1]                                  # probability of default (calibrated)
# or from the MLflow registry:  mlflow.sklearn.load_model("models:/loan_default_model@production")
```

## Decision bands and NPA reduction (src/business.py)

**Cut-offs (chosen on the validation split** with the same pipeline refit on the training split only, so the
scores are honest; saved in `models/model_meta.json`):

| Band | Rule | Validation: loans | Validation: default rate |
|---|---|---|---|
| APPROVE | probability < 0.10 (REVIEW + REJECT still hold >= 90% of defaulters) | 63.8% | 2.9% |
| REVIEW | 0.10 to 0.35 - a credit officer decides | 18.4% | 19.4% |
| REJECT | probability >= 0.35 (turns away <= 2% of good borrowers) | 17.8% | 92.7% |

**Simulation on the test split** (4,862 loans; baseline = approve everyone, which is what really happened):

| | Approve everyone | A: REVIEW approved | B: REVIEW rejected |
|---|---|---|---|
| Defaulted loans approved | 1,063 | 257 | 89 |
| Defaulted amount approved (NPA) | $11.44M | $2.31M | $0.79M |
| **NPA reduction (amount)** | - | **79.8%** | **93.1%** |
| Good borrowers turned away | 0 | 51 (1.3%) | 868 (22.8%) |
| Net result (1 year interest - full loss on defaults) | -$7.72M | +$1.34M | +$1.96M |

The 30% NPA-reduction target is met in both scenarios - but measured against "approve everyone", on a dataset
with a perfect rule, with a simple money model. Read the caveats in `reports/business_simulation.md`.
Charts: `reports/figures/20_decision_bands.png`, `21_npa_reduction.png`.

## SHAP explanations (src/explain.py)

The final model is a `CalibratedClassifierCV` (3 calibrated XGBoost pipelines). `explain.py` runs SHAP's
`TreeExplainer` on each inner XGBoost model (with that pipeline's own preprocessor), averages the 3 results and adds
one-hot / missing-flag columns back to the 11 original features. Base value + contributions = the average log-odds of
the inner models (checked to 1e-5). The probability shown to users comes from the calibrated model.

```python
from explain import explain_one, reason_text
result = explain_one({"person_age": 23, "person_income": 30000, "person_home_ownership": "RENT",
                      "person_emp_length": 1, "loan_intent": "MEDICAL", "loan_grade": "D", "loan_amnt": 12000,
                      "loan_int_rate": 15.5, "loan_percent_income": 0.4, "cb_person_default_on_file": "Y",
                      "cb_person_cred_hist_length": 3})
result["probability"], result["decision"]     # (1.0, 'REJECT')
reason_text(result["reasons"])
# ['Loan is 40% of yearly income - increases risk', 'Grade D loan - increases risk', 'Rents home - increases risk',
#  'Yearly income of $30,000 - increases risk', '1 year employed - increases risk']
```

The explainer is built once (about 3 s) and then explains one applicant in about 20 ms.

| Rank | Feature | Share of SHAP importance |
|---|---|---|
| 1 | Yearly income | 21% |
| 2 | Loan grade | 17% |
| 3 | Loan as % of income | 15% |
| 4 | Home ownership | 14% |
| 5 | Loan purpose | 12% |
| ... | Past default on file (already inside the grade) | 0.3% |

Loan-to-income works like a switch: SHAP -0.45 on average up to 30% of income, +3.73 for renters above 30%
(chart 24). Full table and lending meaning: `reports/shap_insights.md`; walkthrough: `notebooks/02_model_and_shap.ipynb`.

## Results so far

| | Validation ROC-AUC |
|---|---|
| Best single feature (loan_grade_num) | 0.725 |
| Logistic Regression (untuned, quick baseline) | 0.878 |
| Gradient Boosting (untuned, quick baseline) | 0.936 |
| XGBoost (Step 3, untuned) | 0.955 (CV 0.942) |
| XGBoost tuned (Step 4) | 0.957 (CV 0.946) |
| **Final calibrated model - TEST split** | **0.952** |

NPA reduction on the test split: **79.8%** (REVIEW approved) to **93.1%** (REVIEW rejected) vs approving everyone.

No feature passes the 0.90 single-feature leakage alarm and no model passes 0.97 (`reports/leakage_check.md`).
Key EDA findings: `reports/eda_summary.md`.
