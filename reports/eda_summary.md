# EDA Summary - Loan Default Prediction

Data: Kaggle "Credit Risk Dataset", 32,581 loan applicants in the raw file, **32,409 after cleaning**
(165 exact duplicates and 7 impossible rows removed). One row = one loan, `loan_status = 1` = defaulted.
There is no date column. Full analysis and code: `notebooks/01_eda.ipynb`. Charts: `reports/figures/`.
Leakage check numbers: `reports/leakage_check.md` (`python src/leakage_check.py`).

| # | Finding | Chart | What we do with it |
|---|---|---|---|
| 1 | Raw file has 165 exact duplicates and impossible values (age up to 144, employment up to 123 years) | 10 | Removed in `src/clean_data.py`; 7 impossible rows dropped, checks fail loudly |
| 2 | 21.9% of loans default (7,088 of 32,409), about 1 in 5 - imbalanced | 02 | ROC-AUC / PR-AUC / recall, class weights or SMOTE, stratified 70/15/15 split |
| 3 | Default rate rises from 10.0% (grade A) to 98.4% (grade G), with a jump from C (20.8%) to D (59.0%) | 03 | Strongest single feature (AUC 0.725); used once, as `loan_grade_num` (A=1 ... G=7) |
| 4 | Loans above 30% of income default 70.4% vs 15.4% below (11.8% of loans are above) | 08 | Keep `loan_percent_income`; the sharp step favours tree models |
| 5 | Poorest fifth of borrowers defaults 43.3% vs 9.2% for the richest fifth | 07 | Keep income; log1p for Logistic Regression |
| 6 | Interest rate: 8.8% default in the cheapest fifth vs 50.6% in the most expensive | 07 | Keep for trees; dropped for Logistic Regression (duplicate of grade) |
| 7 | Renters default 31.6%, mortgage holders 12.6%, owners 7.5% | 04 | Keep home ownership (one-hot) |
| 8 | Past default on file: 37.9% vs 18.4%; none of these borrowers got grade A or B | 06 | Keep the flag as 1/0; it overlaps with grade (r = 0.54) |
| 9 | Debt consolidation (28.7%), medical (26.8%), home improvement (26.2%) riskiest; venture safest (14.9%) | 05 | Keep loan intent; the drift simulation shifts towards debt consolidation and medical |
| 10 | Longer employment, lower risk (27.9% for 0-1 years vs 16.8% for 9+); age barely matters (24.0% to 20.7%) | 07 | Keep both; expect age to rank low in SHAP |
| 11 | Missing: interest rate 9.5%, employment length 2.7%; missing employment length defaults 31.7% vs 21.6% | 01 | Median imputation + missing flags inside the pipeline (training split only) |
| 12 | Grade ~ interest rate r = 0.93, age ~ credit history length r = 0.88 | 11 | Logistic Regression drops `loan_int_rate` and `cb_person_cred_hist_length` |
| 13 | Defaulters: median loan/income 0.24 vs 0.13, interest 13.5% vs 10.6%, income $41,655 vs $60,000 | 09 | Clear but overlapping differences - a multi-feature model is needed |
| 14 | Best single-feature AUC 0.725, far below the 0.90 leakage alarm; quick models: LR 0.878, GB 0.936 (validation) | 12 | No leakage found; realistic final AUC ~0.93-0.95 |
| 15 | No date column | - | Drift monitoring uses a simulated "current batch" built from the test split (Step 9) |
| 16 | Found in Step 3: on the training split every renter with a loan above 30% of income defaulted (1,650 of 1,650) | 14 | Not leakage, but a sign of rule-made labels; AUC here is likely higher than on real loans - report as a limitation |

Missing values are **not** filled in cleaning. They are imputed inside the model pipeline using medians from the
training split only, so validation/test data never influences the fill values.
