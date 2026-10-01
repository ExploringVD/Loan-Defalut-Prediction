# Leakage check and quick baselines

Made by `python src/leakage_check.py`. Training split fits everything; numbers are on the validation split.
Alarm rules: single feature > 0.90 AUC, or quick model > 0.97 AUC.

## Single-feature AUC (validation)

| Feature | Type | AUC |
|---|---|---|
| loan_grade_num | numeric | 0.725 |
| loan_int_rate | numeric | 0.720 |
| loan_percent_income | numeric | 0.720 |
| person_income | numeric | 0.701 |
| person_home_ownership | categorical | 0.656 |
| cb_person_default_on_file | numeric | 0.586 |
| loan_intent | categorical | 0.575 |
| person_emp_length | numeric | 0.565 |
| loan_amnt | numeric | 0.555 |
| person_age | numeric | 0.522 |
| cb_person_cred_hist_length | numeric | 0.519 |

## Quick untuned models (validation)

| Model | AUC |
|---|---|
| Logistic Regression | 0.878 |
| Gradient Boosting | 0.936 |

## Extra checks

- Validation rows identical to a training row: 0; test rows: 0
  (exact duplicates are removed in cleaning, so no loan is in two splits).
- Gradient Boosting validation AUC when the strongest features are left out (a leak usually sits in one column, so removing it would make the AUC collapse):

| Left out | AUC |
|---|---|
| nothing | 0.936 |
| loan_grade_num | 0.924 |
| loan_grade_num + loan_int_rate | 0.888 |
| loan_percent_income | 0.929 |

**Result:** no feature or model crosses the alarm lines.
