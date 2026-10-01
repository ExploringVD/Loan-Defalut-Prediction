# Drift monitoring - simulated batches

Made by `python src/drift.py`. The dataset has no dates, so drift is **simulated**: two batches of 2,000 loans
from the test split (`src/simulate_batches.py`). Reference = the training split (22,685 loans). Alert rule: more than 30% of the features drifted, or
prediction PSI above 0.2.

| | no_drift batch | drift batch |
|---|---|---|
| Features drifted | 0 of 11 (0%) | 4 of 11 (36%) |
| Drifted features | - | loan_int_rate, loan_percent_income, person_income, loan_intent |
| Prediction PSI | 0.007 | 0.198 |
| Mean predicted probability of default (reference 22.1%) | 22.0% | 30.8% |
| Share in the REJECT band (reference 18.4%) | 17.9% | 26.4% |
| **Status** | **NO DRIFT** | **DRIFT DETECTED** |

## Drift score per feature

Higher score = bigger difference from the training data; a feature drifts when its score reaches the threshold.

| Feature | Test used | no_drift score | drift score | Threshold |
|---|---|---|---|---|
| loan_int_rate | Wasserstein distance (normed) | 0.018 | 0.897 **drift** | 0.1 |
| loan_percent_income | Wasserstein distance (normed) | 0.014 | 0.250 **drift** | 0.1 |
| person_income | Wasserstein distance (normed) | 0.046 | 0.173 **drift** | 0.1 |
| loan_intent | Jensen-Shannon distance | 0.019 | 0.113 **drift** | 0.1 |
| loan_amnt | Wasserstein distance (normed) | 0.016 | 0.033 | 0.1 |
| person_emp_length | Wasserstein distance (normed) | 0.039 | 0.028 | 0.1 |
| cb_person_cred_hist_length | Wasserstein distance (normed) | 0.027 | 0.027 | 0.1 |
| person_age | Wasserstein distance (normed) | 0.039 | 0.021 | 0.1 |
| loan_grade_num | Wasserstein distance (normed) | 0.019 | 0.019 | 0.1 |
| cb_person_default_on_file | Jensen-Shannon distance | 0.002 | 0.006 | 0.1 |
| person_home_ownership | Jensen-Shannon distance | 0.013 | 0.006 | 0.1 |

The drift batch was built with: interest rates +3 points, incomes -15% (so loan / income goes up), and debt
consolidation + medical loans raised from about 35% to 50% of the batch. Interactive HTML reports: `reports/drift/`.
