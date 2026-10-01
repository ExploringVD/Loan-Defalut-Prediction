# SHAP insights - what drives the model

Made by `python src/explain.py` on a random sample of 2,000 test loans (random_state=42).
SHAP values are in log-odds of the 3 inner XGBoost models of the calibrated final model, averaged; one-hot
and missing-flag columns are added back to their original feature. Additivity check: base value + SHAP
values = average model log-odds (max error 1.4e-05). Base value (average log-odds): 0.057.

## Features ranked by importance

| Rank | Feature | Mean \|SHAP\| | Share | How it pushes risk | What it means for lending |
|---|---|---|---|---|---|
| 1 | Yearly income (`person_income`) | 1.084 | 21.4% | higher value -> lower risk (lowest 20% of values +1.23, highest 20% -1.71) | Low income means little room for repayments; high income protects against default even for larger loans. |
| 2 | Loan grade (`loan_grade_num`) | 0.882 | 17.4% | higher value -> higher risk (lowest 20% of values -1.00, highest 20% +0.71) | The lender's own risk grade. Grades D-G push risk up strongly, A-B pull it down - the model confirms the grading but adds the borrower's situation on top. |
| 3 | Loan as % of income (`loan_percent_income`) | 0.752 | 14.9% | higher value -> higher risk (lowest 20% of values -0.24, highest 20% +1.30) | How stretched the borrower is. It works like a switch: up to about 30% of income it hardly matters, above 30% it is one of the strongest pushes towards default - especially for renters (the 'perfect rule' from Step 3). |
| 4 | Home ownership (`person_home_ownership`) | 0.726 | 14.4% | lowest risk: OWN (-4.32), highest risk: RENT (+0.56) | Renters are much riskier than owners or mortgage holders - housing costs and stability. Combined with a high loan-to-income ratio it is the 'perfect rule' found in Step 3. |
| 5 | Loan purpose (`loan_intent`) | 0.589 | 11.7% | lowest risk: VENTURE (-1.57), highest risk: HOMEIMPROVEMENT (+0.70) | Debt consolidation, medical and home-improvement loans are riskier than education or venture loans - the purpose tells something about financial pressure. |
| 6 | Interest rate (`loan_int_rate`) | 0.360 | 7.1% | higher value -> higher risk (lowest 20% of values -0.83, highest 20% +0.37) | Higher rates mean higher monthly payments and reflect risk the lender already priced in. |
| 7 | Loan amount (`loan_amnt`) | 0.299 | 5.9% | higher value -> higher risk (lowest 20% of values -0.36, highest 20% +0.50) | Matters mostly through loan-to-income; on its own the amount says less. |
| 8 | Years employed (`person_emp_length`) | 0.169 | 3.3% | higher value -> lower risk (lowest 20% of values +0.20, highest 20% -0.02) | Longer employment means a more stable income. |
| 9 | Age (`person_age`) | 0.131 | 2.6% | higher value -> lower risk (lowest 20% of values +0.09, highest 20% -0.25) | Weak on its own; small effects, partly through the link with credit history. |
| 10 | Credit history length (`cb_person_cred_hist_length`) | 0.047 | 0.9% | higher value -> lower risk (lowest 20% of values -0.01, highest 20% -0.04) | Weakest feature - longer histories help only slightly. |
| 11 | Past default on file (`cb_person_default_on_file`) | 0.017 | 0.3% | past default +0.01, none -0.01 | Almost no effect in the model even though past defaulters default twice as often (EDA): the information is already in the grade (no borrower with a past default got grade A or B), so the model uses the grade instead. |

## Key points

- The top 3 features (Yearly income, Loan grade, Loan as % of income) carry 54% of the total SHAP importance; Home ownership is next (14%). These match the EDA (charts 03, 04, 07, 08) and the step patterns that made tree models win in Step 3.
- **Loan as % of income is a switch, not a slope:** average SHAP -0.45 for loans up to 30% of income, +2.80 above 30% - and +3.73 for renters above 30% (the perfect-rule group).
- **Home ownership:** owning a home pulls risk down strongly (-4.32), renting pushes it up (+0.56). SHAP values are in log-odds of the uncalibrated model, whose scores range widely, so single values can look large; compare them with each other, not as probabilities.
- **Past default on file** has almost no weight: the grade already contains it.
- Age and credit history length matter least, as the single-feature AUCs (0.522 and 0.519) suggested.
- SHAP explains the uncalibrated XGBoost score (log-odds). The probability shown to users comes from the calibrated model; calibration keeps the order of risk, so the reasons keep their direction.
- Every API prediction stores its top 5 reasons, so a loan officer can see *why* an application was flagged.

Charts: `reports/figures/22_shap_summary_beeswarm.png`, `reports/figures/23_shap_importance_bar.png`.
