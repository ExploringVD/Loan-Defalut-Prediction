# EDA Summary - Loan Default Prediction

Data: 27,003 Lending Club loans (June 2007 - December 2011), after cleaning. Full analysis and code: `notebooks/01_eda.ipynb`. Charts: `reports/figures/`.

| # | Finding | Chart | What we do with it |
|---|---|---|---|
| 1 | Only 14.7% of loans default (about 1 in 7), so the data is imbalanced | 02 | Use ROC-AUC (not accuracy), class weights / SMOTE, stratified splits |
| 2 | Default rate rises from 6.1% (grade A) to 34.2% (grade G) | 03 | Grade / sub-grade are key features |
| 3 | 60-month loans default 25.5% vs 11.2% for 36-month (2.3x) | 03 | Keep `term_months` |
| 4 | Small business loans default 27.0%; car and wedding loans about 10% | 04 | Keep `purpose` (one-hot encoded) |
| 5 | Verified incomes default more (17.1% vs 13.0%) - the lender verified riskier-looking loans | 05 | Keep `verification_status`, explain it carefully |
| 6 | Interest rate: 5.4% default in the cheapest fifth vs 26.4% in the most expensive | 06 | Strongest single feature (AUC 0.667) |
| 7 | Higher income, lower risk (18.6% to 10.9%); high loan-to-income, high risk (11.4% to 22.2%) | 06 | Engineered ratio features are useful |
| 8 | Revolving credit utilisation doubles risk (10.1% to 19.9%) | 06 | Keep `revol_util` |
| 9 | Income is very skewed: median $59,000, max $6,000,000 (9 loans above $1M) | 08 | Log-transform / scale for Logistic Regression |
| 10 | Several features are near-duplicates (int_rate vs sub_grade 0.96, loan_amnt vs installment 0.93) | 09 | Drop duplicates for Logistic Regression |
| 11 | Default rate moves between about 11% and 18% across quarters; loan volume grew about 10x | 11 | Evidence for drift monitoring |
| 12 | No single feature separates the classes well (best single-feature AUC 0.667) | 07, 10 | Multi-feature models; realistic AUC around 0.70-0.75 |

Missing values after cleaning: `mths_since_last_delinq` 64.4% (borrower never late - captured by the `ever_delinquent` flag), `emp_length` 2.7%, `pub_rec_bankruptcies` 1.9%, `revol_util` 0.1%. These are filled inside the model pipeline using training data only.
