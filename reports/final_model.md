# Final model - tuned XGBoost

Made by `python src/tune.py` on 2026-10-01. Registered in MLflow as `loan_default_model` version 1 with alias `production`; saved as `models/loan_default_model.joblib`.

## Chosen model

- **XGBoost pipeline + isotonic calibration**: tree preprocessor (median imputation + missing flag for loan_int_rate, one-hot categoricals) + XGBoost with `scale_pos_weight` = negatives / positives, wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`.
- Tuned with Optuna (TPE sampler, MedianPruner): 50 trials (47 completed, 3 pruned), objective = mean 5-fold CV ROC-AUC on the training split (22,685 loans).
- Final fit on train + validation (27,547 loans); tested once on 4,862 loans.

### Best hyperparameters

| Parameter | Value |
|---|---|
| learning_rate | 0.06891 |
| max_depth | 4 |
| min_child_weight | 3 |
| subsample | 0.8912 |
| colsample_bytree | 0.6544 |
| n_estimators | 1000 |
| reg_lambda | 1.656 |
| gamma | 1.541 |

## Tuning and overfitting check (validation split, model fitted on the training split)

| | CV ROC-AUC | Train ROC-AUC | Validation ROC-AUC | Train - validation gap |
|---|---|---|---|---|
| XGBoost untuned (Step 3) | 0.942 | 0.995 | 0.955 | 0.040 |
| XGBoost tuned | 0.946 | 0.986 | 0.957 | 0.028 |

Tuning reduced overfitting: the train - validation gap went from 0.040 (untuned) to 0.028 (tuned), while validation ROC-AUC went from 0.955 to 0.957.

## Calibration check (validation split)

Calibration = can a predicted 30% be read as "30 of 100 such loans default"? `scale_pos_weight` pushes the
probabilities up, so we checked. ECE = expected calibration error (average gap between predicted and real
default rate); "poor" means ECE above 0.03.

| Version | ROC-AUC | Brier score | ECE |
|---|---|---|---|
| Uncalibrated | 0.957 | 0.0621 | 0.078 |
| Calibrated (isotonic, cv=3) | 0.957 | 0.0509 | 0.010 |

**Decision:** calibration was poor (ECE 0.078 > 0.03). Isotonic calibration lowered the validation Brier score (0.0621 -> 0.0509) and ECE (0.078 -> 0.010), ROC-AUC 0.957 -> 0.957, so the final model is the CALIBRATED one.

## Test results (test split used once)

| Metric | Value |
|---|---|
| ROC-AUC | **0.952** |
| PR-AUC | 0.909 |
| KS statistic | 0.767 |
| Brier score | 0.0514 |
| Expected calibration error | 0.004 |
| Precision @ 0.5 | 0.975 |
| Recall @ 0.5 | 0.726 |
| F1 @ 0.5 | 0.832 |

Confusion matrix at threshold 0.5 (4,862 test loans):

| | Predicted repaid | Predicted default |
|---|---|---|
| **Actual repaid** | 3,779 (correct) | 20 (good loans wrongly flagged) |
| **Actual default** | 291 (defaults missed) | 772 (defaults caught) |

The 0.5 threshold is only a reference point; the real APPROVE / REVIEW / REJECT cut-offs are chosen in Step 5.

## Compared with the baselines

| Model | ROC-AUC | Split |
|---|---|---|
| Logistic Regression (quick, untuned) | 0.878 | validation |
| Gradient Boosting (quick, untuned) | 0.936 | validation |
| XGBoost untuned (Step 3) | 0.955 | validation |
| XGBoost tuned, fitted on train | 0.957 | validation |
| **Final model** (train + validation) | **0.952** | **test** |

## Honest note on the 0.85 AUC target

The synopsis target of ROC-AUC >= 0.85 is **met without leakage**: the final test ROC-AUC is 0.952, in line with the ~0.93-0.95 we expected for this dataset and below the 0.97 leakage alarm. Every feature is known at application time, all
preprocessing is fitted inside the pipeline on training rows only, and the test split was used once.
A result above 0.97 would have been treated as a leakage warning (`python src/leakage_check.py`).
The test score (0.952) is a little below the validation score (0.957) and close to the CV mean (0.946): normal variation between splits, and the CV mean was the tuning target.
One caveat stays: this dataset contains a perfect rule (every renter whose loan is more than 30% of income
defaulted, 1,650 of 1,650 in the training split - see `reports/model_comparison.md`), so the same model would
probably score lower on real lending data.

Project history: the first version used Lending Club data, where the honest ROC-AUC was about 0.70 and keeping
post-loan payment columns gave a fake 0.998 (leakage) - see `archive/lending_club/NOTES.md`.

Charts: `reports/figures/15_test_roc.png`, `16_test_pr.png`, `17_confusion_matrix.png`, `18_calibration.png`,
`19_optuna_history.png`.
