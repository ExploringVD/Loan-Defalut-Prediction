# Business simulation - decision bands and NPA reduction

Made by `python src/business.py`. NPA (non-performing assets) here = loans that defaulted.

## 1. Choosing the cut-offs (validation split)

The saved final model was refit on train + validation, so its scores on the validation split would be too
optimistic. To choose the cut-offs honestly, the **same pipeline** (same best parameters, same isotonic
calibration) was refit on the **training split only** and scored the 4,862 validation loans. The
probabilities are calibrated, so a score of 0.30 means "about 30% chance of default".

### Trade-off for a single cut-off (flag every loan with probability >= cut-off)

| Cut-off | Loans flagged | Default rate if flagged | Default rate if not flagged | Defaulters caught | Good loans lost |
|---|---|---|---|---|---|
| 0.05 | 55.5% | 38.7% | 0.9% | 98.2% | 43.6% |
| 0.10 | 36.2% | 55.4% | 2.9% | 91.6% | 20.6% |
| 0.15 | 28.0% | 68.1% | 3.9% | 87.1% | 11.4% |
| 0.20 | 23.7% | 77.5% | 4.6% | 84.1% | 6.8% |
| 0.25 | 20.7% | 85.1% | 5.4% | 80.5% | 3.9% |
| 0.30 | 19.0% | 89.6% | 6.0% | 77.7% | 2.5% |
| 0.35 | 17.8% | 92.7% | 6.6% | 75.4% | 1.7% |
| 0.40 | 17.1% | 95.1% | 6.8% | 74.2% | 1.1% |
| 0.45 | 16.7% | 96.4% | 6.9% | 73.7% | 0.8% |
| 0.50 | 16.4% | 97.2% | 7.1% | 72.9% | 0.6% |
| 0.55 | 16.1% | 98.1% | 7.2% | 72.3% | 0.4% |
| 0.60 | 15.9% | 98.5% | 7.3% | 71.8% | 0.3% |
| 0.65 | 15.7% | 98.6% | 7.5% | 70.9% | 0.3% |
| 0.70 | 15.4% | 98.9% | 7.8% | 69.8% | 0.2% |
| 0.75 | 15.3% | 99.1% | 7.9% | 69.5% | 0.2% |
| 0.80 | 15.1% | 99.6% | 8.0% | 68.9% | 0.1% |
| 0.85 | 14.9% | 99.7% | 8.2% | 68.0% | 0.1% |
| 0.90 | 14.8% | 99.9% | 8.3% | 67.5% | 0.0% |
| 0.95 | 14.4% | 100.0% | 8.8% | 65.7% | 0.0% |

### The two rules

- **REJECT** if probability >= **0.35**: the lowest cut-off that turns away at most 2% of good borrowers.
- **APPROVE** if probability < **0.10**: the highest cut-off at which REVIEW + REJECT still hold at least 90% of all defaulters.
- **REVIEW** in between (0.10 to 0.35): a credit officer checks the application.

### Other cut-off pairs (validation split)

| Review cut-off | Reject cut-off | APPROVE | REVIEW | REJECT | Default rate in APPROVE | Defaulters rejected | Good loans rejected |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.25 | 44.5% | 34.8% | 20.7% | 0.9% | 80.5% | 3.9% |
| 0.10 | 0.25 | 63.8% | 15.5% | 20.7% | 2.9% | 80.5% | 3.9% |
| 0.10 | 0.35 **(chosen)** | 63.8% | 18.4% | 17.8% | 2.9% | 75.4% | 1.7% |
| 0.10 | 0.50 | 63.8% | 19.8% | 16.4% | 2.9% | 72.9% | 0.6% |
| 0.15 | 0.35 | 72.0% | 10.2% | 17.8% | 3.9% | 75.4% | 1.7% |
| 0.20 | 0.50 | 76.3% | 7.3% | 16.4% | 4.6% | 72.9% | 0.6% |

### The chosen bands

| Band | Validation: share of loans | Validation: default rate | Test: share of loans | Test: default rate | Test: share of all defaulters | Test: share of all good loans |
|---|---|---|---|---|---|---|
| APPROVE | 63.8% | 2.9% | 62.1% | 2.9% | 8.4% | 77.2% |
| REVIEW | 18.4% | 19.4% | 20.3% | 17.1% | 15.8% | 21.5% |
| REJECT | 17.8% | 92.7% | 17.6% | 94.0% | 75.8% | 1.3% |

The bands behave the same on the test split as on the validation split, so the cut-offs were not tuned to
luck. The cut-offs are saved in `models/model_meta.json` (`decision_cutoffs`).

## 2. Business impact (test split, 4,862 loans, final saved model)

Baseline = **approve everyone**, which is what really happened to these loans (1,063 defaulted, 21.9%).

| | Approve everyone | A: REVIEW approved | B: REVIEW rejected |
|---|---|---|---|
| Loans approved | 4,862 | 4,005 | 3,020 |
| Defaulted loans approved | 1,063 | 257 | 89 |
| Defaulted amount approved (NPA) | $11,439,550 | $2,308,875 | $794,900 |
| **NPA reduction - number of defaults** | - | **75.8%** | **91.6%** |
| **NPA reduction - defaulted amount** | - | **79.8%** | **93.1%** |
| NPA ratio (defaulted $ / approved $) | 24.6% | 6.3% | 2.8% |
| Good loans turned away | 0 | 51 (1.3%) | 868 (22.8%) |
| Good loan amount turned away | $0 | $454,375 | $7,493,125 |
| Interest earned on good loans (1 year) | $3,716,379 | $3,647,866 | $2,750,223 |
| Loss on defaulted loans | $11,439,550 | $2,308,875 | $794,900 |
| **Net result** | **-$7,723,171** | **$1,338,991** | **$1,955,323** |

Money assumption (simple, stated on purpose): a defaulted loan loses its **full** amount (no recovery, no
collection), and a repaid loan earns `loan_amnt x loan_int_rate` for **one year** (no funding or operating
costs). Missing interest rates (9.5%) use the training-split median rate of the same grade. Real numbers would
differ - recoveries make defaults cheaper, multi-year loans earn more interest - but the comparison between
the policies is what matters.

## 3. In plain words

- Without the model the lender approved all 4,862 test loans and 1,063 of them defaulted ($11,439,550).
- **Scenario A** (only REJECT is turned away): 257 defaults are still approved - **75.8% fewer defaults** and **79.8% less defaulted money** - while only 51 good borrowers (1.3%) are turned away.
- **Scenario B** (REVIEW is also turned away): **93.1% less defaulted money**, but 868 good borrowers (22.8%) lose their loan. In practice a credit officer would approve some REVIEW loans, so the real result lies between A and B.
- With the money assumption, the portfolio goes from -$7,723,171 (approve everyone) to $1,338,991 (A) or $1,955,323 (B). B looks better only because of the assumption: turning away a good borrower costs one year of interest (about 11% of the loan), while approving a default costs the whole loan. With recoveries, multi-year interest or the cost of losing customers, the gap shrinks - which is why the REVIEW band goes to a person instead of being rejected automatically.

## 4. Honest comparison with the 30% NPA-reduction target

The synopsis target is a **30% NPA reduction**. The simulation gives **79.8%** (scenario A) and **93.1%** (scenario B) less defaulted money, so the target is met in both scenarios. Three reasons to read this carefully:

1. **The baseline is "approve everyone".** A real lender already screens applicants, so the reduction against a real bank's current process would be smaller.
2. **This dataset is easier than real life.** It contains a perfect rule (every renter whose loan is more than 30% of income defaulted - see `reports/model_comparison.md`); the model gets many defaults "for free".
3. **Simple money model** (no recoveries, one year of interest, no costs) - the net result is an illustration.

So: the 30% target is reached in this simulation, but the number should be presented as "on this dataset, versus approving everyone", not as a promise for a real bank.

Charts: `reports/figures/20_decision_bands.png`, `reports/figures/21_npa_reduction.png`.
