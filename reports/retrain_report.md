# Retraining check

Made by `python src/retrain.py` on 2026-10-01. Production model: version 1.

**This demonstrates the mechanism.** The official model metrics stay the Step 4 test numbers (`reports/final_model.md`).

| | |
|---|---|
| New labelled loans | 2,000 test loans, intent-resampled like the drift batch (50% debt consolidation + medical) but with real features and real outcomes |
| Added to training | 1,000 (old training data: 27,547 loans -> 28,547) |
| Hold-out for the comparison | 1,000 loans (22.9% defaulted) |
| Production model ROC-AUC on the hold-out | 0.9596 |
| Candidate model ROC-AUC on the hold-out | 0.9609 (+0.0013) |
| Rule | replace only if the candidate is not worse |
| **Decision** | **REPLACE** |
| Carried out? | no (demo run without --apply) |

Why the demo does not replace the model by default: the new loans come from the test split. A model trained on
them would make the Step 4 test metrics, the business simulation and the SHAP report (all on the test split)
describe a different model. Run `python src/retrain.py --apply` to carry the decision out (a backup is kept;
`python src/retrain.py --restore <backup folder>` puts the old model back).

A difference of 0.0013 AUC on 1,000 loans is small - well within normal sampling noise -
so in practice both models are equally good here; the rule simply prefers the newer data when it is not worse.
