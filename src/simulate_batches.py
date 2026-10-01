"""
Simulated "current" batches for drift monitoring. The dataset has no dates, so we cannot watch it change over
time - instead we build two batches of new applications from the TEST split (its Step 4 metrics are already
final, so this reuse is only for the monitoring demo):

    no_drift : 2,000 random test loans, unchanged                     -> drift monitoring should stay quiet
    drift    : 2,000 test loans with realistic changes                 -> drift monitoring should raise the alarm
               - more DEBTCONSOLIDATION and MEDICAL loans (about 50% of the batch instead of about 35%)
               - interest rates +3 points (loan_int_rate + 3; missing values stay missing)
               - incomes -15% (person_income x 0.85), so loan_percent_income = loan_amnt / person_income again

Both batches use the 11 raw applicant fields (grade letter, Y/N), exactly what the API receives, plus loan_id.
Saved to data/simulated/ (git-ignored).

Run from the project folder:
    python src/simulate_batches.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clean_data import APPLICANT_FIELDS
from load_data import RANDOM_STATE, TARGET, get_splits, load_clean

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "data" / "simulated"
BATCH_SIZE = 2000
SHIFTED_INTENTS = ["DEBTCONSOLIDATION", "MEDICAL"]
TARGET_INTENT_SHARE = 0.50
RATE_SHIFT = 3.0          # interest rate points
INCOME_FACTOR = 0.85      # -15%


def raw_test_loans() -> pd.DataFrame:
    """The test split in the raw applicant format (grade letter, Y/N), with loan_id and the real outcome."""
    rows = load_clean().loc[get_splits()["X_test"].index]
    raw = rows[["loan_id", *APPLICANT_FIELDS, TARGET]].copy()
    raw["cb_person_default_on_file"] = raw["cb_person_default_on_file"].map({1: "Y", 0: "N"})
    return raw.reset_index(drop=True)


def intent_resample(loans: pd.DataFrame, n: int = BATCH_SIZE, share: float = TARGET_INTENT_SHARE,
                    seed: int = RANDOM_STATE) -> pd.DataFrame:
    """n loans where the SHIFTED_INTENTS make up `share` of the batch (sampled without replacement,
    no feature changes, real outcomes kept)."""
    rng = np.random.default_rng(seed)
    is_shifted = loans["loan_intent"].isin(SHIFTED_INTENTS).to_numpy()
    n_shifted = int(round(n * share))
    picked = np.concatenate([
        rng.choice(np.flatnonzero(is_shifted), n_shifted, replace=False),
        rng.choice(np.flatnonzero(~is_shifted), n - n_shifted, replace=False),
    ])
    return loans.iloc[rng.permutation(picked)].reset_index(drop=True)


def apply_shifts(batch: pd.DataFrame) -> pd.DataFrame:
    """Higher interest rates and lower incomes (loan_percent_income recomputed so the fields stay consistent)."""
    shifted = batch.copy()
    shifted["loan_int_rate"] = shifted["loan_int_rate"] + RATE_SHIFT            # NaN + 3 stays NaN
    shifted["person_income"] = (shifted["person_income"] * INCOME_FACTOR).round()
    shifted["loan_percent_income"] = (shifted["loan_amnt"] / shifted["person_income"]).round(2)
    return shifted


def make_batches(seed: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """{'no_drift': ..., 'drift': ...} - 11 raw fields + loan_id. The outcome is dropped: after the shifts the
    real outcomes would no longer match the features (src/retrain.py uses its own, unshifted sample)."""
    loans = raw_test_loans()
    no_drift = loans.sample(BATCH_SIZE, random_state=seed).reset_index(drop=True)
    drift = apply_shifts(intent_resample(loans, seed=seed))
    columns = ["loan_id", *APPLICANT_FIELDS]
    return {"no_drift": no_drift[columns], "drift": drift[columns]}


def describe(batch: pd.DataFrame) -> dict:
    return {
        "interest rate (mean %)": batch["loan_int_rate"].mean(),
        "yearly income (mean $)": batch["person_income"].mean(),
        "loan / income (mean)": batch["loan_percent_income"].mean(),
        "debt consolidation + medical (share)": batch["loan_intent"].isin(SHIFTED_INTENTS).mean(),
        "interest rate missing (share)": batch["loan_int_rate"].isna().mean(),
    }


def load_batch(name: str) -> pd.DataFrame:
    """A saved batch (made on the fly the first time)."""
    path = OUT_DIR / f"{name}_batch.csv"
    if not path.exists():
        save_batches(make_batches())
    return pd.read_csv(path)


def save_batches(batches: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, batch in batches.items():
        batch.to_csv(OUT_DIR / f"{name}_batch.csv", index=False)


def main() -> None:
    batches = make_batches()
    save_batches(batches)
    table = pd.DataFrame({"test split": describe(raw_test_loans()),
                          "no_drift batch": describe(batches["no_drift"]),
                          "drift batch": describe(batches["drift"])})
    print(f"Batches of {BATCH_SIZE:,} loans built from the test split and saved to "
          f"{OUT_DIR.relative_to(PROJECT_DIR)}/\n")
    print(table.map(lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}").to_string())


if __name__ == "__main__":
    main()
