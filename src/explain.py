"""
SHAP explanations for the saved final model.

The final model is a CalibratedClassifierCV: an average of 3 (XGBoost pipeline + isotonic calibration) pairs.
SHAP's TreeExplainer cannot read the calibration, so we explain the 3 inner XGBoost pipelines:
    - each inner pipeline transforms the rows with its OWN preprocessor
    - TreeExplainer gives SHAP values in log-odds of that (uncalibrated) XGBoost model
    - the 3 results are averaged, and the 20 transformed columns are added back to the 11 original features
      (person_home_ownership_* -> person_home_ownership, loan_intent_* -> loan_intent,
       loan_int_rate_missing -> loan_int_rate)
So: base value + sum of contributions = the average log-odds of the 3 inner models. The probability shown to
users comes from the calibrated model. Calibration only re-scales the score (higher log-odds -> higher
probability), so the reasons keep their meaning and direction.

Usage (the API loads the explainer once):
    from explain import explain_one, reason_text
    result = explain_one(applicant)          # applicant = dict with the 11 raw fields
    reason_text(result["reasons"])           # ["Loan is 45% of yearly income - increases risk", ...]

Run directly for the global charts and reports/shap_insights.md:
    python src/explain.py
"""

import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from clean_data import to_model_input
from decision import decision_band
from features import get_feature_names
from load_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "loan_default_model.joblib"
META_PATH = PROJECT_DIR / "models" / "model_meta.json"
TOP_N = 5
GLOBAL_SAMPLE = 2000

FEATURE_LABELS = {
    "person_age": "Age",
    "person_income": "Yearly income",
    "person_emp_length": "Years employed",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as % of income",
    "loan_grade_num": "Loan grade",
    "cb_person_default_on_file": "Past default on file",
    "cb_person_cred_hist_length": "Credit history length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
}
# Which raw applicant field holds the value to show for each model feature
RAW_FIELD = {f: f for f in FEATURES} | {"loan_grade_num": "loan_grade"}

# A real repaid loan from the data (loan_id 6560) - used to warm up the explainers and as an API example
EXAMPLE_APPLICANT = {
    "person_age": 22, "person_income": 48000, "person_home_ownership": "RENT", "person_emp_length": 6,
    "loan_intent": "VENTURE", "loan_grade": "A", "loan_amnt": 6000, "loan_int_rate": 7.49,
    "loan_percent_income": 0.13, "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 4,
}

HOME_TEXT = {"RENT": "Rents home", "OWN": "Owns home", "MORTGAGE": "Has a mortgage", "OTHER": "Other housing"}
INTENT_TEXT = {"DEBTCONSOLIDATION": "debt consolidation", "HOMEIMPROVEMENT": "home improvement",
               "EDUCATION": "education", "MEDICAL": "medical bills", "VENTURE": "a business venture",
               "PERSONAL": "personal use"}


def original_feature(column: str) -> str:
    """Model column after preprocessing -> one of the 11 original features."""
    if column in FEATURES:
        return column
    if column.endswith("_missing") and column.removesuffix("_missing") in FEATURES:
        return column.removesuffix("_missing")
    for cat in CATEGORICAL_FEATURES:
        if column.startswith(cat + "_"):
            return cat
    raise ValueError(f"cannot map model column {column!r} to an original feature")


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


class LoanExplainer:
    """Loads the saved model once and builds one TreeExplainer per inner XGBoost pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
        self.model = joblib.load(model_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.cutoffs = self.meta["decision_cutoffs"]
        # Calibrated model -> its 3 inner pipelines; a plain pipeline -> itself
        pipelines = ([c.estimator for c in self.model.calibrated_classifiers_]
                     if hasattr(self.model, "calibrated_classifiers_") else [self.model])
        self.parts = []
        example = to_model_input(EXAMPLE_APPLICANT)
        for pipe in pipelines:
            explainer = shap.TreeExplainer(pipe.named_steps["model"])
            # shap 0.52 quirk: expected_value is only correct after the first shap_values() call -> warm up once
            explainer.shap_values(pipe.named_steps["preprocess"].transform(example))
            self.parts.append({
                "preprocess": pipe.named_steps["preprocess"],
                "xgb": pipe.named_steps["model"],
                "explainer": explainer,
                "to_feature": [original_feature(c) for c in get_feature_names(pipe)],
            })
        # base value = average log-odds the explanation starts from (before any feature is known)
        self.base_value = float(np.mean([float(p["explainer"].expected_value) for p in self.parts]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default (what users see)."""
        return self.model.predict_proba(X)[:, 1]

    def log_odds(self, X: pd.DataFrame) -> np.ndarray:
        """Average log-odds of the inner XGBoost models (what the SHAP values add up to)."""
        return np.mean([p["xgb"].predict(p["preprocess"].transform(X), output_margin=True) for p in self.parts], axis=0)

    def shap_by_feature(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds) per original feature: one row per loan, 11 columns, averaged over the inner models."""
        total = pd.DataFrame(0.0, index=X.index, columns=FEATURES)
        for p in self.parts:
            values = p["explainer"].shap_values(p["preprocess"].transform(X))
            per_column = pd.DataFrame(values, index=X.index, columns=p["to_feature"])
            total += per_column.T.groupby(level=0).sum().T.reindex(columns=FEATURES, fill_value=0.0)
        return total / len(self.parts)

    def explain(self, applicant: dict, top_n: int = TOP_N) -> dict:
        X = to_model_input(applicant)
        probability = float(self.predict_proba(X)[0])
        contributions = self.shap_by_feature(X).iloc[0]
        order = contributions.abs().sort_values(ascending=False).index
        reasons = []
        for feature in order[:top_n]:
            value = applicant.get(RAW_FIELD[feature])
            reasons.append({
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "value": None if _is_missing(value) else value,
                "contribution": round(float(contributions[feature]), 4),
                "direction": "increases risk" if contributions[feature] > 0 else "decreases risk",
            })
        return {
            "probability": round(probability, 4),
            "decision": decision_band(probability, self.cutoffs),
            "cutoffs": {"review": self.cutoffs["review"], "reject": self.cutoffs["reject"]},
            "base_value": round(self.base_value, 4),
            "log_odds": round(float(self.base_value + contributions.sum()), 4),
            "reasons": reasons,
            "model_version": self.meta["version"],
        }


@lru_cache(maxsize=1)
def get_explainer() -> LoanExplainer:
    """The explainer is created once (first call) and reused afterwards - the API calls this at startup."""
    return LoanExplainer()


def explain_one(applicant: dict, top_n: int = TOP_N) -> dict:
    """Probability, decision band and the top reasons for one applicant (dict with the 11 raw fields)."""
    return get_explainer().explain(applicant, top_n)


def _describe(feature: str, value) -> str:
    """Short, human phrase for one feature value, e.g. 'Loan is 45% of yearly income'."""
    if _is_missing(value):
        return f"{FEATURE_LABELS[feature]} not given"
    v = value.upper() if isinstance(value, str) else value
    return {
        "person_age": lambda: f"Age {float(v):.0f}",
        "person_income": lambda: f"Yearly income of ${float(v):,.0f}",
        "person_emp_length": lambda: f"{float(v):.0f} year{'s' if float(v) != 1 else ''} employed",
        "loan_amnt": lambda: f"Loan amount of ${float(v):,.0f}",
        "loan_int_rate": lambda: f"Interest rate of {float(v):.1f}%",
        "loan_percent_income": lambda: f"Loan is {float(v):.0%} of yearly income",
        "loan_grade_num": lambda: f"Grade {v} loan",
        "cb_person_default_on_file": lambda: "Past default on file" if v in ("Y", 1, "1") else "No past default on file",
        "cb_person_cred_hist_length": lambda: f"{float(v):.0f}-year credit history",
        "person_home_ownership": lambda: HOME_TEXT.get(v, f"Home ownership {v}"),
        "loan_intent": lambda: f"Loan for {INTENT_TEXT.get(v, str(v).lower())}",
    }[feature]()


def reason_text(reasons: list[dict]) -> list[str]:
    """Top reasons as short sentences: 'Grade D loan - increases risk'."""
    return [f"{_describe(r['feature'], r['value'])} - {r['direction']}" for r in reasons]


# ---------------------------------------------------------------------------
# Global explanations (run as a script)
# ---------------------------------------------------------------------------

# What each feature means for lending (used in reports/shap_insights.md)
LENDING_MEANING = {
    "loan_percent_income": "How stretched the borrower is. It works like a switch: up to about 30% of income it hardly "
                           "matters, above 30% it is one of the strongest pushes towards default - especially for "
                           "renters (the 'perfect rule' from Step 3).",
    "loan_grade_num": "The lender's own risk grade. Grades D-G push risk up strongly, A-B pull it down - the model "
                      "confirms the grading but adds the borrower's situation on top.",
    "person_income": "Low income means little room for repayments; high income protects against default even for "
                     "larger loans.",
    "person_home_ownership": "Renters are much riskier than owners or mortgage holders - housing costs and stability. "
                             "Combined with a high loan-to-income ratio it is the 'perfect rule' found in Step 3.",
    "loan_int_rate": "Higher rates mean higher monthly payments and reflect risk the lender already priced in.",
    "loan_intent": "Debt consolidation, medical and home-improvement loans are riskier than education or "
                   "venture loans - the purpose tells something about financial pressure.",
    "person_emp_length": "Longer employment means a more stable income.",
    "loan_amnt": "Matters mostly through loan-to-income; on its own the amount says less.",
    "cb_person_default_on_file": "Almost no effect in the model even though past defaulters default twice as often "
                                 "(EDA): the information is already in the grade (no borrower with a past default "
                                 "got grade A or B), so the model uses the grade instead.",
    "person_age": "Weak on its own; small effects, partly through the link with credit history.",
    "cb_person_cred_hist_length": "Weakest feature - longer histories help only slightly.",
}


def direction_summary(feature: str, X: pd.DataFrame, shap_df: pd.DataFrame) -> str:
    """How the feature pushes risk, read from the SHAP values themselves."""
    s = shap_df[feature]
    if feature in CATEGORICAL_FEATURES:
        by_cat = s.groupby(X[feature]).mean().sort_values()
        return (f"lowest risk: {by_cat.index[0]} ({by_cat.iloc[0]:+.2f}), "
                f"highest risk: {by_cat.index[-1]} ({by_cat.iloc[-1]:+.2f})")
    if feature == "cb_person_default_on_file":
        by_val = s.groupby(X[feature]).mean()
        return f"past default {by_val.get(1.0, np.nan):+.2f}, none {by_val.get(0.0, np.nan):+.2f}"
    # Average push for the lowest 20% and the highest 20% of values (missing values left out)
    v = X[feature].dropna()
    low, high = s[v.index][v <= v.quantile(0.2)].mean(), s[v.index][v >= v.quantile(0.8)].mean()
    word = "higher value -> higher risk" if high > low else "higher value -> lower risk"
    return f"{word} (lowest 20% of values {low:+.2f}, highest 20% {high:+.2f})"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from load_data import get_splits
    from train import FIG, INK, INK2, REPORTS   # importing train also applies the project chart style

    ex = get_explainer()
    X = get_splits()["X_test"].sample(GLOBAL_SAMPLE, random_state=42)
    shap_df = ex.shap_by_feature(X)

    # Check: contributions + base value = average log-odds of the inner models
    gap = np.abs(ex.base_value + shap_df.sum(axis=1).to_numpy() - ex.log_odds(X)).max()
    print(f"Additivity check on {len(X):,} test rows: max error {gap:.2e}")

    importance = shap_df.abs().mean().sort_values(ascending=False)
    share = importance / importance.sum()

    # 22 - beeswarm (shap's own plot). Categorical features have no numeric value -> grey points.
    data = X[FEATURES].copy()
    data[CATEGORICAL_FEATURES] = np.nan
    explanation = shap.Explanation(values=shap_df[FEATURES].to_numpy(), base_values=np.full(len(X), ex.base_value),
                                   data=data.astype(float).to_numpy(), feature_names=[FEATURE_LABELS[f] for f in FEATURES])
    plt.figure()
    with warnings.catch_warnings():   # all-NaN colour values for the categorical rows are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        shap.plots.beeswarm(explanation, max_display=len(FEATURES), show=False, plot_size=(9, 6))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("SHAP value (log-odds): right = pushes towards default, left = towards repaid")
    ax.set_title(f"How each feature moves the risk - {len(X):,} test loans", loc="left", fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Colour = feature value (red high, blue low); grey = category or missing value",
             fontsize=8.5, color=INK2)
    fig.savefig(FIG / "22_shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close(fig)

    # 23 - mean |SHAP| per original feature
    fig, ax = plt.subplots(figsize=(8.5, 5))
    order = importance.sort_values()
    ax.barh([FEATURE_LABELS[f] for f in order.index], order.values, color="#2a78d6", height=0.6)
    for i, f in enumerate(order.index):
        ax.text(order[f] + importance.max() * 0.01, i, f"{order[f]:.2f}  ({share[f]:.0%})", va="center",
                fontsize=9, color=INK2)
    ax.set_xlim(0, importance.max() * 1.22)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (log-odds) - average size of the push, either direction")
    ax.set_title("Feature importance from SHAP (one-hot columns added back together)")
    fig.savefig(FIG / "23_shap_importance_bar.png")
    plt.close(fig)

    # reports/shap_insights.md + csv
    table = pd.DataFrame({
        "rank": range(1, len(importance) + 1),
        "feature": importance.index,
        "label": [FEATURE_LABELS[f] for f in importance.index],
        "mean_abs_shap": importance.values,
        "share": share.values,
        "direction": [direction_summary(f, X, shap_df) for f in importance.index],
    })
    table.to_csv(REPORTS / "shap_importance.csv", index=False)
    top3 = share.iloc[:3].sum()
    stretched = X["loan_percent_income"] > 0.30
    renter = X["person_home_ownership"] == "RENT"
    lpi = shap_df["loan_percent_income"]
    home = shap_df["person_home_ownership"].groupby(X["person_home_ownership"]).mean()
    lines = [
        "# SHAP insights - what drives the model",
        "",
        f"Made by `python src/explain.py` on a random sample of {len(X):,} test loans (random_state=42).",
        "SHAP values are in log-odds of the 3 inner XGBoost models of the calibrated final model, averaged; one-hot",
        "and missing-flag columns are added back to their original feature. Additivity check: base value + SHAP",
        f"values = average model log-odds (max error {gap:.1e}). Base value (average log-odds): {ex.base_value:.3f}.",
        "",
        "## Features ranked by importance",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| | Share | How it pushes risk | What it means for lending |",
        "|---|---|---|---|---|---|",
        *[f"| {r.rank} | {r.label} (`{r.feature}`) | {r.mean_abs_shap:.3f} | {r.share:.1%} | {r.direction} | "
          f"{LENDING_MEANING[r.feature]} |" for r in table.itertuples()],
        "",
        "## Key points",
        "",
        f"- The top 3 features ({', '.join(FEATURE_LABELS[f] for f in importance.index[:3])}) carry {top3:.0%} of the "
        f"total SHAP importance; {FEATURE_LABELS[importance.index[3]]} is next ({share.iloc[3]:.0%}). These match the "
        "EDA (charts 03, 04, 07, 08) and the step patterns that made tree models win in Step 3.",
        f"- **Loan as % of income is a switch, not a slope:** average SHAP {lpi[~stretched].mean():+.2f} for loans up to "
        f"30% of income, {lpi[stretched].mean():+.2f} above 30% - and {lpi[stretched & renter].mean():+.2f} for renters "
        "above 30% (the perfect-rule group).",
        f"- **Home ownership:** owning a home pulls risk down strongly ({home.get('OWN', np.nan):+.2f}), renting pushes it "
        f"up ({home.get('RENT', np.nan):+.2f}). SHAP values are in log-odds of the uncalibrated model, whose scores range "
        "widely, so single values can look large; compare them with each other, not as probabilities.",
        "- **Past default on file** has almost no weight: the grade already contains it.",
        "- Age and credit history length matter least, as the single-feature AUCs (0.522 and 0.519) suggested.",
        "- SHAP explains the uncalibrated XGBoost score (log-odds). The probability shown to users comes from the "
        "calibrated model; calibration keeps the order of risk, so the reasons keep their direction.",
        "- Every API prediction stores its top 5 reasons, so a loan officer can see *why* an application was flagged.",
        "",
        "Charts: `reports/figures/22_shap_summary_beeswarm.png`, `reports/figures/23_shap_importance_bar.png`.",
        "",
    ]
    (REPORTS / "shap_insights.md").write_text("\n".join(lines))
    print(table.round(3).to_string(index=False))
    print("Saved charts 22-23, reports/shap_insights.md and reports/shap_importance.csv")


if __name__ == "__main__":
    main()
