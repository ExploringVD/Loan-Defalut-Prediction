"""
Preprocessing (imputation, scaling, encoding) as an unfitted scikit-learn ColumnTransformer.

The preprocessor is always put inside a Pipeline together with the model, so it is fitted on the
training data only (no leakage) and the same saved pipeline is reused by the API and the drift checks.

Usage:
    from features import build_preprocessor, get_feature_names

    pipe = Pipeline([("preprocess", build_preprocessor("linear")), ("model", LogisticRegression())])
    pipe.fit(X_train, y_train)
    names = get_feature_names(pipe)     # readable names, e.g. for SHAP plots

Two kinds:
    "linear"  for Logistic Regression: log1p on skewed columns, median imputation + missing-value
              indicators, StandardScaler, one-hot encoding, and one column of each highly
              correlated pair dropped (correlated inputs make linear coefficients unstable).
    "tree"    for Random Forest / XGBoost: median imputation (+ indicator for loan_int_rate),
              no scaling and no log (trees only care about the order of values), one-hot encoding.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from load_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Long right tail -> log1p makes them closer to normal (all values are >= 0)
SKEWED_FEATURES = ["person_income", "loan_amnt"]

# Highly correlated pairs (see EDA). For the linear model we keep the first and drop the second:
#   loan_grade_num ~ loan_int_rate (0.93), person_age ~ cb_person_cred_hist_length (0.86)
LINEAR_DROP_FEATURES = ["loan_int_rate", "cb_person_cred_hist_length"]

# For trees only this column gets a "was missing" flag: 9.6% missing, the most of any column
TREE_INDICATOR_FEATURES = ["loan_int_rate"]

# Categories seen fewer times than this in training are grouped into one "other" column
MIN_CATEGORY_COUNT = 20


def _one_hot() -> OneHotEncoder:
    # handle_unknown="ignore": a category never seen in training becomes all zeros instead of an error
    return OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_CATEGORY_COUNT, sparse_output=False)


def _linear_preprocessor(drop_features: list[str]) -> ColumnTransformer:
    numeric = [c for c in NUMERIC_FEATURES if c not in drop_features]
    skewed = [c for c in numeric if c in SKEWED_FEATURES]
    plain = [c for c in numeric if c not in SKEWED_FEATURES]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in drop_features]

    skewed_pipe = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    plain_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    # Columns not listed in any transformer (the dropped ones) are removed by remainder="drop"
    return ColumnTransformer(
        [("skewed", skewed_pipe, skewed), ("num", plain_pipe, plain), ("cat", _one_hot(), categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _tree_preprocessor(drop_features: list[str]) -> ColumnTransformer:
    numeric = [c for c in NUMERIC_FEATURES if c not in drop_features]
    flagged = [c for c in numeric if c in TREE_INDICATOR_FEATURES]
    plain = [c for c in numeric if c not in TREE_INDICATOR_FEATURES]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in drop_features]

    return ColumnTransformer(
        [
            ("flagged", SimpleImputer(strategy="median", add_indicator=True), flagged),
            ("num", SimpleImputer(strategy="median"), plain),
            ("cat", _one_hot(), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor(kind: str, drop_features: list[str] | None = None) -> ColumnTransformer:
    """Return an UNFITTED preprocessor. Put it in a Pipeline with the model and fit on training data only.

    kind: "linear" or "tree".
    drop_features: input columns to leave out. Default: LINEAR_DROP_FEATURES for "linear", none for "tree".
    """
    if kind == "linear":
        drop = LINEAR_DROP_FEATURES if drop_features is None else drop_features
        return _linear_preprocessor(list(drop))
    if kind == "tree":
        return _tree_preprocessor(list(drop_features or []))
    raise ValueError(f'kind must be "linear" or "tree", got {kind!r}')


def get_feature_names(fitted) -> list[str]:
    """Readable output column names of a fitted preprocessor (or of a Pipeline whose first step is one).

    Examples: "loan_int_rate", "person_emp_length_missing", "loan_intent_MEDICAL",
    "person_home_ownership_other".
    """
    if isinstance(fitted, Pipeline):
        fitted = fitted.steps[0][1]
    names = []
    for name in fitted.get_feature_names_out():
        if name.startswith("missingindicator_"):          # sklearn: "missingindicator_loan_int_rate"
            name = name.removeprefix("missingindicator_") + "_missing"
        name = name.replace("infrequent_sklearn", "other")  # sklearn: "loan_intent_infrequent_sklearn"
        names.append(name)
    return names
