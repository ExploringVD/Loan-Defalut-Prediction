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
    "tree"    for Random Forest / XGBoost: median imputation (+ indicator for mths_since_last_delinq),
              no scaling and no log (trees only care about the order of values), one-hot encoding.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from load_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Long right tail -> log1p makes them closer to normal (all values are >= 0)
SKEWED_FEATURES = [
    "annual_inc", "revol_bal", "loan_to_income", "installment_to_income", "revol_bal_to_income",
]

# Highly correlated pairs (see EDA). For the linear model we keep the first and drop the second:
#   int_rate ~ sub_grade_num (0.96), loan_amnt ~ installment (0.93),
#   loan_to_income ~ installment_to_income (0.93), pub_rec ~ pub_rec_bankruptcies (0.84)
LINEAR_DROP_FEATURES = ["sub_grade_num", "installment", "installment_to_income", "pub_rec_bankruptcies"]

# For trees only this column gets a "was missing" flag: 64% missing, and "never delinquent" is information
TREE_INDICATOR_FEATURES = ["mths_since_last_delinq"]

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

    Examples: "int_rate", "emp_length_missing", "purpose_credit_card", "addr_state_other".
    """
    if isinstance(fitted, Pipeline):
        fitted = fitted.steps[0][1]
    names = []
    for name in fitted.get_feature_names_out():
        if name.startswith("missingindicator_"):          # sklearn: "missingindicator_emp_length"
            name = name.removeprefix("missingindicator_") + "_missing"
        name = name.replace("infrequent_sklearn", "other")  # sklearn: "addr_state_infrequent_sklearn"
        names.append(name)
    return names
