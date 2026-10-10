"""Preprocessing for both modules.

Both functions return (X_train, X_test, y_train, y_test) as pandas objects.
Features are NOT scaled here: scaling is done inside the Logistic Regression
pipeline (see train.py), so the saved models accept raw feature values and the
Streamlit app can pass user input straight to them. Missing values are filled
with medians computed on the training split only (no test-set leakage).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.20
RANDOM_STATE = 42

YES_NO = {"yes": 1, "no": 0}
DIABETES_TARGET_MAP = {"positive": 1, "negative": 0}


def _split_and_impute(X, y):
    """Stratified 80/20 split, then impute with training medians only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    return X_train, X_test, y_train, y_test


def _strict_map(series, mapping, name):
    """Map text values case-insensitively and fail loudly on unknown values."""
    mapped = series.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unexpected values in '{name}': {bad}")
    return mapped.astype(int)


def preprocess_diabetes_data(df):
    df = df.copy()
    target_col = [c for c in df.columns if c.lower() in ("class", "target")][0]
    df = df.dropna(subset=[target_col])

    # Gender is dropped on purpose: in this dataset it is strongly confounded
    # with the label (a sampling artefact), so keeping it would let the model
    # and its explanations rely on a non-clinical attribute.
    gender_cols = [c for c in df.columns if c.lower() == "gender"]
    X = df.drop(columns=[target_col] + gender_cols)
    y = _strict_map(df[target_col], DIABETES_TARGET_MAP, target_col)

    # Yes/No text columns -> 1/0 (age is already numeric and is left alone)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = _strict_map(X[col], YES_NO, col)

    return _split_and_impute(X, y)


def preprocess_heart_data(df):
    df = df.copy()
    target_col = "num" if "num" in df.columns else df.columns[-1]
    df = df.dropna(subset=[target_col])

    # num: 0 = no disease, 1-4 = disease present -> binary target
    y = (pd.to_numeric(df[target_col]) > 0).astype(int)
    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

    # ca / thal contain a few missing values; they are imputed after the split
    return _split_and_impute(X, y)
