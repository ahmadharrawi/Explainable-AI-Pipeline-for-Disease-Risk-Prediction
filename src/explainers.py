"""SHAP and LIME helpers shared by the notebook and the Streamlit app.

Both explainers work on the model's positive-class probability and are
model-agnostic, so they work for the calibrated forest and the logistic
regression pipeline alike. Inputs are always DataFrames of RAW (unscaled)
features in the same column order used for training.
"""
import time

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

SEED = 42


def make_predict_fn(model, feature_names):
    """f(array) -> P(positive); restores column names to avoid sklearn warnings."""
    names = list(feature_names)

    def predict_positive(data):
        return model.predict_proba(pd.DataFrame(np.asarray(data), columns=names))[:, 1]

    return predict_positive


def make_predict_proba_fn(model, feature_names):
    """f(array) -> [P(negative), P(positive)] for LIME."""
    names = list(feature_names)

    def predict_proba(data):
        return model.predict_proba(pd.DataFrame(np.asarray(data), columns=names))

    return predict_proba


def build_shap_explainer(model, X_train, max_background=100):
    names = list(X_train.columns)
    background = shap.sample(X_train, min(max_background, len(X_train)), random_state=SEED)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.Explainer(make_predict_fn(model, names), masker, feature_names=names)


def generate_shap_explanations(model, X_train, X_test, feature_names=None):
    """Global SHAP values for X_test.

    Returns (explainer, shap_values, values_array, seconds).
    `shap_values` is a shap.Explanation; `values_array` has shape (n, features)
    and is what shap.summary_plot expects.
    """
    names = list(feature_names) if feature_names is not None else list(X_train.columns)
    X_tr = pd.DataFrame(X_train, columns=names)
    X_te = pd.DataFrame(X_test, columns=names)

    start = time.time()
    explainer = build_shap_explainer(model, X_tr)
    shap_values = explainer(X_te)
    elapsed = time.time() - start
    return explainer, shap_values, shap_values.values, elapsed


def build_lime_explainer(X_train, class_names=("Negative", "Positive")):
    names = list(X_train.columns)
    # Binary 0/1 columns are treated as categorical so LIME does not bin them.
    categorical = [i for i, c in enumerate(names) if set(np.unique(X_train[c])) <= {0, 1}]
    return LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=names,
        class_names=list(class_names),
        categorical_features=categorical,
        mode="classification",
        random_state=SEED,
    )


def generate_lime_explanation(model, X_train, instance, feature_names=None,
                              class_names=("Negative", "Positive"), num_features=10):
    """LIME explanation for one patient. Returns (explanation, seconds)."""
    names = list(feature_names) if feature_names is not None else list(X_train.columns)
    X_tr = pd.DataFrame(X_train, columns=names)

    start = time.time()
    explainer = build_lime_explainer(X_tr, class_names)
    exp = explainer.explain_instance(
        data_row=np.asarray(instance, dtype=float).ravel(),
        predict_fn=make_predict_proba_fn(model, names),
        num_features=num_features,
    )
    return exp, time.time() - start
