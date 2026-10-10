"""Train, compare and save the models for one disease module."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_models():
    return {
        # Trees are scale-invariant, so no scaler is needed for the forest.
        "Calibrated RF": CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), cv=3
        ),
        # Scaling lives inside the pipeline, so raw inputs can be passed in.
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }


def baseline_profile(X_train):
    """Reference input: binary features = 0 (absent), other features = median.

    For the diabetes module this is "no symptoms, median age"; the predicted
    risk should be low. It is a sanity check, not a clinical profile.
    """
    row = {}
    for col in X_train.columns:
        is_binary = set(np.unique(X_train[col])) <= {0, 1}
        row[col] = 0.0 if is_binary else float(X_train[col].median())
    return pd.DataFrame([row], columns=X_train.columns)


def train_and_evaluate_all(X_train, y_train, X_test, y_test, disease_prefix="diabetes"):
    """Compare the models, pick the best one with cross-validation, and save it.

    The winner is chosen by 5-fold stratified CV ROC-AUC on the TRAINING data
    only. The held-out test set is used just for the final reported metrics.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, trained = [], {}

    for name, model in build_models().items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        results.append({
            "Model": name,
            "CV ROC-AUC": cv_auc,
            "Test ROC-AUC": roc_auc_score(y_test, probs),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "Brier": brier_score_loss(y_test, probs),
        })
        trained[name] = model

    res_df = pd.DataFrame(results)
    best_name = res_df.sort_values("CV ROC-AUC", ascending=False).iloc[0]["Model"]
    best_model = trained[best_name]

    base_risk = best_model.predict_proba(baseline_profile(X_train))[0][1]
    print(f"[{disease_prefix.upper()}] Best model (by CV ROC-AUC): {best_name} | "
          f"Baseline-profile risk: {base_risk:.2%}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(best_model, f"models/{disease_prefix}_best_model.pkl")
    joblib.dump(list(X_train.columns), f"models/{disease_prefix}_features.pkl")
    res_df.to_csv(f"results/{disease_prefix}_metrics.csv", index=False)
    return res_df, trained
