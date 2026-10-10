import os
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.explainers import (build_lime_explainer, build_shap_explainer,  # noqa: E402
                            make_predict_proba_fn)
from src.preprocess import preprocess_diabetes_data, preprocess_heart_data  # noqa: E402

st.set_page_config(page_title="Explainable AI for Transparent Medical Disease Classification",
                   layout="wide")

CONDITIONS = {
    "Diabetes": {
        "prefix": "diabetes",
        "csv": "data/diabetes_data.csv",
        "preprocess": preprocess_diabetes_data,
    },
    "Heart Disease": {
        "prefix": "heart",
        "csv": "data/heart_disease_data.csv",
        "preprocess": preprocess_heart_data,
    },
}

YES_NO = {"No": 0, "Yes": 1}

# Cleveland heart-disease fields: label, and value mapping for categorical ones.
# Codes follow the UCI documentation of the dataset.
HEART_FIELDS = {
    "age": ("Age (years)", None),
    "sex": ("Sex", {"Female": 0, "Male": 1}),
    "cp": ("Chest pain type", {"Typical angina": 1, "Atypical angina": 2,
                               "Non-anginal pain": 3, "Asymptomatic": 4}),
    "trestbps": ("Resting blood pressure (mm Hg)", None),
    "chol": ("Serum cholesterol (mg/dl)", None),
    "fbs": ("Fasting blood sugar > 120 mg/dl", YES_NO),
    "restecg": ("Resting ECG result", {"Normal": 0, "ST-T wave abnormality": 1,
                                       "Left ventricular hypertrophy": 2}),
    "thalach": ("Maximum heart rate achieved (bpm)", None),
    "exang": ("Exercise-induced angina", YES_NO),
    "oldpeak": ("ST depression induced by exercise", None),
    "slope": ("Slope of peak exercise ST segment", {"Upsloping": 1, "Flat": 2,
                                                    "Downsloping": 3}),
    "ca": ("Major vessels coloured by fluoroscopy (0-3)", {"0": 0, "1": 1, "2": 2, "3": 3}),
    "thal": ("Thalassemia", {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}),
}


@st.cache_resource
def load_resources(condition):
    """Load the saved model plus the training data used to build the explainers."""
    cfg = CONDITIONS[condition]
    model = joblib.load(f"models/{cfg['prefix']}_best_model.pkl")
    feature_names = joblib.load(f"models/{cfg['prefix']}_features.pkl")
    raw_df = pd.read_csv(cfg["csv"])
    # Re-run the exact training preprocessing so explainers see the same data
    X_train = cfg["preprocess"](raw_df)[0][feature_names]
    return {
        "model": model,
        "features": feature_names,
        "X_train": X_train,
        "shap": build_shap_explainer(model, X_train),
        "lime": build_lime_explainer(X_train, class_names=("Negative", "Positive")),
    }


def pretty(name):
    return name.replace("_", " ").capitalize()


def collect_inputs(condition, feature_names, X_train):
    """Render sidebar widgets and return one row of RAW feature values."""
    values = {}
    for col in feature_names:
        median = float(X_train[col].median())
        if condition == "Diabetes":
            if col == "age":
                values[col] = st.sidebar.number_input("Age (years)", 1, 120, int(round(median)))
            else:
                values[col] = YES_NO[st.sidebar.selectbox(pretty(col), list(YES_NO))]
        else:
            label, options = HEART_FIELDS.get(col, (pretty(col), None))
            if options is None:  # numeric input
                step = 0.1 if col == "oldpeak" else 1.0
                values[col] = st.sidebar.number_input(
                    label, min_value=0.0, value=median, step=step)
            else:
                values[col] = options[st.sidebar.selectbox(label, list(options))]
    return pd.DataFrame([values], columns=feature_names).astype(float)


st.title("Explainable AI for Transparent Medical Disease Classification")
st.warning("Educational prototype only - not a diagnostic tool and not medical advice. "
           "The models were trained on small public datasets and describe EXISTING "
           "disease status in those datasets, not whether a person will develop a disease.")

condition = st.sidebar.selectbox("Condition", list(CONDITIONS))
res = load_resources(condition)
model, feature_names, X_train = res["model"], res["features"], res["X_train"]

st.sidebar.header("Patient input")
input_df = collect_inputs(condition, feature_names, X_train)

prob = float(model.predict_proba(input_df)[0][1])
st.header(f"Classification result: {condition}")
st.metric("Estimated probability of a positive classification", f"{prob:.2%}")
if prob >= 0.5:
    st.error("Classified as POSITIVE (disease pattern present in the model's training data)")
else:
    st.success("Classified as NEGATIVE")

st.header("Why? Feature contributions")
tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

with tab_shap:
    start = time.time()
    sv = res["shap"](input_df)
    shap_secs = time.time() - start
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(plt.gcf())
    plt.close("all")
    st.caption(f"Red bars push the probability up, blue bars push it down. "
               f"Computed in {shap_secs:.2f}s.")

with tab_lime:
    start = time.time()
    exp = res["lime"].explain_instance(
        input_df.values[0], make_predict_proba_fn(model, feature_names), num_features=8)
    lime_secs = time.time() - start
    st.pyplot(exp.as_pyplot_figure())
    plt.close("all")
    st.caption(f"Green bars support a positive classification, red bars a negative one. "
               f"Computed in {lime_secs:.2f}s. LIME is a local approximation and can vary "
               f"slightly between runs.")

st.caption("Explanations show what the model relied on; they do not prove medical "
           "cause or clinical correctness.")
