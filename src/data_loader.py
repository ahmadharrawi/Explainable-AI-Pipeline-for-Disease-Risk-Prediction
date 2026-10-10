"""Download the two UCI datasets and cache them as CSV files."""
import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

DIABETES_ID = 529  # Early Stage Diabetes Risk Prediction
HEART_ID = 45      # Heart Disease (Cleveland)


def _fetch(dataset_id):
    """Return features + target of a UCI dataset as one DataFrame."""
    dataset = fetch_ucirepo(id=dataset_id)
    return pd.concat([dataset.data.features, dataset.data.targets], axis=1)


def load_diabetes_data():
    return _fetch(DIABETES_ID)


def load_heart_disease_data():
    return _fetch(HEART_ID)


def download_all(data_dir="data"):
    """Fetch both datasets, save them to `data_dir`, and return the DataFrames."""
    os.makedirs(data_dir, exist_ok=True)
    diabetes_df = load_diabetes_data()
    diabetes_df.to_csv(os.path.join(data_dir, "diabetes_data.csv"), index=False)
    heart_df = load_heart_disease_data()
    heart_df.to_csv(os.path.join(data_dir, "heart_disease_data.csv"), index=False)
    return diabetes_df, heart_df
