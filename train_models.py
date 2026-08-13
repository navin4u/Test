"""
train_models.py
----------------
Trains 5 classification models on the UCI "Default of Credit Card Clients"
dataset, evaluates each using Accuracy, AUC, Precision, Recall, F1, and MCC,
and saves:
  - trained model objects (.pkl) to model/
  - the fitted scaler (.pkl) to model/
  - test_data.csv (held-out test split, features + true label) to project root
  - a metrics comparison table (metrics.csv) to project root

Source data: model/raw_source_data.csv (UCI ID 350 — "default of credit card
clients"; original source: UCI Machine Learning Repository / Yeh & Lien, 2009).
Because the full dataset is 30,000 rows / ~2.7 MB, a stratified 3,000-row
sample is used for training in this assignment to keep the repo lightweight
and Streamlit Cloud's free tier responsive — this still comfortably exceeds
the assignment's 500-instance minimum.

Run this once to regenerate everything the Streamlit app depends on:
    python model/train_models.py
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
SAMPLE_SIZE = 3000  # stratified subsample of the full 30,000-row dataset
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # project-folder/

# ----------------------------------------------------------------------
# 1. Load and prepare dataset
# ----------------------------------------------------------------------
raw = pd.read_csv(os.path.join(HERE, "raw_source_data.csv"))
raw = raw.drop(columns=["ID"])
raw = raw.rename(columns={"default.payment.next.month": "target"})

# Stratified subsample to keep the repo/app lightweight (still >> 500 rows)
raw_sample, _ = train_test_split(
    raw, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=raw["target"]
)
raw_sample = raw_sample.reset_index(drop=True)

FEATURE_COLS = [c for c in raw_sample.columns if c != "target"]
X = raw_sample[FEATURE_COLS]
y = raw_sample["target"]

print(f"Working dataset shape: {X.shape}, class balance: {dict(y.value_counts())}")

# ----------------------------------------------------------------------
# 2. Train/test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the RAW (unscaled) test split as test_data.csv — this is what gets
# uploaded to the Streamlit app and also goes in the GitHub repo.
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ----------------------------------------------------------------------
# 3. Scale features
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(HERE, "scaler.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(HERE, "feature_names.pkl"))

# ----------------------------------------------------------------------
# 4. Define models
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(model, os.path.join(HERE, fname))

# ----------------------------------------------------------------------
# 5. Save comparison table
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(ROOT, "metrics.csv"), index=False)
print("\nSaved metrics.csv:")
print(results_df.to_string(index=False))

print("\nAll models and artifacts saved to model/ ✅")
