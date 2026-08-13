"""
app.py
------
Interactive Streamlit app to demonstrate 5 classification models trained on
the UCI "Default of Credit Card Clients" dataset.

Features:
  a. CSV upload (test data only)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix + classification report

Run locally with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    return scaler, feature_names, models


@st.cache_data
def load_precomputed_metrics():
    path = os.path.join(HERE, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


scaler, feature_names, models = load_artifacts()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("💳 Credit Card Default Prediction — Model Comparison Demo")
st.markdown(
    """
This app demonstrates **5 classification models** trained on the
**UCI "Default of Credit Card Clients"** dataset — predicting whether a credit
card holder in Taiwan will **default on their payment next month**, based on
23 features covering demographics, credit limit, and 6 months of billing /
repayment history.

Upload the provided `test_data.csv`, pick a model, and see how it performs.
"""
)

# ----------------------------------------------------------------------
# a. Dataset upload (CSV)
# ----------------------------------------------------------------------
st.header("1️⃣ Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (must contain the 23 feature columns + a 'target' column)",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded file with shape {df.shape}")
    st.dataframe(df.head(), use_container_width=True)

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    has_target = "target" in df.columns

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------------
    # b. Model selection dropdown
    # ------------------------------------------------------------------
    st.header("2️⃣ Select a Model")
    model_choice = st.selectbox("Choose a classification model:", list(models.keys()))
    model = models[model_choice]

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    result_df = df.copy()
    result_df["Predicted"] = y_pred
    result_df["Predicted Label"] = np.where(y_pred == 1, "Default", "No Default")
    result_df["Probability (Default)"] = np.round(y_proba, 4)

    st.subheader("Predictions")
    st.dataframe(result_df.head(20), use_container_width=True)

    # ------------------------------------------------------------------
    # c. Display evaluation metrics (only possible if ground truth given)
    # ------------------------------------------------------------------
    st.header("3️⃣ Evaluation Metrics")

    if has_target:
        y_true = df["target"]

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------------------
        # d. Confusion matrix + classification report
        # --------------------------------------------------------------
        st.header("4️⃣ Confusion Matrix & Classification Report")

        col_a, col_b = st.columns(2)

        with col_a:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {model_choice}")
            st.pyplot(fig)

        with col_b:
            report = classification_report(
                y_true, y_pred, target_names=["No Default", "Default"], output_dict=True
            )
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

    else:
        st.info(
            "Uploaded file has no 'target' column, so ground-truth evaluation "
            "metrics can't be computed — showing predictions only."
        )

    # ------------------------------------------------------------------
    # Comparison across ALL models on this uploaded test data
    # ------------------------------------------------------------------
    if has_target:
        st.header("5️⃣ Compare All Models on This Test Data")
        y_true = df["target"]
        rows = []
        for name, m in models.items():
            yp = m.predict(X_scaled)
            ypr = m.predict_proba(X_scaled)[:, 1]
            rows.append({
                "ML Model Name": name,
                "Accuracy": round(accuracy_score(y_true, yp), 4),
                "AUC": round(roc_auc_score(y_true, ypr), 4),
                "Precision": round(precision_score(y_true, yp), 4),
                "Recall": round(recall_score(y_true, yp), 4),
                "F1": round(f1_score(y_true, yp), 4),
                "MCC": round(matthews_corrcoef(y_true, yp), 4),
            })
        all_df = pd.DataFrame(rows)
        st.dataframe(
            all_df.style.highlight_max(
                subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                color="lightgreen",
            ),
            use_container_width=True,
        )

else:
    st.info("👆 Upload `test_data.csv` (included in the GitHub repo) to get started.")

    precomputed = load_precomputed_metrics()
    if precomputed is not None:
        st.header("📊 Pre-computed Metrics (from training run)")
        st.dataframe(precomputed, use_container_width=True)

st.markdown("---")
st.caption(
    "Built for ML Assignment 2 — Logistic Regression, Decision Tree, kNN, "
    "Naive Bayes, Random Forest (Ensemble) on the UCI Default of Credit Card "
    "Clients dataset."
)
