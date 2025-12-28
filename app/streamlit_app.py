
import streamlit as st
import joblib
import numpy as np
import os
import shap
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Paste a single transaction row (Time, V1–V28, Amount)")

# ---- Load model & scaler ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_xgb.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

@st.cache_resource
def load_model_and_scaler():
    from src.train import train

    # Train model (creates models/ files)
    train()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()


@st.cache_resource
def load_shap_explainer(model,background):
    return shap.Explainer(model, background)

# ---- Input ----
row_input = st.text_area(
    "Transaction row (comma-separated, 30 values)",
    height=150,
    placeholder="0.0, -1.3598, -0.07278, 2.5363, ..."
)

if st.button("Predict Fraud"):
    try:
        values = [float(x.strip()) for x in row_input.split(",")]

        if len(values) != 30:
            st.error("❌ Please enter exactly 30 values.")
        else:
            data = np.array(values).reshape(1, -1)
            data_scaled = scaler.transform(data)

            fraud_prob = model.predict_proba(data_scaled)[0][1]
            fraud_label = int(fraud_prob > 0.5)

            st.subheader("🔍 Prediction Result")

            st.metric(
                label="Fraud Probability",
                value=f"{fraud_prob:.4f}"
            )

            if fraud_label == 1:
                st.error("🚨 FRAUDULENT TRANSACTION")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

    except Exception as e:
        st.error(f"Error: {e}")


#--SHAP EXPLANATION--#
show_explain = st.checkbox("Show explanation (SHAP)")

if show_explain and 'values' in locals() and len(values) == 30:
    background = np.zeros((50, 30))  # fast, safe background
    explainer = load_shap_explainer(model, background)
    shap_values = explainer(data_scaled)

    st.subheader("🔎 Why this prediction?")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")

