import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Header ===
st.set_page_config(page_title="PhD-Level Tabular AI Demo", page_icon="🧠", layout="centered")
st.title("🧠 PhD-Level Tabular AI Demo")
st.write("This demo showcases end-to-end tabular AI development — prediction, evaluation, and explainability.")

# === Model Path Handling ===
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "best_model.pkl")
MODEL_PATH = os.path.normpath(MODEL_PATH)

# === Model Loading ===
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.warning(f"⚠️ Model file found but could not be loaded: {e}")
else:
    st.info("ℹ️ No trained model found. Please upload a CSV to test predictions.")
    # (İsteğe bağlı: buraya mini fallback model eklenebilir)

# === CSV Upload Section ===
st.header("📂 Upload CSV File for Prediction")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📊 **Uploaded Data Preview:**")
        st.dataframe(df.head())

        if model is not None:
            preds = model.predict(df)
            st.subheader("🔮 Predictions")
            st.write(preds[:50])  # ilk 20 tahmini göster
        else:
            st.error("❌ Model not loaded, please ensure model file exists in /artifacts.")
    except Exception as e:
        st.error(f"File processing failed: {e}")

# === Optional Debug (hidden) ===
with st.expander("🧩 Debug Info"):
    st.write("Working directory:", os.getcwd())
    st.write("Model path:", MODEL_PATH)
    st.write("Files in artifacts:", os.listdir(os.path.join(BASE_DIR, "..", "artifacts")) if os.path.exists(os.path.join(BASE_DIR, "..", "artifacts")) else "❌ No artifacts folder found")

