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


# === Time Series Forecast Demo ===

from sklearn.linear_model import LinearRegression
import io

st.markdown("---")
st.header("📈 Time Series Forecast Demo")

st.write("This demo shows a simple trend forecasting model using synthetic time series data.")

# Model parametreleri (kullanıcı ayarları)
n_points = st.slider("Number of data points", 50, 300, 100, step=10)
noise_level = st.slider("Noise level", 0.0, 5.0, 2.0, step=0.5)
split_ratio = st.slider("Train/Test split ratio", 0.5, 0.9, 0.8, step=0.05)

# Demo başlat
if st.button("Run Forecast Demo"):
    np.random.seed(42)
    t = np.arange(0, n_points)
    y = 0.5 * t + 5 * np.sin(0.2 * t) + np.random.randn(n_points) * noise_level

    split = int(n_points * split_ratio)
    t_train, t_test = t[:split].reshape(-1, 1), t[split:].reshape(-1, 1)
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression().fit(t_train, y_train)
    y_pred = model.predict(t_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y, label="Actual", color="steelblue", linewidth=2)
    ax.plot(t_test.flatten(), y_pred, label="Predicted", color="darkorange", linewidth=2)
    ax.set_title("Time Series Forecasting (Linear Regression)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.pyplot(fig)

    st.success("✅ Forecast completed!")
    st.caption("Model: LinearRegression — demonstrates basic trend estimation on noisy synthetic data.")
