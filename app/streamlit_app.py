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
            st.write(preds[:20])  # ilk 20 tahmini göster
        else:
            st.error("❌ Model not loaded, please ensure model file exists in /artifacts.")
    except Exception as e:
        st.error(f"File processing failed: {e}")

# === Optional Debug (hidden) ===
with st.expander("🧩 Debug Info"):
    st.write("Working directory:", os.getcwd())
    st.write("Model path:", MODEL_PATH)
    st.write("Files in artifacts:", os.listdir(os.path.join(BASE_DIR, "..", "artifacts")) if os.path.exists(os.path.join(BASE_DIR, "..", "artifacts")) else "❌ No artifacts folder found")

"""import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from PIL import Image



st.title("🧠 PhD-Level Tabular AI Demo")
st.markdown("This demo showcases **end-to-end tabular AI development**, from prediction to explainability and model evaluation.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "best_model.pkl")
MODEL_PATH = os.path.normpath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
else:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully.")

# --- Page Configuration ---
st.set_page_config(
    page_title="PhD-Level Tabular AI Demo",
    page_icon="🧠",
    layout="wide"
)
# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📤 Upload & Predict", "🧩 Model Explanation", "🎯 Model Performance"])

# --- Load model ---
model_path = "../artifacts/best_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("❌ Model file not found. Please train the model first.")
    st.stop()


# =====================================================
# 📤 PAGE 1: UPLOAD & PREDICT
# =====================================================
if page == "📤 Upload & Predict":
    st.subheader("📄 Upload CSV for Prediction")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        preds = model.predict(df)
        df["Prediction"] = preds

        st.success("✅ Predictions completed!")
        st.dataframe(df.head(20))

        st.write(f"**Total Samples:** {len(preds)}")
        st.write(f"**Predicted Class ‘1’ (Positive):** {np.sum(preds == 1)}")
        st.write(f"**Predicted Class ‘0’ (Negative):** {np.sum(preds == 0)}")

        # --- Download predictions ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Full Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload a CSV file to get predictions.")


# =====================================================
# 🧠 PAGE 2: MODEL EXPLANATION (SHAP)
# =====================================================
elif page == "🧩 Model Explanation":
    st.subheader("🧠 Model Explanation (XAI – SHAP)")
    st.markdown("This section provides explainability visualization for the trained model using **SHAP** and fallback feature importances.")

    shap_path = "../figures/shap_summary.png"
    fallback_path = "../figures/shap_summary_fallback.png"

    if os.path.exists(shap_path):
        st.image(Image.open(shap_path), caption="SHAP Summary (Feature Impact)", use_container_width=True)
        st.success("✅ SHAP summary visualization loaded successfully.")
    elif os.path.exists(fallback_path):
        st.image(Image.open(fallback_path), caption="Fallback: Feature Importance Plot", use_container_width=True)
        st.warning("⚠️ SHAP not supported for this model; fallback visualization shown.")
    else:
        st.info("No SHAP visualization found. Please run the notebook to generate SHAP figures first.")

    st.markdown("---")
    st.markdown("**Model Information:**")
    st.code(str(model))


# =====================================================
# 🎯 PAGE 3: MODEL PERFORMANCE
# =====================================================
elif page == "🎯 Model Performance":
    st.subheader("📊 Model Evaluation Metrics")
    st.markdown("Here you can review performance metrics generated during training (Confusion Matrix, ROC Curve, Learning Curve).")

    # Paths to figures
    cm_path = "../figures/confusion_matrix.png"
    roc_path = "../figures/roc_curve.png"
    lc_path = "../figures/learning_curve.png"

    cols = st.columns(3)

    with cols[0]:
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion Matrix not found.")

    with cols[1]:
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption="ROC Curve", use_container_width=True)
        else:
            st.info("ROC Curve not found.")

    with cols[2]:
        if os.path.exists(lc_path):
            st.image(Image.open(lc_path), caption="Learning Curve", use_container_width=True)
        else:
            st.info("Learning Curve not found.")

    st.markdown("---")
    st.markdown("**Tip:** These visuals are automatically generated during notebook execution and saved under the `figures/` directory.")"""