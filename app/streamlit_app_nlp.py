#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:49:43 2025

@author: yakamoz
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

st.title("💬 NLP Sentiment Analysis Demo")
st.write("Upload a CSV with 'text' and 'label' (0=Negative, 1=Positive), or use the demo dataset.")

# --- Bigger, balanced demo dataset (100 rows) ---
pos = [
    "I love this product", "Amazing quality", "Great support", "Very satisfied",
    "Absolutely fantastic", "Highly recommended", "Works perfectly", "Fast delivery",
    "Excellent performance", "Will buy again"
] * 5
neg = [
    "Terrible experience", "Waste of money", "Very disappointed", "Poor quality",
    "Not recommended", "Does not work", "Late delivery", "Bad performance",
    "Support was awful", "I will never buy again"
] * 5

demo_df = pd.DataFrame({"text": pos + neg, "label": [1]*50 + [0]*50})

# Downloadable sample
st.download_button(
    "📄 Download Sample CSV (100 balanced rows)",
    demo_df.to_csv(index=False).encode(),
    file_name="sample_sentiment_data.csv",
    mime="text/csv"
)

# Upload or use demo
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ File loaded")
else:
    st.info("⚙️ No file uploaded — using demo dataset (balanced 100 rows).")
    df = demo_df.copy()

# Basic checks & cleaning
if not {"text", "label"}.issubset(df.columns):
    st.error("❌ CSV must contain 'text' and 'label' columns.")
    st.stop()

df = df.dropna(subset=["text", "label"]).copy()
# force numeric labels 0/1
try:
    df["label"] = df["label"].astype(int)
except Exception:
    # simple mapping if labels are strings
    df["label"] = df["label"].astype(str).str.lower().map({"positive":1, "pos":1, "1":1, "negative":0, "neg":0, "0":0}).fillna(0).astype(int)

# If dataset is tiny, bail early
if len(df) < 20 or df["label"].nunique() < 2:
    st.error("Dataset is too small or has a single class. Please provide a larger, balanced dataset.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vec = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Report with zero_division guard
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
rep_df = pd.DataFrame(report).transpose()
st.subheader("📈 Model Report")
st.dataframe(rep_df)

# Accuracy bar (robust)
acc = float(report.get("accuracy", 0.0))
fig, ax = plt.subplots()
ax.bar(["Accuracy"], [acc])
ax.set_ylim(0, 1)
st.pyplot(fig)

# Confusion Matrix
st.subheader("🧩 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
st.pyplot(fig_cm)

st.success("✅ Training & evaluation completed with stratified split on a balanced dataset.")