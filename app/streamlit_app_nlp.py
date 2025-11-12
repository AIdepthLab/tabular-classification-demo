#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:49:43 2025

@author: yakamoz
"""

# === NLP Sentiment Dashboard ===
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

st.title("NLP Sentiment Analysis Demo")
st.write("Upload a CSV with 'text' and 'label' columns, or use demo data.")

# === Demo Dataset ===
demo_df = pd.DataFrame({
    "text": [
        "I love this product!",
        "It was a waste of time.",
        "Excellent support and quality.",
        "Terrible experience.",
        "Absolutely fantastic!",
        "Worst service ever.",
        "Pretty decent overall.",
        "Highly recommended.",
        "Disappointed and angry.",
        "I will buy again!"
    ],
    "label": [1,0,1,0,1,0,1,1,0,1]
})

# Download button
st.download_button(
    label="Download Sample CSV",
    data=demo_df.to_csv(index=False).encode(),
    file_name="sample_sentiment_data.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("📂 Upload your CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File loaded successfully!")
else:
    st.info("No file uploaded — using demo dataset.")
    df = demo_df.copy()

# === Training ===
if 'text' in df.columns and 'label' in df.columns:
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(report).transpose())

    fig, ax = plt.subplots()
    ax.bar(['Accuracy'], [report['accuracy']], color='mediumseagreen')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

else:
    st.error("CSV must contain 'text' and 'label' columns.")