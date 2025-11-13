"""
Explainable AI (XAI) utilities for tabular models (Pipeline-safe).
Author: AIdepthLAb
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# --- Path Setup ---
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def safe_save_path(filename):
    if not os.path.isabs(filename):
        filename = os.path.join(FIG_DIR, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename


def shap_summary_plot(model, X, save_path: str, show=False):
    """Generate SHAP summary plot; supports Pipeline, tree, linear, and generic models."""
    path = safe_save_path(save_path)

    print("Starting SHAP analysis...")
    print(f"Model type: {type(model).__name__}")

    # --- Handle sklearn Pipeline: unwrap the final estimator
    if isinstance(model, Pipeline):
        print("Detected Pipeline. Extracting final estimator...")
        try:
            model_inner = model.steps[-1][1]
            print(f"➡️ Inner model: {type(model_inner).__name__}")
        except Exception as e:
            print(f"Could not extract inner model: {e}")
            model_inner = model
    else:
        model_inner = model

    # --- Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    try:
        # --- Tree-based models
        if hasattr(model_inner, "feature_importances_"):
            print("Using TreeExplainer")
            explainer = shap.TreeExplainer(model_inner)
            shap_values = explainer(X)
        # --- Linear models
        elif hasattr(model_inner, "coef_"):
            print("Using LinearExplainer")
            explainer = shap.LinearExplainer(model_inner, X)
            shap_values = explainer(X)
        # --- Generic (e.g. MLP)
        else:
            print("Using KernelExplainer (generic)")
            if hasattr(model_inner, "predict_proba"):
                background = shap.sample(X, 50, random_state=42)
                explainer = shap.KernelExplainer(model_inner.predict_proba, background)
                shap_values = explainer.shap_values(X, nsamples=100)
            else:
                raise ValueError("Model has no predict_proba method")

        # --- Plot SHAP summary
        print("Generating SHAP summary plot...")
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        print(f"SHAP summary plot saved: {path}")

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Falling back to feature importance visualization...")
        try:
            feature_importances_plot(model_inner, X, path.replace(".png", "_fallback.png"))
        except Exception as e2:
            print(f"Fallback also failed: {e2}")


def feature_importances_plot(model, X, save_path, show=False):
    """Fallback bar plot for feature importances or coefficients."""
    path = safe_save_path(save_path)
    plt.figure(figsize=(8, 5))

    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        title = "Feature Importances (Tree-based)"
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_).ravel()
        title = "Feature Coefficients (Linear Model)"
    else:
        raise ValueError("Model has neither feature_importances_ nor coef_ attributes.")

    idx = np.argsort(vals)[::-1]
    top_features = np.array(X.columns)[idx][:15]
    top_vals = vals[idx][:15]

    plt.barh(top_features[::-1], top_vals[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Fallback plot saved: {path}")