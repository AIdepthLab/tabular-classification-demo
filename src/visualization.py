#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AIdepthLab
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import learning_curve
import os

def plot_confusion_matrix(y_true, y_pred, save_path: str, show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
       plt.show()
    else:
       plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")

def plot_roc_curve(model, X_test, y_test, save_path: str, show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hasattr(model, "predict_proba"):
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

def plot_learning_curve(estimator, X, y, save_path: str, show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y)
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(train_sizes, test_scores.mean(axis=1), marker="s", label="Validation")
    plt.xlabel("Training samples")
    plt.ylabel("Score (F1)")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()