#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AIdepthLab
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_data():
    """Gömülü Breast Cancer dataset'ini X, y olarak döndürür."""
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y

def load_custom_csv(csv_path: str, target_col: str = None):
    """Harici CSV'yi okur. target_col verilmişse (X, y) döner."""
    df = pd.read_csv(csv_path)
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y
    return df