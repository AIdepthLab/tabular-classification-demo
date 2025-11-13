#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AIdepthLab
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from typing import Dict, Any

def train_with_cv(pipe: Pipeline, param_grid: Dict[str, Any],
                  X_train, y_train, scoring: str = "f1", n_splits: int = 5):
    """
    GridSearchCV ile model eğitimi yapar ve en iyi modeli döndürür.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid