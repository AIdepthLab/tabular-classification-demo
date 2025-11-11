#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 21:57:54 2025

@author: yakamoz
"""

import os
import json
import joblib

def ensure_dir(path: str):
    """Verilen dizin yoksa oluşturur."""
    os.makedirs(path, exist_ok=True)

def save_json(data: dict, path: str):
    """JSON dosyası kaydeder."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path: str):
    """JSON dosyasını okur."""
    with open(path, "r") as f:
        return json.load(f)

def save_model(model, path: str):
    """Modeli .pkl dosyası olarak kaydeder."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)

def load_model(path: str):
    """Kaydedilmiş modeli yükler."""
    return joblib.load(path)