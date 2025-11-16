"""Utility helpers for ML integration in full league analytics.

Functions:
- safe_import_xgboost(): returns xgboost module or None if unavailable
- build_recency_weights(dates, decay): exponential decay weights (newest highest)
- save_models(models_dict, path): persist models with pickle
- load_models(path): load previously saved models
- check_min_samples(df, threshold): boolean gate
"""
from __future__ import annotations
import importlib
import logging
import os
import pickle
from typing import Dict, Any, Sequence
import numpy as np
import pandas as pd


def safe_import_xgboost():
    try:
        return importlib.import_module('xgboost')
    except Exception:
        logging.info("XGBoost not installed - skipping XGBoost models")
        return None


def build_recency_weights(dates: Sequence[Any], decay: float = 0.85) -> np.ndarray:
    """Compute exponential recency weights.
    Newest observation gets highest weight.
    If dates are not datetime-like or missing, fall back to positional ordering.
    Weight_i = decay**k where k increases with age.
    We then normalize to sum=1.
    """
    n = len(dates)
    if n == 0:
        return np.array([])
    # Attempt chronological ordering: assume input is already chronological old->new; if not, sort by date
    try:
        # If dates are datetime, ensure sorted; if unsorted, we reorder weights accordingly
        dt = pd.to_datetime(dates, errors='coerce')
        if dt.isnull().any():
            # Fallback to positional
            age_indices = np.arange(n-1, -1, -1)
        else:
            order = np.argsort(dt.values)  # old -> new
            # Map order to age indices: newest gets age 0
            rank = np.empty(n, dtype=int)
            rank[order] = np.arange(n)  # oldest rank 0
            # Convert rank so that newest has age 0
            age_indices = (n - 1) - rank
        weights = np.power(decay, age_indices)
    except Exception:
        age_indices = np.arange(n-1, -1, -1)
        weights = np.power(decay, age_indices)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights.astype(float)


def save_models(models: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    logging.info(f"Saved ML models to {path}")
    return path


def load_models(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    return models


def check_min_samples(df: pd.DataFrame, threshold: int) -> bool:
    return len(df) >= threshold

