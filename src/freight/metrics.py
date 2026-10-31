"""Regression metrics reported on the original dollar scale."""
from __future__ import annotations

import numpy as np


def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(100 * np.mean(ape)),
        "MedAPE_%": float(100 * np.median(ape)),
    }