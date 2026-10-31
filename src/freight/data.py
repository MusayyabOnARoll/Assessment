"""Data loading helpers shared by validation and final prediction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import build_day_table, engineer, flag_label_outliers

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


def load_all():
    """Return (dev, val, day_table). `dev` carries the is_outlier flag; features only are pooled."""
    tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    day = build_day_table(tr, va)
    dev = engineer(tr, day)
    dev["is_outlier"] = flag_label_outliers(tr).to_numpy()
    val = engineer(va, day)
    return dev, val, day