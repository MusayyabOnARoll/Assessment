"""Cleaning and feature engineering for the freight-rate challenge.

Design rules
------------
* Only *features* are ever pooled across train and validation files (e.g. the daily
  market-index average). No validation label exists, and no training label is used to
  build a feature, so nothing here leaks the target.
* City names are deliberately NOT model inputs: 8 cities appear only in the validation
  file, so the model must generalise through coordinates, distance and equipment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
WEIGHT_CAP = 47_500.0
T0 = pd.Timestamp("2025-01-01")

# Load-level inputs of the stage-1 gradient-boosting model (no time / market inputs).
STAGE1_FEATURES = [
    "distance",
    "hav_miles",
    "circuity",
    "weight_f",
    "weight_missing",
    "weight_capped",
    "equipment_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "d_lat",
    "d_lon",
    "quote_signal",
]
STAGE1_FEATURES_DEV = [f for f in STAGE1_FEATURES if f != "quote_signal"] + ["qs_dev"]


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def linear_base_design(df: pd.DataFrame) -> np.ndarray:
    """Design matrix of the smooth parametric baseline: log-rate is close to linear in log-distance per equipment type."""
    ld = np.log(df["distance"].to_numpy(float))
    eq = df["equipment_code"].to_numpy()
    w = np.log(df["weight_f"].fillna(31_000.0).to_numpy(float))
    cols = [ld, ld**2, w, df["weight_missing"].to_numpy(float), df["qs_dev"].to_numpy(float), df["qs_dev"].to_numpy(float) * ld]
    for code in (1, 2):  # Dry Van is the reference level
        ind = (eq == code).astype(float)
        cols += [ind, ind * ld]
    return np.column_stack(cols)


# --------------------------------------------------------------------------- labels
def log_rpm_residual(df: pd.DataFrame, n_bins: int = 20) -> tuple[pd.Series, float]:
    """Log rate-per-mile minus its median inside equipment x distance-quantile cells, and a robust sigma (MAD)."""
    log_rpm = np.log(df["posted_rate"] / df["distance"])
    cell = pd.qcut(df["distance"], n_bins, labels=False, duplicates="drop")
    baseline = log_rpm.groupby([df["equipment"], cell]).transform("median")
    resid = log_rpm - baseline
    sigma = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
    return resid, sigma


def flag_label_outliers(df: pd.DataFrame, k: float = 6.0, n_bins: int = 20) -> pd.Series:
    """Flag rows whose rate is implausible for their equipment and distance.

    About 1.4 % of training rates are ~3.4x or ~0.28x the normal level, independent of
    every feature (unpredictable label corruption). A robust baseline (median log
    rate-per-mile inside equipment x distance-quantile cells) and a MAD-scaled threshold
    separate them cleanly: the gap between normal and corrupted rows is wide, so any k
    between 4 and 10 flags the same ~677 rows. Used on TRAINING rows only.
    """
    resid, sigma = log_rpm_residual(df, n_bins)
    return (np.abs(resid) > k * sigma).rename("is_outlier")


# --------------------------------------------------------------------------- market
def build_day_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-day market features from the pooled feature columns of all supplied frames.

    market_index barely varies within a day (std ~0.025 vs ~0.17 between days), so the
    daily mean is a de-noised and gap-free version of it. Uses features only.
    """
    pooled = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames], ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"])
    grp = pooled.groupby("date")
    day = pd.DataFrame(
        {
            "mi_day": grp["market_index"].mean(),
            "qs_day": grp["quote_signal"].median(),
            "qs_mean": grp["quote_signal"].mean(),
            "n_loads": grp.size(),
        }
    ).asfreq("D")
    day["mi_day"] = day["mi_day"].interpolate(limit_direction="both")
    day["qs_day"] = day["qs_day"].interpolate(limit_direction="both")
    day["qs_mean"] = day["qs_mean"].interpolate(limit_direction="both")
    day["mi_smooth7"] = day["mi_day"].rolling(7, center=True, min_periods=4).mean()
    day["mi_dev7"] = day["mi_day"] - day["mi_smooth7"]  # the weekly cycle of the index
    day["dow"] = day.index.dayofweek
    day["t"] = (day.index - T0).days.astype(float)
    return day


# --------------------------------------------------------------------------- features
def engineer(df: pd.DataFrame, day_table: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # weight: negative values are sign errors (|w| is valid, min 5,000); keep NaN as-is
    # (gradient boosting handles missing natively) with an explicit flag; 47,500 is a cap.
    w = out["weight"]
    out["weight_missing"] = w.isna().astype(int)
    out["weight_f"] = w.abs()
    out["weight_capped"] = (out["weight_f"] >= WEIGHT_CAP).astype(int)

    out["equipment_code"] = out["equipment"].map(EQUIPMENT_CODES)
    if out["equipment_code"].isna().any():
        raise ValueError("Unknown equipment type found")

    out["hav_miles"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["circuity"] = out["distance"] / out["hav_miles"].clip(lower=1.0)
    out["d_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["d_lon"] = out["delivery_lon"] - out["pickup_lon"]

    # market features: row value where present, otherwise the day's average
    day_cols = day_table[["mi_day", "mi_smooth7", "mi_dev7", "dow", "t", "qs_day", "qs_mean"]]
    out = out.join(day_cols, on="date")
    # quote_signal split into its between-day part (qs_mean, a market-level input for stage 2)
    # and its within-day part (qs_dev, a load-level input for stage 1)
    out["qs_dev"] = out["quote_signal"] - out["qs_mean"]
    out["market_index_filled"] = out["market_index"].fillna(out["mi_day"])
    if "posted_rate" in out:
        out["log_rate"] = np.log(out["posted_rate"])
    return out