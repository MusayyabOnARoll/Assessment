"""Fit the final model on all cleaned development data and write the two prediction files.

Run:  python -m freight.train_predict
Outputs (in outputs/):
    validation_predictions.csv        load_id,predicted_rate for all 12,000 validation loads
    december_chart_predictions.csv    the 31 fixed December rows with predicted_rate filled in
    model.joblib                      the fitted model
    final_model_summary.json          training-set facts and sanity checks
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .data import DATA, OUTPUTS, load_all
from .features import STAGE1_FEATURES_DEV, engineer
from .model import DEFAULT_HGB, FreightRateModel

FINAL_ADJUST = ("mi", "qs", "dow", "t")


def december_frame(day: pd.DataFrame, dev: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model inputs for the fixed December chart loads.

    The chart file only has pickup/delivery/distance/equipment/weight/date, so the remaining
    inputs are filled from information that exists for those dates in validation.csv (features,
    never labels): coordinates from the city table, the day's market index, and the day's mean
    quote_signal (=> a neutral qs_dev of 0, i.e. an "average" load on that day).
    """
    raw = pd.read_csv(DATA / "december_chart_inputs.csv")
    coords = {}
    for frame in (dev, val):
        for side in ("pickup", "delivery"):
            for city, lat, lon in frame[[side, f"{side}_lat", f"{side}_lon"]].drop_duplicates().itertuples(index=False):
                coords[city] = (lat, lon)
    X = raw.drop(columns="predicted_rate").copy()
    X["date"] = pd.to_datetime(X["date"])
    for side in ("pickup", "delivery"):
        X[f"{side}_lat"] = X[side].map(lambda c: coords[c][0])
        X[f"{side}_lon"] = X[side].map(lambda c: coords[c][1])
    X["market_index"] = X["date"].map(day["mi_day"])
    X["quote_signal"] = X["date"].map(day["qs_mean"])
    return raw, engineer(X, day)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    dev, val, day = load_all()
    train = dev[~dev["is_outlier"]]
    params = dict(DEFAULT_HGB)
    model = FreightRateModel(features=STAGE1_FEATURES_DEV, adjust=FINAL_ADJUST, hgb_params=params).fit(train)

    # ---- validation predictions (same order as the provided template)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred = pd.Series(model.predict(val), index=val["load_id"], name="predicted_rate")
    if set(template["load_id"]) != set(pred.index) or pred.index.duplicated().any():
        raise SystemExit("validation.csv and the template do not hold the same load_id values")
    out = pd.DataFrame({"load_id": template["load_id"], "predicted_rate": pred.loc[template["load_id"]].round(2).to_numpy()})
    if not np.isfinite(out["predicted_rate"]).all() or (out["predicted_rate"] <= 0).any():
        raise SystemExit("non-finite or non-positive predictions")
    out.to_csv(OUTPUTS / "validation_predictions.csv", index=False)

    # ---- December chart predictions
    raw, dec = december_frame(day, dev, val)
    raw = raw.copy()
    raw["predicted_rate"] = model.predict(dec).round(2)
    raw.to_csv(OUTPUTS / "december_chart_predictions.csv", index=False)

    joblib.dump(model, OUTPUTS / "model.joblib")
    val_rpm = out["predicted_rate"].to_numpy() / val.set_index("load_id").loc[out["load_id"], "distance"].to_numpy()
    summary = {
        "n_dev_rows": int(len(dev)),
        "n_training_rows_after_outlier_removal": int(len(train)),
        "n_outliers_removed": int(dev["is_outlier"].sum()),
        "hgb_params": params,
        "stage2_terms": list(FINAL_ADJUST),
        "validation_pred_rate_mean": float(out["predicted_rate"].mean()),
        "validation_pred_rate_per_mile_median": float(np.median(val_rpm)),
        "train_clean_rate_per_mile_median": float((train["posted_rate"] / train["distance"]).median()),
        "december_pred_min_max": [float(raw["predicted_rate"].min()), float(raw["predicted_rate"].max())],
    }
    (OUTPUTS / "final_model_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()