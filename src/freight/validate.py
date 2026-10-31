"""Validation experiments: out-of-time rolling-origin CV, ablations, leaky-split and unseen-city checks.

Run:  python -m freight.validate         (from the repo root, with PYTHONPATH=src)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .data import REPORTS, load_all
from .features import STAGE1_FEATURES, STAGE1_FEATURES_DEV
from .metrics import score
from .model import FreightRateModel

# Expanding-window folds: train on every clean row before the test block, test on the next two months.
FOLDS = [
    ("May-Jun", "2025-05-01", "2025-07-01"),
    ("Jul-Aug", "2025-07-01", "2025-09-01"),
    ("Sep-Oct", "2025-09-01", "2025-11-01"),
]
NO_QS = [f for f in STAGE1_FEATURES if f != "quote_signal"]
FINAL_ADJUST = ("mi", "qs", "dow", "t")
FINAL_NAME = "FINAL: linear baseline + booster, qs split, market + qs level + dow + trend"

# Single-stage gradient boosting: time/market information goes straight into the trees.
SINGLE_STAGE = [
    ("GBM, load features only (no quote_signal)", NO_QS),
    ("GBM + quote_signal", STAGE1_FEATURES),
    ("GBM + quote_signal + raw market_index", STAGE1_FEATURES + ["market_index_filled"]),
    ("GBM + quote_signal + date index", STAGE1_FEATURES + ["t"]),
]
# Two-stage: (stage-1 feature set, linear baseline?, [(name, stage-2 terms)]); stage 1 is fitted once per fold and shared.
TWO_STAGE = [
    (STAGE1_FEATURES, False, [
        ("2-stage: day-of-week only", ("dow",)),
        ("2-stage: market + dow + trend", ("mi", "dow", "t")),
    ]),
    (STAGE1_FEATURES_DEV, False, [
        ("2-stage, qs split: market + dow", ("mi", "dow")),
        ("2-stage, qs split: market + qs level + dow", ("mi", "qs", "dow")),
        ("2-stage, qs split: market + qs level + dow + trend", FINAL_ADJUST),
    ]),
    (STAGE1_FEATURES_DEV, True, [
        (FINAL_NAME, FINAL_ADJUST),
        ("FINAL terms + recent shift", FINAL_ADJUST + ("recent",)),
    ]),
]


def _eval(model, test) -> dict:
    pred = model.predict(test)
    clean = ~test["is_outlier"].to_numpy()
    m = score(test["posted_rate"][clean], pred[clean])
    a = score(test["posted_rate"], pred)
    return {**m, "MAE_all_rows": a["MAE"], "RMSE_all_rows": a["RMSE"]}


def _split(dev, start, end):
    train = dev[(dev["date"] < start) & ~dev["is_outlier"]]
    test = dev[(dev["date"] >= start) & (dev["date"] < end)]
    return train, test


def rolling_origin(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, start, end in FOLDS:
        t0 = time.time()
        train, test = _split(dev, start, end)
        meta = {"fold": fold, "n_train": len(train), "n_test": len(test)}
        for name, feats in SINGLE_STAGE:
            m = FreightRateModel(features=feats, adjust=(), linear_base=False).fit(train)
            rows.append({"config": name, **meta, **_eval(m, test)})
        for feats, linear_base, variants in TWO_STAGE:
            base = FreightRateModel(features=feats, adjust=("dow",), linear_base=linear_base).fit_stage1(train)
            for name, adj in variants:
                base.set_adjust(adj)
                rows.append({"config": name, **meta, **_eval(base, test)})
        print(f"fold {fold} done in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def leaky_random_split(dev: pd.DataFrame) -> dict:
    """Reference only: a random 80/20 split mixes months, so it flatters a model that is really judged on the future."""
    mask = np.random.RandomState(0).rand(len(dev)) < 0.8
    train, test = dev[mask & ~dev["is_outlier"].to_numpy()], dev[~mask]
    return _eval(FreightRateModel(features=STAGE1_FEATURES_DEV, adjust=FINAL_ADJUST).fit(train), test)


def unseen_city_check(dev: pd.DataFrame, n_draws: int = 5, n_cities: int = 6) -> pd.DataFrame:
    """Hold out every load touching a random set of cities (the validation file has 8 such cities)."""
    cities = sorted(set(dev["pickup"]) | set(dev["delivery"]))
    rows = []
    for seed in range(n_draws):
        held = set(np.random.RandomState(seed).choice(cities, n_cities, replace=False))
        touches = dev["pickup"].isin(held) | dev["delivery"].isin(held)
        train, test = dev[~touches & ~dev["is_outlier"]], dev[touches]
        m = FreightRateModel(features=STAGE1_FEATURES_DEV, adjust=FINAL_ADJUST).fit(train)
        rows.append({"draw": seed, "held_out": ", ".join(sorted(held)), "n_test": len(test), **_eval(m, test)})
    return pd.DataFrame(rows)


def main():
    dev, _, _ = load_all()
    REPORTS.mkdir(exist_ok=True)
    res = rolling_origin(dev)
    res.to_csv(REPORTS / "rolling_origin_results.csv", index=False)
    summary = res.pivot(index="config", columns="fold", values="MAPE_%")[[f for f, *_ in FOLDS]]
    summary["mean"] = summary.mean(axis=1)
    order = [c for c in res["config"].drop_duplicates()]
    summary = summary.loc[order]
    summary.to_csv(REPORTS / "rolling_origin_mape_summary.csv")
    print("\nMAPE % on clean test rows, expanding-window out-of-time folds")
    print(summary.round(2).to_string())

    leaky = leaky_random_split(dev)
    (REPORTS / "leaky_random_split.json").write_text(json.dumps(leaky, indent=2))
    print("\nrandom 80/20 (leaky reference):", {k: round(v, 3) for k, v in leaky.items()})
    city = unseen_city_check(dev)
    city.to_csv(REPORTS / "unseen_city_check.csv", index=False)
    print("\nunseen-city check:\n", city.round(3).to_string())


if __name__ == "__main__":
    main()