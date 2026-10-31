"""Two-stage freight-rate model.

Stage 1  Gradient boosting on *load* features (distance, geography, equipment, weight,
         quote_signal) predicting log(posted_rate). It sees no calendar or market inputs, so
         it cannot memorise "which month had which price level".

Stage 2  A small weighted linear model on the *daily mean residual* of stage 1 (computed
         out-of-fold) that captures the market/seasonal level: day-of-week cycle, the market
         index, an optional trend, and an optional recency shift. Because it is linear and
         low-dimensional it extrapolates to unseen months far more safely than trees do.

Prediction = exp(stage1 + stage2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .features import STAGE1_FEATURES, linear_base_design

DEFAULT_HGB = dict(
    max_iter=400,
    early_stopping=False,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=0,
)

STAGE2_TERMS = ("dow", "mi", "qs", "mi_dev7", "mi_smooth7", "t", "recent")


class _Stage1:
    """Optional smooth linear baseline + gradient boosting on the baseline's residual.

    With `linear_base=True` the booster only has to learn what the power-law-like baseline cannot,
    so in sparse regions (very short hauls, cities never seen in training) predictions fall back to
    the smooth baseline instead of an arbitrary tree leaf.
    """

    def __init__(self, features, hgb_params, linear_base):
        self.features, self.hgb_params, self.linear_base = features, hgb_params, linear_base

    def fit(self, df, y):
        self.base_ = Ridge(alpha=1e-3).fit(linear_base_design(df), y) if self.linear_base else None
        offset = self.base_.predict(linear_base_design(df)) if self.linear_base else 0.0
        self.hgb_ = HistGradientBoostingRegressor(**self.hgb_params).fit(df[self.features], np.asarray(y) - offset)
        return self

    def predict(self, df):
        offset = self.base_.predict(linear_base_design(df)) if self.linear_base else 0.0
        return self.hgb_.predict(df[self.features]) + offset


def _design(day: pd.DataFrame, terms: tuple[str, ...]) -> np.ndarray:
    cols = []
    for term in terms:
        if term == "dow":
            for d in range(7):
                cols.append((day["dow"].to_numpy() == d).astype(float))
        elif term == "mi":
            cols.append(day["mi_day"].to_numpy(float))
        elif term == "qs":
            cols.append(day["qs_mean"].to_numpy(float))
        elif term in ("mi_dev7", "mi_smooth7", "t"):
            cols.append(day[term].to_numpy(float))
        elif term == "recent":
            continue  # handled as a constant shift after the fit
        else:
            raise ValueError(f"unknown stage-2 term {term!r}")
    if not cols:
        return np.zeros((len(day), 0))
    return np.column_stack(cols)


@dataclass
class FreightRateModel:
    features: list[str] = field(default_factory=lambda: list(STAGE1_FEATURES))
    adjust: tuple[str, ...] = ("dow", "mi", "t")
    recent_days: int = 28
    hgb_params: dict = field(default_factory=lambda: dict(DEFAULT_HGB))
    ridge_alpha: float = 1e-3
    linear_base: bool = True
    n_oof: int = 5
    seed: int = 0

    # ------------------------------------------------------------------ stage 1
    def fit_stage1(self, df: pd.DataFrame) -> "FreightRateModel":
        y = df["log_rate"]
        self.stage1_ = _Stage1(self.features, self.hgb_params, self.linear_base).fit(df, y)
        if self.adjust:  # out-of-fold residuals feed stage 2 (no in-sample optimism)
            oof = np.zeros(len(df))
            for tr_idx, te_idx in KFold(self.n_oof, shuffle=True, random_state=self.seed).split(df):
                m = _Stage1(self.features, self.hgb_params, self.linear_base).fit(df.iloc[tr_idx], y.iloc[tr_idx])
                oof[te_idx] = m.predict(df.iloc[te_idx])
            self._train_day_resid_ = (
                df.assign(res=y.to_numpy() - oof)
                .groupby("date")
                .agg(res=("res", "mean"), n=("res", "size"), mi_day=("mi_day", "first"),
                     mi_smooth7=("mi_smooth7", "first"), mi_dev7=("mi_dev7", "first"),
                     qs_mean=("qs_mean", "first"), dow=("dow", "first"), t=("t", "first"))
                .sort_index()
            )
        return self

    # ------------------------------------------------------------------ stage 2
    def set_adjust(self, adjust: tuple[str, ...]) -> "FreightRateModel":
        """(Re)fit stage 2 from the stored out-of-fold residuals."""
        self.adjust = tuple(adjust)
        self.stage2_ = None
        self.shift_ = 0.0
        day = self._train_day_resid_
        terms = tuple(t for t in self.adjust if t != "recent")
        if terms:
            Z = _design(day, terms)
            self._mu_, self._sd_ = Z.mean(0), Z.std(0) + 1e-9
            Z = (Z - self._mu_) / self._sd_
            self.stage2_ = Ridge(alpha=self.ridge_alpha).fit(Z, day["res"], sample_weight=day["n"])
        if "recent" in self.adjust:
            tail = day.iloc[-self.recent_days:]
            fitted = self._stage2_day(tail)
            self.shift_ = float(np.average(tail["res"] - fitted, weights=tail["n"]))
        return self

    def _stage2_day(self, day: pd.DataFrame) -> np.ndarray:
        terms = tuple(t for t in self.adjust if t != "recent")
        if self.stage2_ is None:
            return np.zeros(len(day))
        Z = (_design(day, terms) - self._mu_) / self._sd_
        return self.stage2_.predict(Z)

    def fit(self, df: pd.DataFrame) -> "FreightRateModel":
        self.fit_stage1(df)
        if self.adjust:
            self.set_adjust(self.adjust)
        else:
            self.stage2_, self.shift_ = None, 0.0
        return self

    # ------------------------------------------------------------------ predict
    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        base = self.stage1_.predict(df)
        if not self.adjust:
            return base
        return base + self._stage2_day(df) + self.shift_

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))