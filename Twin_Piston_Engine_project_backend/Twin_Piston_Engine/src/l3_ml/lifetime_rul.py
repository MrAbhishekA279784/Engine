"""
Lifetime-scale RUL (method ported from aerotwin_ml: aerotwin/sim/faults.py and
aerotwin/models/{health,rul}.py; driven by THIS backend's simulator and L2/L3
pipeline, see scripts/generate_lifetime_dataset.py and scripts/train_rul.py).

    fault process   severity s(t) = ((t - onset) / life)^k for t >= onset,
                    failure when s reaches 1 (onset 60-300 h, life 150-450 h,
                    k 1.4-2.6), so early signatures are genuinely small
    severity model  LightGBM regressor: ML feature schema 2.0.0 -> s_hat
    RUL model       LightGBM (Huber) on the HISTORY of s_hat only: level,
                    smoothed level, slopes over 20 and 60 monitoring windows,
                    windows since degradation was first seen, slope ratio.
                    The history is indexed by monitoring window (one window per
                    MONITOR_CADENCE_H engine hours), never by engine hours:
                    engine hours are the time axis of the RUL TARGETS only.
    uncertainty     per-band 5-95 % error quantiles on VALIDATION engines -> a
                    90 % interval (SRD-FUN-133/135)
    status          NO_DEGRADATION / INDETERMINATE (< MIN_HISTORY_WINDOWS since
                    degradation) / ESTIMATED (SRD-FUN-134)

Training-only imports (lightgbm, pandas) are inside the functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

RUL_CAP_H = 400.0
BANDS_H = ((0.0, 50.0), (50.0, 150.0), (150.0, RUL_CAP_H))
DEGRADATION_ON = 0.10
MIN_HISTORY_WINDOWS = 12
MONITOR_CADENCE_H = 5.0
HISTORY_COLS = ["sev", "sev_sm", "slope_20", "slope_60", "windows_since_degradation", "slope_ratio"]


def band_name(lo: float, hi: float) -> str:
    return f"{int(lo)}-{int(hi)}h"


@dataclass
class FaultLife:
    """Onset and power-law progression over engine hours (aerotwin FaultProcess)."""

    onset_h: float = np.inf
    life_h: float = np.inf
    exponent: float = 2.0

    @property
    def failure_h(self) -> float:
        return self.onset_h + self.life_h

    @classmethod
    def sample(cls, rng: np.random.Generator, faulty: bool) -> "FaultLife":
        if not faulty:
            return cls()
        return cls(onset_h=float(rng.uniform(60.0, 300.0)), life_h=float(rng.uniform(150.0, 450.0)),
                   exponent=float(rng.uniform(1.4, 2.6)))

    def severity(self, t_h: float) -> float:
        if not np.isfinite(self.onset_h) or t_h < self.onset_h:
            return 0.0
        return float(min(1.0, ((t_h - self.onset_h) / self.life_h) ** self.exponent))


def history_features(sev_hat: np.ndarray):
    """Causal features of one engine's severity-estimate history, one row per
    monitoring window (window index axis; no engine hours)."""
    import pandas as pd

    s = pd.Series(np.asarray(sev_hat, float))
    sm = s.ewm(span=8, adjust=False).mean()
    out = pd.DataFrame({"sev": s.to_numpy(), "sev_sm": sm.to_numpy()})
    for lag in (20, 60):
        out[f"slope_{lag}"] = ((sm - sm.shift(lag)) / lag).to_numpy() * 100.0
    on = (sm >= DEGRADATION_ON).to_numpy()
    first = int(np.argmax(on)) if on.any() else len(on)
    idx = np.arange(len(on))
    out["windows_since_degradation"] = np.where(idx >= first, idx - first, -1)
    out["slope_ratio"] = out["slope_20"] / (out["slope_60"].abs() + 1e-3)
    return out.fillna(0.0)


class SeverityModel:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, x, sev, x_val, sev_val) -> "SeverityModel":
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.6, random_state=self.seed,
                                       deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)
        self.model.fit(x, sev, eval_set=[(x_val, sev_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def predict(self, x) -> np.ndarray:
        return np.clip(self.model.predict(x), 0.0, 1.0)


class RULModel:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.band_q: dict[str, tuple[float, float]] = {}

    def fit(self, h, rul_h: np.ndarray) -> "RULModel":
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(objective="huber", alpha=20.0, n_estimators=600, learning_rate=0.03,
                                       num_leaves=31, min_child_samples=60, subsample=0.8, subsample_freq=1,
                                       random_state=self.seed, deterministic=True, force_row_wise=True,
                                       n_jobs=8, verbose=-1)
        self.model.fit(h[HISTORY_COLS].to_numpy(float), np.minimum(rul_h, RUL_CAP_H))
        return self

    def predict_raw(self, h) -> np.ndarray:
        return np.clip(self.model.predict(h[HISTORY_COLS].to_numpy(float)), 0.0, RUL_CAP_H)

    def _band(self, pred: float) -> str:
        for lo, hi in BANDS_H:
            if lo <= pred < hi:
                return band_name(lo, hi)
        return band_name(*BANDS_H[-1])

    def calibrate(self, h_val, rul_val: np.ndarray) -> "RULModel":
        """5-95 % error quantiles per PREDICTED-RUL band on validation engines."""
        pred = self.predict_raw(h_val)
        err = np.minimum(rul_val, RUL_CAP_H) - pred
        for lo, hi in BANDS_H:
            m = (pred >= lo) & (pred < (hi if hi < RUL_CAP_H else RUL_CAP_H + 1))
            key = band_name(lo, hi)
            self.band_q[key] = ((float(np.quantile(err[m], 0.05)), float(np.quantile(err[m], 0.95)))
                                if m.sum() >= 20 else (-hi, hi))
        return self

    def predict(self, h) -> dict[str, np.ndarray]:
        pred = self.predict_raw(h)
        w = h["windows_since_degradation"].to_numpy()
        status = np.where(w < 0, "NO_DEGRADATION", np.where(w < MIN_HISTORY_WINDOWS, "INDETERMINATE", "ESTIMATED"))
        q = np.array([self.band_q[self._band(p)] for p in pred])
        est = status == "ESTIMATED"
        return {"rul_h": np.where(est, pred, np.nan), "rul_lo_h": np.where(est, np.maximum(0.0, pred + q[:, 0]), np.nan),
                "rul_hi_h": np.where(est, pred + q[:, 1], np.nan), "status": status, "rul_raw_h": pred}


def band_metrics(true_h: np.ndarray, pred_h: np.ndarray, lo_h: np.ndarray | None = None,
                 hi_h: np.ndarray | None = None) -> dict[str, Any]:
    """MAE (and 90 % interval coverage) per TRUE-RUL band, true RUL capped at RUL_CAP_H."""
    t = np.minimum(np.asarray(true_h, float), RUL_CAP_H)
    p = np.asarray(pred_h, float)
    out: dict[str, Any] = {}
    for lo, hi in BANDS_H:
        m = (t >= lo) & ((t < hi) if hi < RUL_CAP_H else (t <= hi)) & np.isfinite(p)
        row: dict[str, Any] = {"n": int(m.sum()), "mae_h": float(np.mean(np.abs(p[m] - t[m]))) if m.any() else None}
        if lo_h is not None and m.any():
            row["coverage_90"] = float(np.mean((t[m] >= lo_h[m]) & (t[m] <= hi_h[m])))
        out[band_name(lo, hi)] = row
    return out


class LifetimeRULEstimator:
    """Runtime use of an adopted lifetime RUL bundle (scripts/train_rul.py).

    observe() is called with the engine hours and the ML feature row of each
    record; one severity estimate is kept per MONITOR_CADENCE_H engine hours
    (engine hours are only the sampling axis, never a model input). The RUL
    comes from the severity-estimate history with the validation-calibrated
    90 % interval. validated is True only for status ESTIMATED; otherwise the
    status (NO_DEGRADATION / INDETERMINATE) is reported without a number.
    """

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.features = list(bundle["features"])
        self.cadence_h = float(bundle["monitor_cadence_h"])
        self._sev: list[float] = []
        self._next_h: float | None = None

    def observe(self, engine_hours: float, feature_row: dict[str, float]) -> dict[str, Any]:
        if self._next_h is None or engine_hours >= self._next_h:
            x = np.asarray([[feature_row.get(n, np.nan) for n in self.features]], float)
            self._sev.append(float(self.bundle["severity_model"].predict(x)[0]))
            self._next_h = engine_hours + self.cadence_h
        h = history_features(np.asarray(self._sev))
        p = self.bundle["rul_model"].predict(h.iloc[[-1]])
        status = str(p["status"][0])
        return {"status": status, "rul_h": float(p["rul_h"][0]), "rul_lo_h": float(p["rul_lo_h"][0]),
                "rul_hi_h": float(p["rul_hi_h"][0]), "severity_estimate": self._sev[-1],
                "windows": len(self._sev), "validated": status == "ESTIMATED"}
