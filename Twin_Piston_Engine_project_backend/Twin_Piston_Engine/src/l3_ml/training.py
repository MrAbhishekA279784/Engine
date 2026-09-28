"""
Model training and evaluation helpers (Prompt 16). Training-only: sklearn,
LightGBM and pandas are imported inside the functions, so the runtime does not
need them.

    train_anomaly_detector   IsolationForest (median imputer + missing
                             indicators) on NOMINAL trajectories of the train
                             split; threshold = the 95th percentile of the
                             validation NOMINAL records' scores (FPR <= 5 %)
    train_fault_classifier   LightGBM multi-class over FAULT_CLASS_COUNT,
                             balanced class weights, early stopping on the
                             validation split
    train_rul_regressor      LightGBM on causal rolling-window features
                             (trajectory-wise trailing means); kept only if it
                             beats the baseline RUL estimator in every horizon
                             band on validation

Nothing here looks at the test or holdout_op splits.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.core.provenance import FAULT_CLASS_COUNT, FaultClass
from src.l3_ml.ml_features import ML_FEATURE_SCHEMA_VERSION, ML_FEATURES

RUL_BANDS_H = (("<=50h", 0.0, 50.0), ("50-150h", 50.0, 150.0), (">150h", 150.0, math.inf))
RUL_WINDOW_S = 30


OOD_GROUP_MISSING_RATE = 0.01


def group_full_missing_rates(df) -> dict[str, float]:
    """Share of rows in which EVERY feature of a sensor group is missing."""
    from src.l3_ml.ml_features import FEATURE_GROUPS, SENSOR_GROUPS
    out = {}
    for g in SENSOR_GROUPS:
        cols = [n for n in ML_FEATURES if FEATURE_GROUPS[n] == g]
        out[g] = float(df[cols].isna().all(axis=1).mean())
    return out


def _xy(df, features=ML_FEATURES):
    return df[list(features)].to_numpy(dtype=float), df["label"].to_numpy(dtype=int)


def train_anomaly_detector(train, val, seed: int, fpr_target: float = 0.05) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    nominal = train[train["injected_class"] == int(FaultClass.NOMINAL)]
    x_tr, _ = _xy(nominal)
    est = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                        IsolationForest(n_estimators=300, max_samples=4096, random_state=seed, n_jobs=-1))
    est.fit(x_tr)
    x_va, y_va = _xy(val)
    scores = -est.score_samples(x_va)
    nominal_scores = scores[y_va == int(FaultClass.NOMINAL)]
    threshold = float(np.quantile(nominal_scores, 1.0 - fpr_target))
    fpr = float(np.mean(nominal_scores >= threshold))
    from sklearn.metrics import average_precision_score
    pr_auc = float(average_precision_score(y_va != int(FaultClass.NOMINAL), scores))
    return {"estimator": est, "threshold": threshold, "group_missing_rate": group_full_missing_rates(nominal),
            "val_metrics": {"fpr": fpr, "pr_auc": pr_auc, "n_train_records": int(len(x_tr))}}


def train_fault_classifier(train, val, seed: int) -> dict[str, Any]:
    import lightgbm as lgb

    x_tr, y_tr = _xy(train)
    x_va, y_va = _xy(val)
    est = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
        random_state=seed, deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)
    est.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    gain = est.booster_.feature_importance(importance_type="gain")
    total = float(np.sum(gain)) or 1.0
    importance = {name: float(g / total) for name, g in zip(ML_FEATURES, gain)}
    from sklearn.metrics import f1_score
    pred = est.predict(x_va)
    return {"estimator": est, "importance": importance, "group_missing_rate": group_full_missing_rates(train),
            "val_metrics": {"macro_f1": float(f1_score(y_va, pred, average="macro",
                                                       labels=list(range(FAULT_CLASS_COUNT)))),
                            "best_iteration": int(est.best_iteration_ or est.n_estimators)}}


def rul_windowed_features(df):
    """Trailing RUL_WINDOW_S-record means per trajectory (causal) + current row."""
    df = df.sort_values(["trajectory_id", "time_s"])
    rolled = (df.groupby("trajectory_id")[list(ML_FEATURES)]
              .rolling(RUL_WINDOW_S, min_periods=1).mean().reset_index(level=0, drop=True))
    rolled.columns = [f"win_{c}" for c in rolled.columns]
    return df.join(rolled)


def rul_band_errors(true_h, pred_h) -> dict[str, Any]:
    out = {}
    true_h, pred_h = np.asarray(true_h, float), np.asarray(pred_h, float)
    for name, lo, hi in RUL_BANDS_H:
        m = (true_h > lo if lo > 0 else true_h >= lo) & (true_h <= hi) & np.isfinite(pred_h)
        n = int(np.sum(m))
        out[name] = {"n": n, "mae_h": float(np.mean(np.abs(pred_h[m] - true_h[m]))) if n else None}
    return out


def train_rul_regressor(train, val, seed: int) -> dict[str, Any]:
    import lightgbm as lgb

    def prep(d):
        d = rul_windowed_features(d[d["injected_class"] != int(FaultClass.NOMINAL)])
        d = d[np.isfinite(d["ttf_s"])]
        cols = list(ML_FEATURES) + [f"win_{c}" for c in ML_FEATURES]
        return d, d[cols].to_numpy(float), d["ttf_s"].to_numpy(float) / 3600.0

    tr, x_tr, y_tr = prep(train)
    va, x_va, y_va = prep(val)
    est = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=31, random_state=seed,
                            deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)
    est.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    ml = rul_band_errors(y_va, est.predict(x_va))
    base = rul_band_errors(y_va, va["rul_baseline_h"].to_numpy(float))
    populated = [b for b, _, _ in RUL_BANDS_H if ml[b]["n"] and base[b]["n"]]
    beats = bool(populated) and all(ml[b]["mae_h"] < base[b]["mae_h"] for b in populated) and \
        len(populated) == len(RUL_BANDS_H)
    return {"estimator": est, "val_ml": ml, "val_baseline": base, "populated_bands": populated,
            "adopted": beats,
            "decision": ("adopted: beats the baseline in every horizon band" if beats else
                         "not adopted: it must beat the baseline in every horizon band on validation; "
                         f"populated bands {populated} of {[b for b, _, _ in RUL_BANDS_H]}")}


def make_bundle(kind: str, name: str, version: str, estimator, feature_names, extra: dict,
                provenance: dict, metrics: dict, description: str) -> dict[str, Any]:
    n_out = 1 if kind == "anomaly" else FAULT_CLASS_COUNT
    return {
        "kind": kind,
        "estimator": estimator,
        "feature_names": list(feature_names),
        "feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "metadata": {"name": name, "version": version, "trained_at": datetime.now(timezone.utc),
                     "input_shape": [len(feature_names)], "output_shape": [n_out],
                     "accuracy_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                     "description": description},
        "provenance": {**provenance, "class_count": FAULT_CLASS_COUNT},
        **extra,
    }
