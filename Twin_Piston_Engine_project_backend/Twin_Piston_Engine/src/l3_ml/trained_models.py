"""
Trained-model bundles (Prompt 16).

A bundle is a joblib-saved dict written by scripts/train_models.py:

    kind                  "anomaly" | "classifier"
    estimator             fitted sklearn Pipeline (imputer + IsolationForest) or
                          LightGBM classifier
    feature_names         ML feature schema (src/l3_ml/ml_features.py)
    feature_schema_version
    threshold             anomaly score threshold (anomaly only)
    importance            per-feature importance, normalised to sum 1 (classifier)
    metadata              ModelMetadata fields (name, version, trained_at, metrics...)
    provenance            dataset version, dataset hash, code SHA, seeds,
                          class_count (= FAULT_CLASS_COUNT)

The wrappers below turn a bundle into a BaseMLModel whose predict() returns
the dict formats AnomalyDetector / FaultClassifier already understand.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.core.provenance import FAULT_CLASS_COUNT, FaultClass
from src.core.schemas import ModelMetadata
from src.l3_ml.ml_infrastructure import BaseMLModel, MLFeatureVector

TOP_CONTRIBUTIONS = 5
# A sensor group that was fully missing in fewer than this share of training
# rows is out of distribution when fully missing at inference: the model
# abstains (MODEL_UNAVAILABLE) instead of returning a confident class.
OOD_GROUP_MISSING_RATE = 0.01


def unseen_missing_groups(bundle: dict[str, Any], names: list[str], values: list[float]) -> list[str]:
    from src.l3_ml.ml_features import FEATURE_GROUPS
    rates = bundle.get("group_missing_rate") or {}
    missing: dict[str, bool] = {}
    for n, v in zip(names, values):
        g = FEATURE_GROUPS.get(n)
        if g is not None:
            missing[g] = missing.get(g, True) and math.isnan(v)
    return sorted(g for g, m in missing.items() if m and rates.get(g, 0.0) < OOD_GROUP_MISSING_RATE)


def _abstain(model: BaseMLModel, feature_vector: MLFeatureVector, groups: list[str]):
    from src.core.provenance import InferenceStatus
    from src.l3_ml.ml_infrastructure import InferenceResult
    return InferenceResult(
        model_name=model.metadata.name, model_version=model.metadata.version,
        feature_schema_version=feature_vector.schema_version, prediction=None,
        status=InferenceStatus.MODEL_UNAVAILABLE, timestamp=feature_vector.timestamp, quality=0.0,
        model_metadata=model.metadata,
        error_message=(f"sensor group(s) {groups} fully missing: this configuration was not seen in training "
                       "(< 1 % of rows); the model abstains and the rule-based result applies"))


def bundle_metadata(bundle: dict[str, Any]) -> ModelMetadata:
    return ModelMetadata(**bundle["metadata"])


class TrainedAnomalyModel(BaseMLModel):
    """IsolationForest on NOMINAL trajectories; score = -score_samples (higher
    = more anomalous); threshold chosen on validation for FPR <= 5 %."""

    accepts_missing = True  # the pipeline's imputer was fitted on training data

    def __init__(self, bundle: dict[str, Any]) -> None:
        super().__init__(bundle_metadata(bundle), bundle["feature_schema_version"])
        self.bundle = bundle
        self._est = bundle["estimator"]
        self._threshold = float(bundle["threshold"])

    def predict(self, feature_vector: MLFeatureVector) -> Any:
        ood = unseen_missing_groups(self.bundle, feature_vector.feature_names, feature_vector.values)
        if ood:
            return _abstain(self, feature_vector, ood)
        x = np.asarray([feature_vector.values], dtype=float)
        score = float(-self._est.score_samples(x)[0])
        present = sum(1 for v in feature_vector.values if not math.isnan(v)) / len(feature_vector.values)
        return {"score": score, "threshold": self._threshold, "is_anomaly": score >= self._threshold,
                "evidence": {"feature_coverage": present, "model": "IsolationForest",
                             "imputed_features": [n for n, v in zip(feature_vector.feature_names,
                                                                    feature_vector.values) if math.isnan(v)]}}


class TrainedFaultModel(BaseMLModel):
    """LightGBM multi-class over FAULT_CLASS_COUNT classes. Returns the class
    probabilities, the importance-weighted feature coverage (the classifier
    scales its confidence by it) and the top per-feature contributions
    (LightGBM pred_contrib = TreeSHAP values) for the predicted class."""

    accepts_missing = True  # LightGBM handles NaN natively

    def __init__(self, bundle: dict[str, Any]) -> None:
        super().__init__(bundle_metadata(bundle), bundle["feature_schema_version"])
        self.bundle = bundle
        self._est = bundle["estimator"]
        self._names = list(bundle["feature_names"])
        self._importance = np.asarray([bundle["importance"].get(n, 0.0) for n in self._names], dtype=float)
        classes = [int(c) for c in self._est.classes_]
        self._class_index = {c: i for i, c in enumerate(classes)}

    def predict(self, feature_vector: MLFeatureVector) -> Any:
        ood = unseen_missing_groups(self.bundle, feature_vector.feature_names, feature_vector.values)
        if ood:
            return _abstain(self, feature_vector, ood)
        x = np.asarray([feature_vector.values], dtype=float)
        p_model = self._est.predict_proba(x)[0]
        probs = [0.0] * FAULT_CLASS_COUNT
        for c, i in self._class_index.items():
            probs[c] = float(p_model[i])
        present = ~np.isnan(x[0])
        coverage = float(np.sum(self._importance[present]) / np.sum(self._importance)) if self._importance.sum() > 0 \
            else float(present.mean())
        best = int(np.argmax(probs))
        contrib = self._est.predict(x, pred_contrib=True)
        contrib = np.asarray(contrib[0] if isinstance(contrib, list) else contrib)
        n_f = len(self._names)
        row = contrib.reshape(-1)
        if row.size == (n_f + 1) * len(self._class_index):
            row = row[self._class_index[best] * (n_f + 1):(self._class_index[best] + 1) * (n_f + 1)]
        feats = row[:n_f]
        order = np.argsort(-np.abs(feats))[:TOP_CONTRIBUTIONS]
        top = [{"feature": self._names[i], "value": None if math.isnan(x[0][i]) else float(x[0][i]),
                "contribution": float(feats[i])} for i in order]
        return {"probabilities": probs, "feature_coverage": coverage,
                "evidence": {"top_features": top, "feature_coverage": coverage,
                             "explained_class": FaultClass(best).name, "method": "LightGBM pred_contrib (TreeSHAP)"}}


def wrap_bundle(bundle: dict[str, Any]) -> BaseMLModel:
    if bundle.get("kind") == "anomaly":
        return TrainedAnomalyModel(bundle)
    if bundle.get("kind") == "classifier":
        return TrainedFaultModel(bundle)
    raise ValueError(f"unknown model bundle kind {bundle.get('kind')!r}")
