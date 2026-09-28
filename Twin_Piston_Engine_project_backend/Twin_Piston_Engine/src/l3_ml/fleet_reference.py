"""
Fleet healthy-residual reference (Prompt 17b; OI-25).

The fleet residual of a HEALTHY engine is not zero: the L2 expectations carry
an operating-point dependent bias (OI-5; EGT about -80 K in descent to -230 K
in climb). The frozen models learned those residuals as "healthy". Per-engine
adaptation therefore corrects each engine TOWARDS THIS REFERENCE at the same
operating point, not towards zero.

    reference_ch(op) = quadratic polynomial (with interactions) in the
                       standardized operating point (rpm, MAP, altitude,
                       ambient temperature, ambient pressure), ridge least
                       squares on healthy fleet records

Fitted only on healthy records (no fault active) of TRAIN engines. Stored as
JSON next to the model bundles it belongs to.
"""

from __future__ import annotations

import json
import math
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np

OP_KEYS: tuple[str, ...] = ("rpm", "map_pressure_pa", "altitude_m", "ambient_temp_k", "ambient_pressure_pa")
RIDGE = 1e-6


def op_vector(op: Any) -> np.ndarray | None:
    """OperatingPoint (or mapping) -> vector in OP_KEYS order; None if incomplete."""
    vals = []
    for k in OP_KEYS:
        v = op.get(k) if isinstance(op, dict) else getattr(op, k, None)
        if v is None or not math.isfinite(float(v)):
            return None
        vals.append(float(v))
    return np.asarray(vals)


def _design(z: np.ndarray) -> np.ndarray:
    n, d = z.shape
    cols = [np.ones(n)] + [z[:, i] for i in range(d)]
    cols += [z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


class FleetResidualReference:
    def __init__(self, mean: list[float], scale: list[float], coefs: dict[str, list[float]],
                 fit_stats: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> None:
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coefs = {ch: np.asarray(c, float) for ch, c in coefs.items()}
        self.fit_stats = fit_stats or {}
        self.meta = meta or {}

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.coefs)

    @classmethod
    def fit(cls, ops: np.ndarray, residuals: dict[str, np.ndarray], meta: dict[str, Any] | None = None,
            min_rows: int = 30) -> FleetResidualReference:
        ops = np.asarray(ops, float)
        ok_op = np.all(np.isfinite(ops), axis=1)
        mean = ops[ok_op].mean(axis=0)
        scale = ops[ok_op].std(axis=0)
        scale[scale < 1e-9] = 1.0
        coefs, stats = {}, {}
        for ch, r in residuals.items():
            r = np.asarray(r, float)
            m = ok_op & np.isfinite(r)
            if m.sum() < min_rows:
                continue
            x = _design((ops[m] - mean) / scale)
            coef = np.linalg.solve(x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ r[m])
            fitted = x @ coef
            coefs[ch] = coef
            stats[ch] = {"rows": int(m.sum()), "residual_sd": float(np.std(r[m])),
                         "sd_after_reference": float(np.std(r[m] - fitted))}
        return cls(mean.tolist(), scale.tolist(), {k: v.tolist() for k, v in coefs.items()}, stats, meta)

    def predict_many(self, ops: np.ndarray, channel: str) -> np.ndarray:
        ops = np.atleast_2d(np.asarray(ops, float))
        if channel not in self.coefs:
            return np.full(len(ops), np.nan)
        return _design((ops - self.mean) / self.scale) @ self.coefs[channel]

    def predict(self, op: Any, channel: str) -> float | None:
        v = op_vector(op)
        if v is None or channel not in self.coefs:
            return None
        return float(self.predict_many(v[None, :], channel)[0])

    def to_dict(self) -> dict[str, Any]:
        return {"op_keys": list(OP_KEYS), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coefs": {k: v.tolist() for k, v in self.coefs.items()}, "fit_stats": self.fit_stats,
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FleetResidualReference:
        assert tuple(d["op_keys"]) == OP_KEYS, "fleet reference operating-point keys differ from the code"
        return cls(d["mean"], d["scale"], d["coefs"], d.get("fit_stats"), d.get("meta"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> FleetResidualReference:
        return cls.from_dict(json.loads(Path(path).read_text()))


FLEET_REFERENCE_FILE = "fleet_residual_reference.json"


def load_for_settings(settings: Any) -> FleetResidualReference | None:
    """<model_dir>/<adaptation.fleet_reference_version>/fleet_residual_reference.json, or None."""
    v = settings.adaptation.fleet_reference_version
    if not v:
        return None
    p = Path(settings.model_dir) / v / FLEET_REFERENCE_FILE
    return FleetResidualReference.load(p) if p.exists() else None
