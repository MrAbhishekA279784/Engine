"""
Residual drift monitoring (Prompt 17; SIH section D).

For each residual channel the baseline keeps a reference distribution as
histogram bins (quantile edges of the healthy reference residuals). The drift
score is the Population Stability Index of the recent window against it:

    PSI = sum_i (p_i - q_i) ln(p_i / q_i),   p recent, q reference (floored at eps)

Bands (config drift.*): < psi_moderate STABLE, < psi_significant MODERATE,
otherwise SIGNIFICANT. The overall score is the largest channel PSI.

The reference comes from the engine's active baseline (adaptation.py). Without
one, the monitor builds it from the first reference_min_records records the
caller marks healthy (self-commissioning), and reports that it did.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.schemas import ChannelDrift, DriftState

DRIFT_CHANNELS: tuple[str, ...] = (
    "egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4", "cht", "coolant", "fuel_flow", "oil_pressure", "oil_temp",
    "vibration_rms", "brake_power_kw", "fuel_rail_pressure", "fuel_delivery_ratio",
    "injector_flow_ratio_cyl1", "injector_flow_ratio_cyl2", "injector_flow_ratio_cyl3", "injector_flow_ratio_cyl4")


# Running-median residuals (Prompt 13): consecutive records are strongly
# autocorrelated, so their PSI is inflated; reported, not in the overall score.
AUTOCORRELATED_CHANNELS: frozenset[str] = frozenset(
    {"fuel_delivery_ratio", "injector_flow_ratio_cyl1", "injector_flow_ratio_cyl2", "injector_flow_ratio_cyl3",
     "injector_flow_ratio_cyl4"})


def overall_score(scores: dict[str, float]) -> float | None:
    vals = [v for ch, v in scores.items() if ch not in AUTOCORRELATED_CHANNELS]
    return max(vals) if vals else None


def residual_values(residual_state: Any) -> dict[str, float]:
    """Valid residual values of the drift channels (missing ones omitted)."""
    out: dict[str, float] = {}
    if residual_state is None:
        return out
    for ch in DRIFT_CHANNELS:
        tv = residual_state.residuals.get(ch)
        if tv is not None and tv.valid and tv.value is not None and math.isfinite(tv.value):
            out[ch] = float(tv.value)
    return out


class DriftReference:
    """Per-channel histogram reference: inner bin edges and bin probabilities."""

    def __init__(self, edges: dict[str, list[float]], probs: dict[str, list[float]], version: str) -> None:
        self.edges, self.probs, self.version = edges, probs, version

    @classmethod
    def fit(cls, samples: dict[str, list[float]], bins: int, version: str, min_records: int) -> "DriftReference":
        edges, probs = {}, {}
        for ch, vals in samples.items():
            v = np.asarray(vals, float)
            if v.size < min_records:
                continue
            inner = np.unique(np.quantile(v, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
            counts = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(inner) + 1)
            edges[ch] = inner.tolist()
            probs[ch] = (counts / counts.sum()).tolist()
        return cls(edges, probs, version)

    def to_dict(self) -> dict[str, Any]:
        return {"edges": self.edges, "probs": self.probs, "version": self.version}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DriftReference":
        return cls(d["edges"], d["probs"], d["version"])


def psi(reference_probs: list[float], inner_edges: list[float], values: list[float], eps: float) -> float:
    q = np.maximum(np.asarray(reference_probs, float), eps)
    counts = np.bincount(np.searchsorted(np.asarray(inner_edges, float), np.asarray(values, float), side="right"),
                         minlength=len(q))
    p = np.maximum(counts / max(counts.sum(), 1), eps)
    return float(np.sum((p - q) * np.log(p / q)))


class DriftMonitor:
    """Stateful: observe() once per record; evaluate() gives the DriftState."""

    def __init__(self, settings: AppSettings | None = None, reference: DriftReference | None = None) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.drift
        self.reference = reference
        self._window = {ch: deque(maxlen=self.cfg.window_records) for ch in DRIFT_CHANNELS}
        self._commissioning: dict[str, list[float]] = {ch: [] for ch in DRIFT_CHANNELS}
        self.self_commissioned = False

    def band(self, value: float | None) -> str:
        if value is None:
            return "UNKNOWN"
        if value < self.cfg.psi_moderate:
            return "STABLE"
        return "MODERATE" if value < self.cfg.psi_significant else "SIGNIFICANT"

    def observe(self, residual_state: Any, healthy: bool = False) -> None:
        vals = residual_values(residual_state)
        if self.reference is None:
            if healthy:
                for ch, v in vals.items():
                    self._commissioning[ch].append(v)
                n = max((len(v) for v in self._commissioning.values()), default=0)
                if n >= self.cfg.reference_min_records:
                    self.reference = DriftReference.fit(self._commissioning, self.cfg.bins, "self-commissioned",
                                                         self.cfg.reference_min_records)
                    self.self_commissioned = True
            return
        for ch, v in vals.items():
            self._window[ch].append(v)

    def evaluate(self, timestamp: datetime | None = None) -> DriftState:
        ts = timestamp or datetime.now(timezone.utc)
        if self.reference is None:
            n = max((len(v) for v in self._commissioning.values()), default=0)
            return DriftState(timestamp=ts, reference_status="BUILDING" if n else "NO_REFERENCE",
                              channels={ch: ChannelDrift(channel=ch, reason="no reference yet") for ch in DRIFT_CHANNELS})
        chans: dict[str, ChannelDrift] = {}
        scores: dict[str, float] = {}
        for ch in DRIFT_CHANNELS:
            w = list(self._window[ch])
            if ch not in self.reference.probs:
                chans[ch] = ChannelDrift(channel=ch, window_records=len(w), reason="channel not in the reference")
            elif len(w) < self.cfg.min_window_records:
                chans[ch] = ChannelDrift(channel=ch, window_records=len(w),
                                         reason=f"window has {len(w)} records (needs {self.cfg.min_window_records})")
            else:
                s = psi(self.reference.probs[ch], self.reference.edges[ch], w, self.cfg.epsilon)
                chans[ch] = ChannelDrift(channel=ch, psi=s, band=self.band(s), window_records=len(w),
                                         reason=("running-median residual (autocorrelated): not in the overall score"
                                                 if ch in AUTOCORRELATED_CHANNELS else None))
                scores[ch] = s
        overall = overall_score(scores)
        return DriftState(timestamp=ts, channels=chans, overall_psi=overall, overall_band=self.band(overall),
                          reference_version=self.reference.version, reference_status="READY")
