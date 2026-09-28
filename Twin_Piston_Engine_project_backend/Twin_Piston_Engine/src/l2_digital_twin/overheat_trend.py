"""
Overheating trend prediction (L2; Prompt 14; SIH C "overheating trends").

For CHT, each EGT, oil temperature and coolant temperature:

    expected(t)  = M-11 expectation at the operating point, passed through a
                   first-order lag (L2's own thermal time constant), so the
                   expected trajectory follows a power change the way the
                   engine does
    residual(t)  = measured(t) - expected(t)
    slope        = Theil-Sen slope of the residual over a rolling window, with
                   Sen's confidence interval (scipy.stats.theilslopes); this is
                   measured slope - expected slope (the M-09 slope-residual idea)
    time_to_limit_s = (limit - current) / slope, range from the CI bounds

A time-to-limit is reported only when the residual slope is positive, its
lower confidence bound is above zero, it exceeds min_slope_k_per_min, and the
measured temperature itself is rising (a flat or falling temperature cannot reach its
limit, however it compares with the expectation; the same practical
minimum slope applies). Otherwise it is valid=False
("no trend"). An abrupt operating-point change restarts the windows: the
residual steps there, which is an offset, not a rate (cf. M-09 steady-state
gating). A healthy climb raises CHT, but its
expected trajectory rises with it, so the residual slope stays near zero.
Limits are the Appendix B alarm limits held in config (VERIFY; see
OverheatTrendConfig).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable

import numpy as np
from scipy.stats import theilslopes

from src.core.config import AppSettings, get_settings
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import ChannelTrend, OperatingPoint, OverheatTrendState, ProvenanceTaggedValue, make_tagged
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, operating_point_from_record

NO_TREND = "no trend"
CHANNEL_LABELS = {"cht": "CHT", "oil_temp": "Oil temperature", "coolant": "Coolant temperature",
                  **{f"egt_cyl{i}": f"EGT cylinder {i}" for i in range(1, 5)}}


def _invalid(reason: str) -> ProvenanceTaggedValue[float | None]:
    return ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False,
                                 validity_reason=ChannelValidity.MISSING, quality=0.0, fault_flag=reason)


def _valid(v: float, q: float = 1.0) -> ProvenanceTaggedValue[float | None]:
    return make_tagged(v, Provenance.DERIVED, valid=True, quality=q)


def _median_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen point estimate (median pairwise slope) without the CI."""
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    keep = dt > 0.0
    return float(np.median((y[j] - y[i])[keep] / dt[keep])) if keep.any() else math.nan


class TrendWindow:
    """Rolling time window of samples (t, value columns...) with a Theil-Sen
    slope and Sen's confidence interval. Shared by the overheating trends and
    the L3 health-index trend (one implementation)."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.rows: deque[tuple[float, ...]] = deque()

    def clear(self) -> None:
        self.rows.clear()

    def add(self, t_s: float, *values: float) -> None:
        if self.rows and t_s <= self.rows[-1][0]:
            self.rows.clear()  # time went backwards: new sequence
        self.rows.append((t_s, *values))
        while self.rows and t_s - self.rows[0][0] > self.window_s:
            self.rows.popleft()

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def span_s(self) -> float:
        return self.rows[-1][0] - self.rows[0][0] if self.rows else 0.0

    def median_dt_s(self) -> float:
        t = np.asarray([r[0] for r in self.rows])
        return float(np.median(np.diff(t))) if len(t) > 1 else math.inf

    def fit(self, col: int = 0, confidence: float = 0.95) -> tuple[float, float, float]:
        """(slope, low, high) per second for value column col."""
        arr = np.asarray(self.rows)
        slope, _, lo, hi = theilslopes(arr[:, col + 1], arr[:, 0] - arr[0, 0], alpha=confidence)
        return float(slope), float(lo), float(hi)

    def median_slope(self, col: int = 0) -> float:
        arr = np.asarray(self.rows)
        return _median_slope(arr[:, 0] - arr[0, 0], arr[:, col + 1])


class OperatingPointStepDetector:
    """True when the operating point jumps between consecutive records (rpm,
    MAP or ambient step above the configured limits): a trend window must
    restart there, because the series steps (an offset, not a rate)."""

    def __init__(self, rpm_step: float, map_step_pa: float, ambient_step_k: float) -> None:
        self.limits = (rpm_step, map_step_pa, ambient_step_k)
        self._last: tuple[float, float, float] | None = None

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "OperatingPointStepDetector":
        c = settings.overheat
        return cls(c.reset_rpm_step, c.reset_map_step_pa, c.reset_ambient_step_k)

    def reset(self) -> None:
        self._last = None

    def changed(self, op: OperatingPoint | None) -> bool:
        if op is None:
            return False
        cur = (op.rpm, op.map_pressure_pa, op.ambient_temp_k)
        prev, self._last = self._last, cur
        return prev is not None and any(abs(a - b) > lim for a, b, lim in zip(cur, prev, self.limits))


class _Channel:
    def __init__(self, name: str, getter: Callable[[NormalizedSignalRecord], ChannelValue],
                 expectation: Callable[[OperatingPoint], float | None], lag_s: float, limit_k: float,
                 window_s: float) -> None:
        self.name, self.getter, self.expectation = name, getter, expectation
        self.lag_s, self.limit_k, self.window_s = lag_s, limit_k, window_s
        self.hist = TrendWindow(window_s)  # columns: residual, measured, expected
        self.lagged: float | None = None
        self.last_t: float | None = None

    def reset(self) -> None:
        self.hist.clear()
        self.lagged = self.last_t = None


class OverheatTrendModel:
    """Stateful rolling windows; call evaluate() once per record in time order."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.overheat
        exp = HealthyExpectationModel(self._settings)
        s, c = self._settings, self.cfg

        def egt(i: int):
            return lambda op: (exp.expected_egt_per_cylinder_k(op) or (None,) * 4)[i]

        self._steps = OperatingPointStepDetector.from_settings(self._settings)
        self._channels = [
            _Channel("cht", lambda r: r.cht_cyl_1, exp.expected_cht_k, c.lag_cht_s,
                     s.cooling.max_cht_c + 273.15, c.window_s),
            *(_Channel(f"egt_cyl{i + 1}", (lambda k: (lambda r: getattr(r, f"egt_cyl_{k}")))(i + 1), egt(i),
                       c.lag_egt_s, s.egt_diagnostics.egt_critical_temp_k, c.window_s) for i in range(4)),
            _Channel("oil_temp", lambda r: r.oil_temp, exp.expected_oil_temp_k, c.lag_oil_s,
                     s.lubrication.lhi_over_temp_limit_c + 273.15, c.window_s),
            _Channel("coolant", lambda r: r.coolant_temp, exp.expected_coolant_temp_k, c.lag_coolant_s,
                     s.cooling.max_coolant_c + 273.15, c.window_s),
        ]

    def reset_state(self) -> None:
        self._steps.reset()
        for ch in self._channels:
            ch.reset()

    def evaluate(self, record: NormalizedSignalRecord) -> OverheatTrendState:
        op = operating_point_from_record(record)
        op_ok = all(ch.valid for ch in (record.rpm, record.map_pressure, record.ambient_pressure, record.ambient_temp))
        t_s = record.timestamp.timestamp() if record.timestamp is not None else None
        trends: dict[str, ChannelTrend] = {}
        alerts: list[str] = []
        if self._steps.changed(op if op_ok else None):
            for ch in self._channels:
                ch.hist.clear()  # the lagged expectation carries on
        for ch in self._channels:
            trend = self._channel(ch, record, op, op_ok, t_s)
            trends[ch.name] = trend
            if trend.status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
                alerts.append(overheat_message(trend))
        rank = {DiagnosticStatus.NORMAL: 1, DiagnosticStatus.WARNING: 2, DiagnosticStatus.CRITICAL: 3}
        valid = [t.status for t in trends.values() if t.status != DiagnosticStatus.INVALID]
        return OverheatTrendState(timestamp=record.timestamp, channels=trends,
                                  status=max(valid, key=rank.get) if valid else DiagnosticStatus.INVALID,
                                  alerts=alerts)

    def _channel(self, ch: _Channel, record: NormalizedSignalRecord, op: OperatingPoint, op_ok: bool,
                 t_s: float | None) -> ChannelTrend:
        c = self.cfg
        meas = ch.getter(record)
        limit_c = ch.limit_k - 273.15
        if not meas.valid or meas.value is None:
            return ChannelTrend(channel=ch.name, limit_c=limit_c,
                                current_c=_invalid(meas.fault_flag or "channel invalid"),
                                time_to_limit_s=_invalid(meas.fault_flag or "channel invalid"))
        current_tv = _valid(meas.value - 273.15, meas.quality)
        expected = ch.expectation(op) if op_ok else None
        if expected is None or t_s is None:
            reason = "operating point invalid; no expectation" if t_s is not None else "no timestamp"
            return ChannelTrend(channel=ch.name, limit_c=limit_c, current_c=current_tv,
                                time_to_limit_s=_invalid(reason))

        # lagged expectation (starts at the expectation itself, never at the measurement)
        if ch.lagged is None or ch.last_t is None or t_s <= ch.last_t:
            ch.reset()
            ch.lagged = expected
        else:
            ch.lagged += (expected - ch.lagged) * (1.0 - math.exp(-(t_s - ch.last_t) / ch.lag_s))
        ch.last_t = t_s
        ch.hist.add(t_s, meas.value - ch.lagged, meas.value, ch.lagged)

        n = len(ch.hist)
        span = ch.hist.span_s
        base = dict(channel=ch.name, limit_c=limit_c, current_c=current_tv, samples=n)
        if n < c.min_samples or span < 0.5 * ch.window_s:
            return ChannelTrend(**base, time_to_limit_s=_invalid(
                f"{NO_TREND}: window has {n} samples over {span:.0f} s (needs {c.min_samples} over "
                f"{0.5 * ch.window_s:.0f} s)"), status=DiagnosticStatus.NORMAL)

        slope, lo, hi = ch.hist.fit(0, c.confidence)
        slope_m, lo_m, hi_m = slope * 60.0, lo * 60.0, hi * 60.0
        base.update(
            measured_slope_k_min=_valid(ch.hist.median_slope(1) * 60.0),
            expected_slope_k_min=_valid(ch.hist.median_slope(2) * 60.0),
            slope_residual_k_min=_valid(slope_m),
            slope_residual_ci_k_min=(lo_m, hi_m),
        )
        remaining = ch.limit_k - meas.value
        if remaining <= 0.0:
            return ChannelTrend(**base, time_to_limit_s=_valid(0.0), time_to_limit_range_s=(0.0, 0.0),
                                status=DiagnosticStatus.CRITICAL)
        measured_slope_m = base["measured_slope_k_min"].value
        if lo_m <= 0.0 or slope_m < c.min_slope_k_per_min or not measured_slope_m >= c.min_slope_k_per_min:
            why = ("slope residual not distinct from zero" if lo_m <= 0.0
                   else f"slope residual below {c.min_slope_k_per_min} K/min" if slope_m < c.min_slope_k_per_min
                   else f"measured temperature not rising by {c.min_slope_k_per_min} K/min "
                        f"({measured_slope_m:+.2f} K/min)")
            return ChannelTrend(**base, time_to_limit_s=_invalid(
                f"{NO_TREND}: {why} ({slope_m:+.2f} K/min, CI {lo_m:+.2f}..{hi_m:+.2f})"),
                status=DiagnosticStatus.NORMAL)
        ttl = remaining / slope
        ttl_range = (remaining / hi, remaining / lo)
        if ttl <= c.alert_horizon_s / 3.0:
            status = DiagnosticStatus.CRITICAL
        elif ttl <= c.alert_horizon_s:
            status = DiagnosticStatus.WARNING
        else:
            status = DiagnosticStatus.NORMAL
        return ChannelTrend(**base, time_to_limit_s=_valid(ttl), time_to_limit_range_s=ttl_range, status=status)


def overheat_message(trend: ChannelTrend) -> str:
    """Advisory text, e.g. 'CHT rising 0.8 K/min faster than expected for this
    power; projected to reach the alarm limit in about 14 min (range 9-22).'"""
    label = CHANNEL_LABELS.get(trend.channel, trend.channel)
    ttl = trend.time_to_limit_s
    if ttl.valid and ttl.value == 0.0:
        return f"{label} at or above the alarm limit ({trend.limit_c:.0f} C)."
    lo, hi = trend.time_to_limit_range_s or (ttl.value, ttl.value)
    return (f"{label} rising {trend.slope_residual_k_min.value:.1f} K/min faster than expected for this power; "
            f"projected to reach the alarm limit in about {ttl.value / 60.0:.0f} min "
            f"(range {lo / 60.0:.0f}-{hi / 60.0:.0f}).")
