"""
Vibration Processing Subsystem — Original Module 9.

Fifth processing stage of L2 Digital Twin.
Performs deterministic signal processing (time-domain feature extraction,
FFT spectral analysis, windowing, and RPM order tracking) on accelerometer telemetry.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (Module 5) or explicit accelerometer signal buffer
    - Output: VibrationState (canonical domain schema) & VibrationSignalProcessingResult
    - Zero simulator internals or ground truth dependencies
    - Zero misfire, combustion stability, ML fault classification, health index, or advisory logic
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, VibrationConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import ProvenanceTaggedValue, VibrationState, make_tagged
from src.core.sensor_physics.M05_nyquist_guard import band_is_representable
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.physics.M01_excess_kurtosis import excess_kurtosis
from src.l2_digital_twin.physics.M02_half_order_fraction import HALF_ORDER_HI, half_order_fraction
from src.l2_digital_twin.physics.M12_rotating_imbalance import lateral_vertical_ratio, one_x_velocity_fraction
from src.l2_digital_twin.physics.M03_firing_order_fraction import (
    FIRING_BAND_TOLERANCE,
    firing_order,
    firing_order_fraction,
)
from src.l2_digital_twin.physics.M04_envelope_demodulation import NyquistViolation, envelope_rms
from src.l2_digital_twin.physics import M10_lubrication_vibration_indices as M10

logger = get_logger(__name__)


class AxisVibrationFeatures(BaseModel):
    """Time-domain and spectral features for a single accelerometer axis."""

    axis: str
    valid: bool
    mean: float
    rms: float
    peak: float
    peak_to_peak: float
    std_dev: float
    variance: float
    crest_factor: float
    dominant_frequency_hz: float
    dominant_amplitude: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_energy: float
    band_energy_low: float
    band_energy_mid: float
    band_energy_high: float
    dominant_order: float | None = None
    # Condition features (M-01..M-04). None means "not computable" (degenerate
    # window, rpm unknown, band above Nyquist) and is never replaced by 0.0.
    excess_kurtosis: float | None = None
    half_order_fraction: float | None = None
    firing_order_fraction: float | None = None
    envelope_rms: float | None = None

    model_config = ConfigDict(frozen=True)


CONDITION_FEATURES = ("excess_kurtosis", "half_order_fraction", "firing_order_fraction", "envelope_rms")


class VibrationSignalProcessingResult(BaseModel):
    """Complete multi-axis vibration analysis result container."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    valid: bool = True  # False when no full-window burst was available
    sampling_frequency_hz: float
    window_size: int
    axis_features: dict[str, AxisVibrationFeatures]
    overall_rms_m_s2: float
    overall_peak_m_s2: float
    overall_crest_factor: float
    dominant_frequency_hz: float
    dominant_amplitude_m_s2: float
    dominant_order: float | None = None
    # Overall condition features: maximum over the X, Y, Z axes; None if no
    # axis produced a valid value.
    excess_kurtosis: float | None = None
    half_order_fraction: float | None = None
    firing_order_fraction: float | None = None
    envelope_rms_m_s2: float | None = None
    vibration_health_index: float | None = None
    vhi_band: str = "UNKNOWN"
    vhi_factors: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class VibrationProcessor:
    """Deterministic vibration signal processor for time-domain and FFT spectral analysis."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: VibrationConfig = self._settings.vibration

    def _apply_window(self, signal: np.ndarray, window_type: str) -> tuple[np.ndarray, float]:
        """Apply windowing function and compute coherent gain scale factor."""
        n = len(signal)
        if n == 0:
            return signal, 1.0

        w_type = window_type.lower()
        if w_type == "hanning":
            w = np.hanning(n)
        elif w_type == "hamming":
            w = np.hamming(n)
        else:
            w = np.ones(n)

        w_scale = float(np.mean(w)) if np.mean(w) > 0 else 1.0
        return signal * w, w_scale

    @staticmethod
    def _order_fraction_or_none(
        fn: Any, arr: np.ndarray, fs_hz: float, rpm: float | None, order_hi: float, **kwargs: Any
    ) -> float | None:
        """Order-band energy fraction, or None when it is not computable.

        M-02/M-03 return 0.0 for "no rpm", a degenerate window, or a band above
        Nyquist, which is indistinguishable from "no energy in the band". Those
        cases are detected here and reported as None instead.
        """
        if rpm is None or not math.isfinite(rpm) or rpm <= 0.0 or fs_hz <= 0.0:
            return None
        if arr.size < 8 or float(np.var(arr)) <= 0.0:
            return None
        if not band_is_representable(order_hi * rpm / 60.0, fs_hz):
            return None
        return float(fn(arr, fs_hz, rpm, **kwargs))

    def vhi_references(self, rpm: float) -> tuple[float, float, float]:
        """(RMS, crest, envelope) healthy references at this rpm, interpolated
        from VibrationConfig (provisional, simulator-derived; OI-9)."""
        c = self._cfg
        return (
            float(np.interp(rpm, c.vhi_ref_rpm, c.vhi_ref_rms_m_s2)),
            float(np.interp(rpm, c.vhi_ref_rpm, c.vhi_ref_crest)),
            float(np.interp(rpm, c.vhi_ref_rpm, c.vhi_ref_envelope_m_s2)),
        )

    def _envelope_rms_or_none(self, arr: np.ndarray, fs_hz: float) -> float | None:
        """M-04 envelope RMS in the configured band; None if not computable."""
        if arr.size < 16:
            return None
        try:
            return envelope_rms(arr, fs_hz, self._cfg.envelope_band_low_hz, self._cfg.envelope_band_high_hz,
                                self._cfg.envelope_taper_hz, self._cfg.envelope_time_taper_alpha)
        except NyquistViolation:
            return None

    def compute_axis_features(
        self,
        samples: Sequence[float],
        fs_hz: float,
        axis_name: str = "X",
        rpm: float | None = None,
    ) -> AxisVibrationFeatures:
        """Compute time-domain and FFT spectral features for a single sample array."""
        if len(samples) == 0 or fs_hz <= 0 or not all(math.isfinite(s) for s in samples):
            return AxisVibrationFeatures(
                axis=axis_name,
                valid=False,
                mean=0.0,
                rms=0.0,
                peak=0.0,
                peak_to_peak=0.0,
                std_dev=0.0,
                variance=0.0,
                crest_factor=0.0,
                dominant_frequency_hz=0.0,
                dominant_amplitude=0.0,
                spectral_centroid_hz=0.0,
                spectral_bandwidth_hz=0.0,
                spectral_energy=0.0,
                band_energy_low=0.0,
                band_energy_mid=0.0,
                band_energy_high=0.0,
                dominant_order=None,
            )

        arr = np.array(samples, dtype=float)
        n = len(arr)

        # 1. Time-domain statistics
        mean_val = float(np.mean(arr))
        detrended = arr - mean_val
        var_val = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        rms_val = float(np.sqrt(np.mean(arr ** 2)))
        peak_val = float(np.max(np.abs(arr)))
        p2p_val = float(np.max(arr) - np.min(arr))
        cf_val = (peak_val / rms_val) if rms_val > 0 else 0.0

        # 2. Spectral (FFT) analysis
        windowed, w_scale = self._apply_window(detrended, self._cfg.window_function)
        fft_complex = np.fft.rfft(windowed)
        n_half = len(fft_complex)

        mags = (2.0 * np.abs(fft_complex)) / (n * w_scale)
        if n_half > 0:
            mags[0] = np.abs(fft_complex[0]) / n

        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

        ac_mags = mags[1:] if n_half > 1 else mags
        ac_freqs = freqs[1:] if n_half > 1 else freqs

        if len(ac_mags) > 0 and np.max(ac_mags) > 0:
            max_idx = int(np.argmax(ac_mags))
            dom_freq = float(ac_freqs[max_idx])
            dom_amp = float(ac_mags[max_idx])

            sum_mag = float(np.sum(ac_mags))
            if sum_mag > 0:
                spec_centroid = float(np.sum(ac_freqs * ac_mags) / sum_mag)
                spec_bandwidth = float(np.sqrt(np.sum(((ac_freqs - spec_centroid) ** 2) * ac_mags) / sum_mag))
            else:
                spec_centroid = 0.0
                spec_bandwidth = 0.0

            spec_energy = float(np.sum(ac_mags ** 2))

            low_mask = (ac_freqs >= 0.0) & (ac_freqs < self._cfg.freq_band_low_max_hz)
            mid_mask = (ac_freqs >= self._cfg.freq_band_low_max_hz) & (ac_freqs < self._cfg.freq_band_mid_max_hz)
            high_mask = (ac_freqs >= self._cfg.freq_band_mid_max_hz) & (ac_freqs <= self._cfg.freq_band_high_max_hz)

            band_low = float(np.sum(ac_mags[low_mask] ** 2)) if np.any(low_mask) else 0.0
            band_mid = float(np.sum(ac_mags[mid_mask] ** 2)) if np.any(mid_mask) else 0.0
            band_high = float(np.sum(ac_mags[high_mask] ** 2)) if np.any(high_mask) else 0.0
        else:
            dom_freq = 0.0
            dom_amp = 0.0
            spec_centroid = 0.0
            spec_bandwidth = 0.0
            spec_energy = 0.0
            band_low = 0.0
            band_mid = 0.0
            band_high = 0.0

        # 3. Order context
        dom_order: float | None = None
        if rpm is not None and rpm > 0:
            f_rot = rpm / 60.0
            if f_rot > 0:
                dom_order = dom_freq / f_rot

        # 4. Condition features (M-01..M-04)
        n_cyl = self._settings.engine.num_cylinders
        kurt = excess_kurtosis(arr)
        half_frac = self._order_fraction_or_none(half_order_fraction, arr, fs_hz, rpm, HALF_ORDER_HI)
        firing_frac = self._order_fraction_or_none(
            firing_order_fraction, arr, fs_hz, rpm,
            firing_order(n_cyl) * (1.0 + FIRING_BAND_TOLERANCE), n_cylinders=n_cyl,
        )
        env_rms = self._envelope_rms_or_none(arr, fs_hz)

        return AxisVibrationFeatures(
            axis=axis_name,
            valid=True,
            mean=mean_val,
            rms=rms_val,
            peak=peak_val,
            peak_to_peak=p2p_val,
            std_dev=std_val,
            variance=var_val,
            crest_factor=cf_val,
            dominant_frequency_hz=dom_freq,
            dominant_amplitude=dom_amp,
            spectral_centroid_hz=spec_centroid,
            spectral_bandwidth_hz=spec_bandwidth,
            spectral_energy=spec_energy,
            band_energy_low=band_low,
            band_energy_mid=band_mid,
            band_energy_high=band_high,
            dominant_order=dom_order,
            excess_kurtosis=kurt,
            half_order_fraction=half_frac,
            firing_order_fraction=firing_frac,
            envelope_rms=env_rms,
        )

    def process_window(
        self,
        x_samples: Sequence[float],
        y_samples: Sequence[float],
        z_samples: Sequence[float],
        fs_hz: float | None = None,
        rpm: float | None = None,
        timestamp: datetime | None = None,
    ) -> tuple[VibrationState, VibrationSignalProcessingResult]:
        """Process a multi-axis sample window."""
        fs = fs_hz if (fs_hz is not None and fs_hz > 0) else self._cfg.sampling_frequency_hz
        ts = timestamp or datetime.now()

        x_feat = self.compute_axis_features(x_samples, fs, "X", rpm)
        y_feat = self.compute_axis_features(y_samples, fs, "Y", rpm)
        z_feat = self.compute_axis_features(z_samples, fs, "Z", rpm)

        if x_feat.valid and y_feat.valid and z_feat.valid and len(x_samples) == len(y_samples) == len(z_samples):
            rms_arr = np.sqrt(np.array(x_samples)**2 + np.array(y_samples)**2 + np.array(z_samples)**2)
            comb_feat = self.compute_axis_features(rms_arr, fs, "RMS", rpm)
        else:
            comb_feat = self.compute_axis_features([], fs, "RMS", rpm)

        axis_map = {"X": x_feat, "Y": y_feat, "Z": z_feat, "RMS": comb_feat}

        valid_feats = [f for f in (x_feat, y_feat, z_feat) if f.valid]
        if valid_feats:
            top_feat = max(valid_feats, key=lambda f: f.dominant_amplitude)
            dom_freq = top_feat.dominant_frequency_hz
            dom_amp = top_feat.dominant_amplitude
            dom_order = top_feat.dominant_order
        else:
            dom_freq = 0.0
            dom_amp = 0.0
            dom_order = None

        # Overall condition features: max over the three physical axes.
        overall: dict[str, float | None] = {}
        for name in CONDITION_FEATURES:
            vals = [getattr(f, name) for f in (x_feat, y_feat, z_feat)
                    if f.valid and getattr(f, name) is not None]
            overall[name] = max(vals) if vals else None

        # Vibration Health Index (M-10). References depend on rpm.
        vhi_res = None
        if comb_feat.valid and rpm is not None and rpm > 0 and overall["envelope_rms"] is not None:
            r_rms, r_crest, r_env = self.vhi_references(rpm)
            vhi_res = M10.vibration_health_index(
                comb_feat.rms, r_rms, comb_feat.crest_factor, r_crest, overall["envelope_rms"], r_env
            )
        vhi_val = vhi_res.value if (vhi_res is not None and vhi_res.valid) else None

        def _cond(name: str) -> ProvenanceTaggedValue[Any]:
            v = overall[name]
            return make_tagged(v, Provenance.DERIVED, valid=v is not None, quality=1.0 if v is not None else 0.0)

        result = VibrationSignalProcessingResult(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            sampling_frequency_hz=fs,
            window_size=len(x_samples),
            axis_features=axis_map,
            overall_rms_m_s2=comb_feat.rms if comb_feat.valid else 0.0,
            overall_peak_m_s2=comb_feat.peak if comb_feat.valid else 0.0,
            overall_crest_factor=comb_feat.crest_factor if comb_feat.valid else 0.0,
            dominant_frequency_hz=dom_freq,
            dominant_amplitude_m_s2=dom_amp,
            dominant_order=dom_order,
            excess_kurtosis=overall["excess_kurtosis"],
            half_order_fraction=overall["half_order_fraction"],
            firing_order_fraction=overall["firing_order_fraction"],
            envelope_rms_m_s2=overall["envelope_rms"],
            vibration_health_index=vhi_val,
            vhi_band=M10.vhi_band(vhi_val),
            vhi_factors=dict(vhi_res.factors) if vhi_val is not None else {},
        )

        state = VibrationState(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            rms_x_m_s2=make_tagged(x_feat.rms, Provenance.DERIVED, valid=x_feat.valid, quality=1.0 if x_feat.valid else 0.0),
            rms_y_m_s2=make_tagged(y_feat.rms, Provenance.DERIVED, valid=y_feat.valid, quality=1.0 if y_feat.valid else 0.0),
            rms_z_m_s2=make_tagged(z_feat.rms, Provenance.DERIVED, valid=z_feat.valid, quality=1.0 if z_feat.valid else 0.0),
            overall_rms_m_s2=make_tagged(comb_feat.rms, Provenance.DERIVED, valid=comb_feat.valid, quality=1.0 if comb_feat.valid else 0.0),
            peak_m_s2=make_tagged(comb_feat.peak, Provenance.DERIVED, valid=comb_feat.valid, quality=1.0 if comb_feat.valid else 0.0),
            crest_factor=make_tagged(comb_feat.crest_factor, Provenance.DERIVED, valid=comb_feat.valid, quality=1.0 if comb_feat.valid else 0.0),
            dominant_freq_hz=make_tagged(dom_freq, Provenance.DERIVED, valid=bool(valid_feats), quality=1.0 if valid_feats else 0.0),
            dominant_amplitude_m_s2=make_tagged(dom_amp, Provenance.DERIVED, valid=bool(valid_feats), quality=1.0 if valid_feats else 0.0),
            dominant_order=make_tagged(dom_order, Provenance.DERIVED, valid=(dom_order is not None), quality=1.0 if dom_order is not None else 0.0),
            excess_kurtosis=_cond("excess_kurtosis"),
            half_order_fraction=_cond("half_order_fraction"),
            firing_order_fraction=_cond("firing_order_fraction"),
            envelope_rms_m_s2=_cond("envelope_rms"),
            vibration_health_index=make_tagged(vhi_val, Provenance.DERIVED, valid=vhi_val is not None,
                                               quality=1.0 if vhi_val is not None else 0.0),
            vhi_band=M10.vhi_band(vhi_val),
            vhi_dominant_factor=vhi_res.dominant_factor if vhi_val is not None else "none",
        )

        return state, result

    def process_record(
        self, record: NormalizedSignalRecord
    ) -> tuple[VibrationState, VibrationSignalProcessingResult]:
        """Process the record's accelerometer burst.

        Spectral and time-domain features are computed only over the per-record
        burst sampled at its own rate (vibration_burst_*.sample_rate_hz), and
        only when every axis holds at least window_size samples. The last
        window_size samples are used. Otherwise every feature is valid=False.

        Features are never computed over fewer than window_size samples, and
        never over single samples accumulated across 1 Hz records: those would
        span ~34 minutes, not one analysis window.
        """
        rpm_val = record.rpm.value if (record.rpm.valid and record.rpm.value > 0) else None
        bursts = (record.vibration_burst_x, record.vibration_burst_y, record.vibration_burst_z)
        n_win = self._cfg.window_size
        fs = bursts[0].sample_rate_hz

        usable = (
            all(b.valid and len(b) >= n_win for b in bursts)
            and fs > 0.0
            and all(b.sample_rate_hz == fs for b in bursts)
        )
        if not usable:
            state, result = self._invalid_result(record, bursts, n_win)
            return self._with_imbalance(state, record, None, 0.0), result

        state, result = self.process_window(
            bursts[0].samples[-n_win:],
            bursts[1].samples[-n_win:],
            bursts[2].samples[-n_win:],
            fs_hz=fs,
            rpm=rpm_val,
            timestamp=record.timestamp,
        )
        axes = {"x": bursts[0].samples[-n_win:], "y": bursts[1].samples[-n_win:], "z": bursts[2].samples[-n_win:]}
        return self._with_imbalance(state, record, axes, fs), result

    def _with_imbalance(self, state: VibrationState, record: NormalizedSignalRecord,
                        axes: dict[str, Any] | None, fs: float) -> VibrationState:
        """M-12: propeller-shaft 1X velocity dominance on the lateral axis and
        the lateral/vertical 1X ratio (SRD-FUN-070)."""
        def bad(reason: str) -> ProvenanceTaggedValue[Any]:
            return make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0).model_copy(
                update={"validity_reason": ChannelValidity.MISSING, "fault_flag": reason})

        prop = record.propeller_speed
        if axes is None:
            reason = "accelerometer burst unavailable"
        elif not (prop.valid and prop.value and prop.value > 0.0):
            reason = f"propeller speed invalid ({prop.fault_flag or 'no value'})"
        else:
            reason = ""
        if reason:
            return state.model_copy(update={"prop_1x_hz": bad(reason),
                                            "prop_1x_velocity_fraction_lateral": bad(reason),
                                            "prop_1x_lateral_vertical_ratio": bad(reason)})
        f1 = prop.value / 60.0
        lat, vert = axes[self._cfg.axis_lateral], axes[self._cfg.axis_vertical]
        frac = one_x_velocity_fraction(lat, fs, f1)
        ratio = lateral_vertical_ratio(lat, vert, fs, f1)
        tag = lambda v, why: (make_tagged(v, Provenance.DERIVED, valid=True, quality=prop.quality)  # noqa: E731
                              if v is not None else bad(why))
        return state.model_copy(update={
            "prop_1x_hz": tag(f1, ""),
            "prop_1x_velocity_fraction_lateral": tag(frac, "1X band not representable in the burst"),
            "prop_1x_lateral_vertical_ratio": tag(ratio, "1X band not representable or no vertical 1X energy"),
        })

    def _invalid_result(
        self, record: NormalizedSignalRecord, bursts: tuple[Any, ...], n_win: int
    ) -> tuple[VibrationState, VibrationSignalProcessingResult]:
        """All features valid=False with value None; no single-sample fallback."""
        lengths = [len(b) for b in bursts]
        reason = (
            f"Accelerometer burst unavailable or shorter than window_size={n_win} "
            f"(samples per axis: {lengths})"
        )
        logger.debug(reason)
        empty = self.compute_axis_features([], self._cfg.sampling_frequency_hz, "X")

        def _undef() -> ProvenanceTaggedValue[Any]:
            tv = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0)
            return tv.model_copy(update={
                "validity_reason": ChannelValidity.MISSING,
                "fault_flag": reason,
            })

        result = VibrationSignalProcessingResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            valid=False,
            sampling_frequency_hz=0.0,
            window_size=0,
            axis_features={
                ax: empty.model_copy(update={"axis": ax}) for ax in ("X", "Y", "Z", "RMS")
            },
            overall_rms_m_s2=0.0,
            overall_peak_m_s2=0.0,
            overall_crest_factor=0.0,
            dominant_frequency_hz=0.0,
            dominant_amplitude_m_s2=0.0,
            dominant_order=None,
        )
        state = VibrationState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            rms_x_m_s2=_undef(),
            rms_y_m_s2=_undef(),
            rms_z_m_s2=_undef(),
            overall_rms_m_s2=_undef(),
            peak_m_s2=_undef(),
            crest_factor=_undef(),
            dominant_freq_hz=_undef(),
            dominant_amplitude_m_s2=_undef(),
            dominant_order=_undef(),
            excess_kurtosis=_undef(),
            half_order_fraction=_undef(),
            firing_order_fraction=_undef(),
            envelope_rms_m_s2=_undef(),
            vibration_health_index=_undef(),
        )
        return state, result


def evaluate_vibration_processor(
    record: NormalizedSignalRecord,
    settings: AppSettings | None = None,
) -> tuple[VibrationState, VibrationSignalProcessingResult]:
    """Convenience function for Module 9 vibration processing on a telemetry record."""
    processor = VibrationProcessor(settings)
    return processor.process_record(record)
