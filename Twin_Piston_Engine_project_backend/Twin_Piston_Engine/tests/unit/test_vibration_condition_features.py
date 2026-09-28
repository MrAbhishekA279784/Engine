"""
Vibration condition features (M-01..M-04) and the M-05 Nyquist guard.

    - Excess kurtosis: Gaussian ~0, impulsive > 3, undefined -> valid=False.
    - Half-order (0.5X) fraction: the misfire signature.
    - Firing-order (2X) fraction: steady across rpm on a healthy engine.
    - Envelope RMS (300-1000 Hz): bearing early warning, rises before overall RMS.
    - Order fractions need rpm; without it they are valid=False, never 0.0.
    - Settings with a band at or above fs/2 fail to load.
"""

import numpy as np
import pytest
import yaml

from src.core.config import load_settings
from src.core.exceptions import ConfigurationError
from src.core.provenance import FaultClass
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
)
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.vibration_processor import VibrationProcessor, evaluate_vibration_processor

FS = 2048.0
N = 2048
NEW_STATE_FIELDS = ("excess_kurtosis", "half_order_fraction", "firing_order_fraction", "envelope_rms_m_s2")


def _sim(rpm: float, fault: FaultClass = FaultClass.NOMINAL, severity: float = 0.0):
    scenario = None
    if fault != FaultClass.NOMINAL:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0, duration_s=10.0)
    gt = ForwardPhysicsModel(seed=7).compute_ground_truth(
        time_s=1.0, rpm=rpm, map_pa=110000.0, fault_scenario=scenario)
    raw = SensorForwardModel(seed=7).convert_to_raw_record(gt, scenario)
    return evaluate_vibration_processor(convert_raw_to_engineering_state(raw))


# --------------------------------------------------------------------------
# M-01 excess kurtosis
# --------------------------------------------------------------------------

def test_gaussian_burst_has_near_zero_excess_kurtosis() -> None:
    samples = np.random.default_rng(1).normal(0.0, 1.0, N)
    feat = VibrationProcessor().compute_axis_features(samples, FS, "X", rpm=4000.0)
    assert abs(feat.excess_kurtosis) < 0.5


def test_impulsive_burst_has_high_excess_kurtosis() -> None:
    samples = np.random.default_rng(1).normal(0.0, 1.0, N)
    samples[::128] += 12.0  # periodic impacts, as from a spalled race
    feat = VibrationProcessor().compute_axis_features(samples, FS, "X", rpm=4000.0)
    assert feat.excess_kurtosis > 3.0


def test_undefined_kurtosis_is_invalid_not_zero() -> None:
    flat = [1.0] * N  # dead channel: zero variance
    state, res = VibrationProcessor().process_window(flat, flat, flat, fs_hz=FS, rpm=4000.0)
    assert res.axis_features["X"].excess_kurtosis is None
    assert state.excess_kurtosis.valid is False
    assert state.excess_kurtosis.value is None
    # Order fractions of a flat line are not "zero energy", they are undefined.
    assert state.half_order_fraction.valid is False
    assert state.firing_order_fraction.valid is False


# --------------------------------------------------------------------------
# M-02 / M-03 order fractions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rpm", [2000.0, 3500.0, 5500.0])
def test_misfire_half_order_fraction_at_least_50x_nominal(rpm: float) -> None:
    nominal, _ = _sim(rpm)
    misfire, _ = _sim(rpm, FaultClass.MISFIRE, 0.8)
    assert nominal.half_order_fraction.valid and misfire.half_order_fraction.valid
    assert misfire.half_order_fraction.value >= 50.0 * nominal.half_order_fraction.value


def test_firing_order_fraction_steady_across_rpm() -> None:
    fractions = [_sim(rpm)[0].firing_order_fraction.value for rpm in (2000.0, 3000.0, 4000.0, 5000.0, 5800.0)]
    assert min(fractions) > 0.8
    assert max(fractions) - min(fractions) < 0.02


def test_order_fractions_invalid_without_rpm() -> None:
    samples = np.random.default_rng(2).normal(0.0, 1.0, N)
    state, res = VibrationProcessor().process_window(samples, samples, samples, fs_hz=FS, rpm=None)
    assert state.half_order_fraction.valid is False and state.half_order_fraction.value is None
    assert state.firing_order_fraction.valid is False and state.firing_order_fraction.value is None
    assert res.axis_features["X"].half_order_fraction is None
    # rpm-independent features are still computed
    assert state.excess_kurtosis.valid and state.envelope_rms_m_s2.valid


def test_order_band_above_nyquist_is_invalid() -> None:
    """At fs=256 Hz and 5800 rpm, 2X*1.15 = 222 Hz > 128 Hz: not representable."""
    samples = np.random.default_rng(3).normal(0.0, 1.0, 256)
    feat = VibrationProcessor().compute_axis_features(samples, 256.0, "X", rpm=5800.0)
    assert feat.firing_order_fraction is None


# --------------------------------------------------------------------------
# M-04 envelope RMS
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rpm", [2000.0, 4000.0, 5500.0])
def test_bearing_wear_raises_envelope_before_overall_rms(rpm: float) -> None:
    nominal, _ = _sim(rpm)
    bearing, _ = _sim(rpm, FaultClass.BEARING_WEAR, 0.05)
    env_ratio = bearing.envelope_rms_m_s2.value / nominal.envelope_rms_m_s2.value
    rms_ratio = bearing.overall_rms_m_s2.value / nominal.overall_rms_m_s2.value
    assert env_ratio >= 2.0
    assert rms_ratio < 1.3


# --------------------------------------------------------------------------
# Overall = max over axes; provenance; no-burst path
# --------------------------------------------------------------------------

def test_overall_feature_is_max_over_axes() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, N)
    y = x.copy()
    y[::128] += 12.0
    z = rng.normal(0.0, 1.0, N)
    state, res = VibrationProcessor().process_window(x, y, z, fs_hz=FS, rpm=4000.0)
    for state_name, feat_name in zip(NEW_STATE_FIELDS, ("excess_kurtosis", "half_order_fraction",
                                                         "firing_order_fraction", "envelope_rms")):
        axis_vals = [getattr(res.axis_features[a], feat_name) for a in ("X", "Y", "Z")]
        assert getattr(state, state_name).value == pytest.approx(max(axis_vals))
        assert getattr(state, state_name).provenance.value == "DERIVED"
    assert res.excess_kurtosis == pytest.approx(res.axis_features["Y"].excess_kurtosis)


def test_record_without_burst_reports_new_features_invalid() -> None:
    from tests.unit.test_vibration_burst import _base_raw

    state, res = evaluate_vibration_processor(convert_raw_to_engineering_state(_base_raw()))
    for name in NEW_STATE_FIELDS:
        tv = getattr(state, name)
        assert tv.valid is False and tv.value is None
    assert res.excess_kurtosis is None and res.envelope_rms_m_s2 is None


# --------------------------------------------------------------------------
# M-05 Nyquist guard at settings load
# --------------------------------------------------------------------------

def _load_with_vibration(tmp_path, **vibration) -> None:
    base = yaml.safe_load(open("config/default.yaml", encoding="utf-8"))
    base.setdefault("vibration", {}).update(vibration)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return load_settings(path)


def test_default_vibration_config_loads(tmp_path) -> None:
    settings = _load_with_vibration(tmp_path)
    assert settings.vibration.envelope_band_high_hz < settings.vibration.sampling_frequency_hz / 2.0


def test_envelope_band_above_nyquist_fails_to_load(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="envelope_band_high_hz"):
        _load_with_vibration(tmp_path, sampling_frequency_hz=2048.0,
                             envelope_band_low_hz=2000.0, envelope_band_high_hz=10000.0)


@pytest.mark.parametrize("field, value", [
    ("envelope_band_high_hz", 1024.0),   # exactly fs/2
    ("freq_band_high_max_hz", 1024.0),
    ("freq_band_high_max_hz", 2000.0),
])
def test_band_edge_at_or_above_nyquist_fails_to_load(tmp_path, field: str, value: float) -> None:
    with pytest.raises(ConfigurationError, match=field):
        _load_with_vibration(tmp_path, sampling_frequency_hz=2048.0, **{field: value})
