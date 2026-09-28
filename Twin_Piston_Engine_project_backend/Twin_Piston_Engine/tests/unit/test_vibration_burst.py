"""
Per-record 2048 Hz accelerometer and crank bursts (L1 -> L2).

    - No burst, or a burst shorter than window_size: every vibration feature
      is valid=False (never a single-sample RMS or a crest factor of 1.0).
    - Nominal simulator burst: crest factor 1.3-2.0, dominant order ~2.0
      (firing order of a four-cylinder four-stroke).
    - Misfire raises the 0.5X component and slows every second revolution.
    - Bearing wear adds an amplitude-modulated 300-1000 Hz carrier.
    - Bursts survive the integrity hash, HMAC signature, SQLite repository,
      and API ingestion; records without bursts keep their old hash.
"""

import hashlib
import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_telemetry_repository
from src.core.provenance import FaultClass, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_repository import InMemoryRawTelemetryRepository, SQLiteRawTelemetryRepository
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import (
    ForwardPhysicsModel,
    FaultScenarioConfig,
    SensorForwardModel,
)
from src.l1_data.simulator.vibration_burst import BEARING_CARRIER_HZ
from src.l1_data.telemetry_security import PacketSigner, PacketVerifier
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.vibration_processor import evaluate_vibration_processor

RPM = 4000.0


def _base_raw(**overrides) -> RawSignalRecord:
    fields = dict(
        sequence_number=1,
        source_type=Provenance.REAL,
        egt_cyl1_hot_uv=30000.0,
        egt_cyl2_hot_uv=30500.0,
        egt_cyl3_hot_uv=29800.0,
        egt_cyl4_hot_uv=30100.0,
        egt_cold_c=25.0,
        cht_hot_uv=12000.0,
        cht_cold_c=25.0,
        oil_rtd_ohms=135.0,
        oil_p_counts=2048,
        map_counts=2048,
        adc_vref_counts=4095,
        crank_period_us=15000.0,
        fuel_pulse_hz=100.0,
        accel_counts_xyz=(25, 15, 98),
        ambient_temp_c=20.0,
        ambient_press_pa=101325.0,
        signal_quality=SignalQuality(score=1.0),
    )
    fields.update(overrides)
    rec = RawSignalRecord(**fields)
    return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})


def _sim_raw(fault: FaultClass = FaultClass.NOMINAL, severity: float = 0.0,
             cylinders: tuple[int, ...] = (1,), enable_noise: bool = True) -> RawSignalRecord:
    scenario = None
    if fault != FaultClass.NOMINAL:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0,
                                       duration_s=10.0, affected_cylinders=list(cylinders))
    gt = ForwardPhysicsModel(seed=7).compute_ground_truth(
        time_s=1.0, rpm=RPM, map_pa=110000.0, fault_scenario=scenario)
    return SensorForwardModel(seed=7).convert_to_raw_record(gt, scenario, enable_noise=enable_noise)


def _features(raw: RawSignalRecord):
    return evaluate_vibration_processor(convert_raw_to_engineering_state(raw))


def _order_amplitude(samples, order: float, fs: float = 2048.0) -> float:
    x = np.asarray(samples, dtype=float)
    spec = 2.0 * np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) / (len(x) * 0.5)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    target = order * RPM / 60.0
    band = (freqs > target - 2.0) & (freqs < target + 2.0)
    return float(spec[band].max())


# --------------------------------------------------------------------------
# No / short burst -> invalid, never a single-sample result
# --------------------------------------------------------------------------

def test_record_without_burst_reports_all_features_invalid() -> None:
    state, res = _features(_base_raw())
    for tv in (state.rms_x_m_s2, state.rms_y_m_s2, state.rms_z_m_s2, state.overall_rms_m_s2,
               state.peak_m_s2, state.crest_factor, state.dominant_freq_hz,
               state.dominant_amplitude_m_s2, state.dominant_order):
        assert tv.valid is False
        assert tv.value is None
        assert tv.fault_flag
    assert state.crest_factor.value != 1.0
    assert res.valid is False
    assert not any(f.valid for f in res.axis_features.values())


def test_burst_shorter_than_window_is_not_analysed() -> None:
    n = 1024  # below window_size = 2048
    burst = tuple(int(50 * math.sin(2 * math.pi * 133.3 * i / 2048)) for i in range(n))
    state, res = _features(_base_raw(accel_burst_counts_x=burst, accel_burst_counts_y=burst,
                                     accel_burst_counts_z=burst, accel_burst_fs_hz=2048.0))
    assert state.crest_factor.valid is False
    assert state.crest_factor.value is None
    assert res.valid is False


def test_single_sample_channels_are_kept() -> None:
    eng = convert_raw_to_engineering_state(_base_raw())
    assert eng.vibration_x.valid and eng.vibration_x.value == pytest.approx(2.5)
    assert eng.vibration_burst_x.valid is False and len(eng.vibration_burst_x) == 0
    assert eng.crank_period_burst.valid is False


# --------------------------------------------------------------------------
# Simulator bursts
# --------------------------------------------------------------------------

def test_simulator_record_carries_bursts() -> None:
    raw = _sim_raw()
    assert len(raw.accel_burst_counts_x) == len(raw.accel_burst_counts_y) == len(raw.accel_burst_counts_z) == 2048
    assert raw.accel_burst_fs_hz == 2048.0
    assert len(raw.crank_period_burst_us) == 32
    eng = convert_raw_to_engineering_state(raw)
    assert eng.vibration_burst_x.valid and eng.vibration_burst_x.sample_rate_hz == 2048.0
    assert eng.vibration_burst_x.samples[0] == pytest.approx(raw.accel_burst_counts_x[0] / 10.0)
    assert eng.crank_period_burst.valid and len(eng.crank_period_burst) == 32


def test_nominal_burst_crest_factor_and_firing_order() -> None:
    state, res = _features(_sim_raw())
    assert res.valid and res.window_size == 2048
    assert state.crest_factor.valid
    assert 1.3 <= state.crest_factor.value <= 2.0
    assert state.dominant_order.valid
    assert state.dominant_order.value == pytest.approx(2.0, abs=0.05)


def test_misfire_raises_half_order() -> None:
    nominal = _sim_raw()
    misfire = _sim_raw(FaultClass.MISFIRE, severity=0.8)
    amp_nom = _order_amplitude(np.array(nominal.accel_burst_counts_x) / 10.0, 0.5)
    amp_mis = _order_amplitude(np.array(misfire.accel_burst_counts_x) / 10.0, 0.5)
    assert amp_mis > 5.0 * max(amp_nom, 0.05)


@pytest.mark.parametrize("cylinder, slowed", [(1, 0), (3, 0), (2, 1), (4, 1)])
def test_misfire_slows_every_second_revolution(cylinder: int, slowed: int) -> None:
    periods = np.array(_sim_raw(FaultClass.MISFIRE, 1.0, (cylinder,), enable_noise=False).crank_period_burst_us)
    nominal = 60.0e6 / RPM
    assert np.all(periods[slowed::2] > nominal * 1.01)
    assert np.allclose(periods[1 - slowed::2], nominal)


def test_nominal_crank_burst_is_uniform() -> None:
    periods = np.array(_sim_raw(enable_noise=False).crank_period_burst_us)
    assert np.allclose(periods, 60.0e6 / RPM)


def test_bearing_wear_adds_resonance_impacts() -> None:
    """Impacts at BPFO ring the 800 Hz structural resonance (was: a 600 Hz AM
    carrier that was also the dominant spectral line). Impact energy is spread
    over BPFO harmonics around the resonance, so the dominant line stays at the
    2X firing order; the check is the energy in the resonance band instead."""
    _, nom = _features(_sim_raw())
    _, brg = _features(_sim_raw(FaultClass.BEARING_WEAR, severity=0.8))
    x_nom, x_brg = nom.axis_features["X"], brg.axis_features["X"]
    assert x_brg.band_energy_high > 20.0 * max(x_nom.band_energy_high, 1e-6)
    x = np.asarray(_sim_raw(FaultClass.BEARING_WEAR, severity=0.8).accel_burst_counts_x, dtype=float) / 10.0
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / 2048.0)
    high = freqs >= 500.0
    peak_hz = freqs[high][np.argmax(spec[high])]
    assert abs(peak_hz - BEARING_CARRIER_HZ) < 150.0  # resonance region


# --------------------------------------------------------------------------
# Integrity, signature, persistence, API
# --------------------------------------------------------------------------

def test_hash_unchanged_for_records_without_bursts() -> None:
    rec = _base_raw()
    legacy = {
        "seq": rec.sequence_number, "egt1": rec.egt_cyl1_hot_uv, "egt2": rec.egt_cyl2_hot_uv,
        "egt3": rec.egt_cyl3_hot_uv, "egt4": rec.egt_cyl4_hot_uv, "egt_cold": rec.egt_cold_c,
        "cht": rec.cht_hot_uv, "cht_cold": rec.cht_cold_c, "oil_rtd": rec.oil_rtd_ohms,
        "oil_p": rec.oil_p_counts, "map": rec.map_counts, "vref": rec.adc_vref_counts,
        "crank_us": rec.crank_period_us, "fuel_hz": rec.fuel_pulse_hz,
        "accel": list(rec.accel_counts_xyz), "amb_t": rec.ambient_temp_c, "amb_p": rec.ambient_press_pa,
    }
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode("utf-8")).hexdigest()
    assert rec.compute_integrity_hash() == expected


def test_hash_and_signature_cover_bursts() -> None:
    raw = _sim_raw()
    tampered_x = list(raw.accel_burst_counts_x)
    tampered_x[100] += 1
    tampered = raw.model_copy(update={"accel_burst_counts_x": tuple(tampered_x)})
    assert tampered.compute_integrity_hash() != raw.compute_integrity_hash()

    sig = PacketSigner().sign_record(raw)
    assert PacketVerifier().verify_signature(raw, sig)
    assert not PacketVerifier().verify_signature(tampered, sig)
    crank = raw.model_copy(update={"crank_period_burst_us": raw.crank_period_burst_us[:-1] + (1.0,)})
    assert not PacketVerifier().verify_signature(crank, sig)


def test_mismatched_burst_axes_rejected() -> None:
    with pytest.raises(ValueError):
        _base_raw(accel_burst_counts_x=(1, 2, 3), accel_burst_counts_y=(1, 2),
                  accel_burst_counts_z=(1, 2, 3), accel_burst_fs_hz=2048.0)
    with pytest.raises(ValueError):
        _base_raw(accel_burst_counts_x=(1,), accel_burst_counts_y=(1,), accel_burst_counts_z=(1,))


def test_sqlite_repository_round_trips_bursts() -> None:
    raw = _sim_raw()
    repo = SQLiteRawTelemetryRepository(":memory:")
    assert repo.save(raw)
    loaded = repo.get_by_sequence(raw.sequence_number)
    assert loaded == raw
    assert loaded.compute_integrity_hash() == raw.integrity_hash


def test_api_ingest_carries_bursts() -> None:
    raw = _sim_raw().model_copy(update={"sequence_number": 987654})
    payload = json.loads(raw.model_dump_json(include={
        "sequence_number", "source_type", "egt_cyl1_hot_uv", "egt_cyl2_hot_uv", "egt_cyl3_hot_uv",
        "egt_cyl4_hot_uv", "egt_cold_c", "cht_hot_uv", "cht_cold_c", "oil_rtd_ohms", "oil_p_counts",
        "map_counts", "adc_vref_counts", "crank_period_us", "fuel_pulse_hz", "accel_counts_xyz",
        "ambient_temp_c", "ambient_press_pa", "accel_burst_counts_x", "accel_burst_counts_y",
        "accel_burst_counts_z", "accel_burst_fs_hz", "crank_period_burst_us",
    }))
    repo = InMemoryRawTelemetryRepository()
    app = create_app()
    app.dependency_overrides[get_telemetry_repository] = lambda: repo
    with TestClient(app) as client:
        response = client.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 201
    assert response.json()["accepted"] is True
    stored = repo.get_by_sequence(987654)
    assert stored.accel_burst_counts_x == raw.accel_burst_counts_x
    assert stored.accel_burst_fs_hz == 2048.0
    assert stored.crank_period_burst_us == raw.crank_period_burst_us


async def test_physics_simulator_adapter_attaches_bursts() -> None:
    from src.l1_data.adapters.simulator_adapter import PhysicsSimulatorAdapter

    async with PhysicsSimulatorAdapter(rpm=RPM) as adapter:
        raw = await adapter.read_next()
    assert len(raw.accel_burst_counts_x) == 2048 and raw.accel_burst_fs_hz == 2048.0
    assert len(raw.crank_period_burst_us) == 32
    assert raw.integrity_hash == raw.compute_integrity_hash()
    state, _ = _features(raw)
    assert state.dominant_order.value == pytest.approx(2.0, abs=0.05)
