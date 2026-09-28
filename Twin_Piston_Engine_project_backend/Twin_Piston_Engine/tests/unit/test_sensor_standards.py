"""
Sensor transfer functions against published standards, end to end.

    - Type K thermocouple: NIST ITS-90 (NIST Monograph 175) reference EMFs
    - Pt100 RTD:           IEC 60751 Callendar-Van Dusen reference resistances
    - Cold-junction compensation adds the cold junction's EMF, not its degrees
    - Round trip: simulator ground truth -> RawSignalRecord -> L2 sensor_inverse

Every conversion here goes through the production L2 path
(SensorInverseModel), not the core helpers directly.
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from src.core.provenance import FaultClass, Provenance
from src.core.schemas import SignalQuality
from src.core.sensor_physics import pt100_resistance_ohms, type_k_emf_uv
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    SensorForwardModel,
    SimulationGroundTruth,
)
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state

C_TO_K = 273.15


def _raw(
    egt_uv: float = 30000.0,
    egt_cold_c: float = 25.0,
    cht_uv: float = 12000.0,
    cht_cold_c: float = 25.0,
    oil_ohms: float = 135.0,
) -> RawSignalRecord:
    rec = RawSignalRecord(
        sequence_number=1,
        source_type=Provenance.REAL,
        egt_cyl1_hot_uv=egt_uv,
        egt_cyl2_hot_uv=egt_uv,
        egt_cyl3_hot_uv=egt_uv,
        egt_cyl4_hot_uv=egt_uv,
        egt_cold_c=egt_cold_c,
        cht_hot_uv=cht_uv,
        cht_cold_c=cht_cold_c,
        oil_rtd_ohms=oil_ohms,
        oil_p_counts=2048,
        map_counts=2048,
        adc_vref_counts=4095,
        crank_period_us=15000.0,
        fuel_pulse_hz=100.0,
        accel_counts_xyz=(30, 40, 0),
        ambient_temp_c=20.0,
        ambient_press_pa=101325.0,
        signal_quality=SignalQuality(score=1.0),
    )
    return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})


# --------------------------------------------------------------------------
# NIST ITS-90 Type K check points (cold junction 0 °C)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("emf_uv", "temp_c"),
    [(4096.0, 100.0), (20644.0, 500.0), (33275.0, 800.0), (48838.0, 1200.0)],
)
def test_type_k_nist_check_points(emf_uv: float, temp_c: float) -> None:
    eng = convert_raw_to_engineering_state(_raw(egt_uv=emf_uv, egt_cold_c=0.0,
                                                cht_uv=emf_uv, cht_cold_c=0.0))
    assert eng.egt_cyl_1.valid is True
    assert eng.egt_cyl_1.value - C_TO_K == pytest.approx(temp_c, abs=0.05)
    assert eng.cht_cyl_1.valid is True
    assert eng.cht_cyl_1.value - C_TO_K == pytest.approx(temp_c, abs=0.05)


def test_type_k_cold_junction_adds_emf_not_degrees() -> None:
    """Hot 800 °C, cold 25 °C. The linear model read 807.05 °C here."""
    measured_uv = type_k_emf_uv(800.0) - type_k_emf_uv(25.0)
    eng = convert_raw_to_engineering_state(_raw(egt_uv=measured_uv, egt_cold_c=25.0))
    assert eng.egt_cyl_1.valid is True
    assert eng.egt_cyl_1.value - C_TO_K == pytest.approx(800.0, abs=0.05)


def test_type_k_negative_microvolts_are_valid() -> None:
    """Cold engine in hot ambient: hot junction 10 °C, cold junction 40 °C."""
    measured_uv = type_k_emf_uv(10.0) - type_k_emf_uv(40.0)
    assert measured_uv < 0.0
    eng = convert_raw_to_engineering_state(_raw(egt_uv=measured_uv, egt_cold_c=40.0,
                                                cht_uv=measured_uv, cht_cold_c=40.0))
    assert eng.egt_cyl_1.valid is True
    assert eng.egt_cyl_1.value - C_TO_K == pytest.approx(10.0, abs=0.05)
    assert eng.cht_cyl_1.valid is True
    assert eng.cht_cyl_1.value - C_TO_K == pytest.approx(10.0, abs=0.05)


@pytest.mark.parametrize("emf_uv", [60000.0, -8000.0, math.inf, math.nan])
def test_type_k_outside_range_is_invalid_not_substituted(emf_uv: float) -> None:
    """Above 1372 °C / below -270 °C / non-finite: invalid, value NaN, reason given."""
    eng = convert_raw_to_engineering_state(_raw(egt_uv=emf_uv, cht_uv=emf_uv))
    for ch in (eng.egt_cyl_1, eng.cht_cyl_1):
        assert ch.valid is False
        assert math.isnan(ch.value)
        assert ch.fault_flag
    assert eng.oil_temp.valid is True  # failure isolated to the TC channels


# --------------------------------------------------------------------------
# IEC 60751 Pt100 check points
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("ohms", "temp_c"),
    [(138.505, 100.0), (157.325, 150.0), (84.271, -40.0)],
)
def test_pt100_iec_check_points(ohms: float, temp_c: float) -> None:
    eng = convert_raw_to_engineering_state(_raw(oil_ohms=ohms))
    assert eng.oil_temp.valid is True
    assert eng.oil_temp.value - C_TO_K == pytest.approx(temp_c, abs=0.01)


# --------------------------------------------------------------------------
# Round trip: simulator ground truth -> RawSignalRecord -> L2 inverse
# --------------------------------------------------------------------------

def _ground_truth(temp_k: float, ambient_k: float) -> SimulationGroundTruth:
    return SimulationGroundTruth(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        time_s=0.0,
        sequence_number=1,
        rpm=5000.0,
        map_pressure_pa=100000.0,
        throttle_pct=75.0,
        altitude_m=1000.0,
        ambient_temp_k=ambient_k,
        ambient_pressure_pa=101325.0,
        egt_k=[temp_k, temp_k + 1.0, temp_k - 1.0, temp_k + 0.5],
        cht_k=temp_k,
        oil_temp_k=temp_k,
    )


@pytest.mark.parametrize("ambient_k", [233.15, 288.15, 323.15])
def test_round_trip_simulator_to_l2_inverse(ambient_k: float) -> None:
    """EGT/CHT/oil round trip error < 0.05 K across 250-1450 K.

    Oil above 1123.15 K (850 °C) is outside the IEC 60751 Pt100 range; there
    the check only proves the forward and inverse CVD agree with each other.
    """
    forward = SensorForwardModel(seed=0)
    worst = 0.0
    for temp_k in np.linspace(250.0, 1450.0, 121):
        gt = _ground_truth(float(temp_k), ambient_k)
        raw = forward.convert_to_raw_record(gt, enable_noise=False)
        eng = convert_raw_to_engineering_state(raw)

        pairs = [
            (eng.egt_cyl_1, gt.egt_k[0]),
            (eng.egt_cyl_2, gt.egt_k[1]),
            (eng.egt_cyl_3, gt.egt_k[2]),
            (eng.egt_cyl_4, gt.egt_k[3]),
            (eng.cht_cyl_1, gt.cht_k),
            (eng.oil_temp, gt.oil_temp_k),
        ]
        for ch, truth in pairs:
            assert ch.valid is True, (temp_k, ch.fault_flag)
            worst = max(worst, abs(ch.value - truth))
    assert worst < 0.05, f"worst round-trip error {worst:.6f} K"


def test_simulator_forward_uses_junction_emf_difference() -> None:
    gt = _ground_truth(1073.15, 298.15)
    raw = SensorForwardModel(seed=0).convert_to_raw_record(gt, enable_noise=False)
    assert raw.egt_cyl1_hot_uv == pytest.approx(type_k_emf_uv(800.0) - type_k_emf_uv(25.0), abs=1e-6)
    assert raw.oil_rtd_ohms == pytest.approx(pt100_resistance_ohms(800.0), abs=1e-9)


def test_simulator_egt_dropout_still_invalid_in_l2() -> None:
    """Negative uV is no longer a fault signal; the dropout is flagged on the channel."""
    gt = _ground_truth(1000.0, 288.15)
    fault = FaultScenarioConfig(
        fault_class=FaultClass.SENSOR_FAULT,
        onset_time_s=0.0,
        duration_s=10.0,
        affected_channel="egt_cyl1_hot_uv",
    )
    raw = SensorForwardModel(seed=0).convert_to_raw_record(gt, fault, enable_noise=False)
    eng = convert_raw_to_engineering_state(raw)
    assert eng.egt_cyl_1.valid is False
    assert eng.egt_cyl_2.valid is True
