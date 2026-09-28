"""
End-to-end physics audit across fault modes (Prompt 9).

Each scenario runs the simulator for 120 s (1 Hz, 4000 rpm, 110 kPa; fault
onset at 30 s) through the REAL pipeline: RawSignalRecord -> HMAC sign ->
TelemetryValidator -> sensor_inverse -> L2 twin, diagnostics and health
indices (stateful across records). Indicators are read at t = 119 s and
compared with NOMINAL under the same seed. The same records also go through
the replay pipeline and the FastAPI app.

Strict xfails record indicators that did NOT behave as intended, with the
measured values; see docs/PHYSICS_AUDIT_RESULTS.md and docs/OPEN_ITEMS.md.
No threshold was tuned for these tests.
"""

from functools import lru_cache

import pytest

from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from tests.scientific.physics_audit_harness import (
    FAULT_ONSET_S,
    SCENARIOS,
    api_responses,
    invalid_values_are_null,
    json_has_nan,
    run_real_pipeline,
    run_replay_pipeline,
    simulate,
)

BY_NAME = {s.name: s for s in SCENARIOS}


@lru_cache(maxsize=None)
def records(name: str):
    return tuple(simulate(BY_NAME[name]))


@lru_cache(maxsize=None)
def snaps(name: str):
    return tuple(run_real_pipeline(list(records(name))))


def last(name: str):
    return snaps(name)[-1]


# --------------------------------------------------------------------------
# L1: every record signed, verified and accepted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_all_records_signed_verified_accepted(name: str) -> None:
    assert all(s.accepted for s in snaps(name))


def test_tampered_record_is_rejected() -> None:
    rec = records("NOMINAL")[10]
    sig = PacketSigner().sign_record(rec)
    tampered = rec.model_copy(update={"egt_cyl1_hot_uv": rec.egt_cyl1_hot_uv + 500.0})
    assert not TelemetryValidator().validate_packet(tampered, signature=sig).accepted


# --------------------------------------------------------------------------
# NOMINAL
# --------------------------------------------------------------------------

def test_nominal_all_bands_normal_after_windows_fill() -> None:
    """From t = 30 s (CSI window full) to 119 s, no WARNING/ALARM/CRITICAL
    in combustion, CSI, LHI, VHI, EGT or lubrication status."""
    steady = snaps("NOMINAL")[int(FAULT_ONSET_S):]
    offenders = [(i, s.bad_bands()) for i, s in enumerate(steady) if s.bad_bands()]
    assert offenders == []
    end = last("NOMINAL")
    assert (end.verdict, end.csi_band, end.lhi_band, end.vhi_band) == ("NORMAL", "NORMAL", "NORMAL", "NORMAL")


def test_nominal_no_alarm_from_first_record() -> None:
    """Was a strict xfail (OI-13: CSI ALARM at records 2-4, max 1.11). The thermal
    term is now excluded until the 30 s CHT window is full."""
    offenders = [(i, s.bad_bands()) for i, s in enumerate(snaps("NOMINAL")) if s.bad_bands()]
    assert offenders == []


# --------------------------------------------------------------------------
# Fault modes: the intended indicator moves
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["MISFIRE 0.3", "MISFIRE 1.0"])
def test_misfire_confirmed_and_csi_up(name: str) -> None:
    """MISFIRE 0.3 / 1.0: CONFIRMED; CSI 0.0142 -> 0.0172 / 0.0242 (crank term, OI-11)."""
    end, nom = last(name), last("NOMINAL")
    assert end.verdict == "CONFIRMED"
    assert end.csi > nom.csi


def test_misfire_1_0_localised_to_cylinder_1() -> None:
    assert last("MISFIRE 1.0").misfire_detected == [True, False, False, False]


def test_bearing_wear_envelope_and_vhi_up_misfire_not_confirmed() -> None:
    """BEARING_WEAR 0.05: envelope 0.075 -> 1.215 m/s^2, VHI 1.05 -> 30.7."""
    end, nom = last("BEARING_WEAR 0.05"), last("NOMINAL")
    assert end.envelope_rms > 5.0 * nom.envelope_rms
    assert end.vhi > 1.8 > nom.vhi
    assert end.verdict != "CONFIRMED"


def test_oil_degradation_lhi_down() -> None:
    """OIL_DEGRADATION 1.0: LHI 1.034 -> 0.369 (ALARM)."""
    assert last("OIL_DEGRADATION 1.0").lhi < 0.8 < last("NOMINAL").lhi


@pytest.mark.xfail(strict=True, reason=(
    "OI-22: COOLING_FAULT is a physical radiator blockage since Prompt 14 (was a +35 K CHT offset). "
    "89 s after onset at 4000 rpm / 110 kPa the CHT residual moved -45.6 -> -35.1 K (+10.4 K): the "
    "coolant loop heats over minutes and the thermostat absorbs part of the loss."))
def test_cooling_fault_cht_residual_up() -> None:
    """COOLING_FAULT 1.0: CHT residual rises by more than 20 K. The absolute
    residual stays negative because the expectation is uncalibrated (OI-5)."""
    assert last("COOLING_FAULT 1.0").cht_residual - last("NOMINAL").cht_residual > 20.0


def test_cooling_fault_heats_the_coolant() -> None:
    """COOLING_FAULT 1.0 (radiator blockage, Prompt 14): the coolant loop heats
    once the thermostat is fully open; the coolant residual rises."""
    end, nom = last("COOLING_FAULT 1.0"), last("NOMINAL")
    assert end.coolant_temp_c > nom.coolant_temp_c + 5.0
    assert end.coolant_residual_k > nom.coolant_residual_k + 5.0
    assert nom.coolant_status == "NORMAL"


def test_cooling_fault_does_not_trip_lhi() -> None:
    """Was a strict xfail (OI-14a: LHI 1.309 ALARM). Simulator oil pressure now
    follows oil viscosity: LHI 1.007."""
    assert last("COOLING_FAULT 1.0").lhi_band == "NORMAL"


def test_intake_boost_leak_map_and_air_down() -> None:
    """INTAKE_BOOST_LEAK 1.0: MAP 109.7 -> 74.7 kPa, air 0.0510 -> 0.0342 kg/s."""
    end, nom = last("INTAKE_BOOST_LEAK 1.0"), last("NOMINAL")
    assert end.map_pa < 0.8 * nom.map_pa
    assert end.air_kg_s < 0.8 * nom.air_kg_s


def test_intake_boost_leak_sfc_up() -> None:
    """Was a strict xfail (OI-14b: SFC flat at 0.237). Fuel now follows the
    post-leak air: SFC 0.2368 -> 0.2621 (+10.7 %). It rises despite OI-1 because
    friction is a larger share of the lower IMEP."""
    assert last("INTAKE_BOOST_LEAK 1.0").sfc > 1.05 * last("NOMINAL").sfc


# --------------------------------------------------------------------------
# Electrical faults (Prompt 12): simulator FaultMode, physical perturbations
# --------------------------------------------------------------------------

NON_ELECTRICAL = ("combustion", "csi", "lhi", "vhi", "egt", "lubrication", "sfc")


def _non_electrical_bad(name: str) -> list[str]:
    return [b for b in last(name).bad_bands() if b.split("=")[0] in NON_ELECTRICAL]


def test_nominal_electrical_normal() -> None:
    end = last("NOMINAL")
    assert (end.charging_status, end.ripple_status, end.ehi_band) == ("NORMAL", "NORMAL", "NORMAL")
    assert end.r_int_mohm is None and end.r_int_reason == "regulator holding bus; not observable"


def test_charging_setpoint_drift_flags_charging_only() -> None:
    end = last("CHARGING_FAULT setpoint_drift 1.0")
    assert end.charging_status in ("WARNING", "CRITICAL")
    assert end.charging_residual_median < -1.0
    assert _non_electrical_bad("CHARGING_FAULT setpoint_drift 1.0") == []


def test_open_diode_ripple_up_charging_in_band() -> None:
    end, nom = last("CHARGING_FAULT open_diode 1.0"), last("NOMINAL")
    assert end.ripple_pct >= 3.0 * nom.ripple_pct
    assert end.ripple_status in ("WARNING", "CRITICAL")
    assert end.charging_status == "NORMAL"
    assert _non_electrical_bad("CHARGING_FAULT open_diode 1.0") == []


def test_battery_degradation_unobservable_while_regulated() -> None:
    """At 4000 rpm the regulator holds the bus, so battery R_int growth is not
    observable: R_int stays invalid with its reason and no false alarm is raised.
    (R_int observability below cut-in is tested in test_electrical_health.py.)"""
    end = last("BATTERY_DEGRADATION 1.0")
    assert end.r_int_mohm is None and end.r_int_reason == "regulator holding bus; not observable"
    assert end.battery_status in ("INVALID", "UNKNOWN")
    assert end.charging_status == "NORMAL"
    assert _non_electrical_bad("BATTERY_DEGRADATION 1.0") == []


# --------------------------------------------------------------------------
# Fuel system, injectors and knock (Prompt 13), at cruise 4000 rpm / 110 kPa
# --------------------------------------------------------------------------

def test_nominal_injection_normal() -> None:
    end = last("NOMINAL")
    assert (end.injection_status, end.fuel_system_status, end.knock_status) == ("NORMAL", "NORMAL", "NORMAL")
    assert end.identified_injectors == [] and abs(end.fuel_delivery_ratio - 1.0) < 0.03


def test_injector_clog_at_cruise_moves_the_physics() -> None:
    """Cylinder 3 runs hot for its fuel (+32 K vs the others) and measured
    fuel falls below the command (delivery 0.962); no other cylinder is blamed."""
    end, nom = last("INJECTOR_FAULT clog cyl3 0.15"), last("NOMINAL")
    assert end.egt_differential_k[2] > 20.0
    assert end.fuel_delivery_ratio < nom.fuel_delivery_ratio - 0.02
    assert all(c == 3 for c in end.identified_injectors)
    assert "bound only" in end.injector_flow_reason[2]


@pytest.mark.xfail(strict=True, reason=(
    "OI-21: at cruise (operating λ 1.057 after the clog, L2 peak-EGT λ 1.111) EGT is near its peak, so "
    "cylinder 3's flow ratio is only bounded (<= 0.951, warning band 0.95) and the fuel delivery ratio "
    "0.962 is inside +/-0.05: a 15 % single-injector clog is not identified at cruise. It is at rated "
    "power (rich of peak): flow ratio 0.863."))
def test_injector_clog_identified_at_cruise() -> None:
    assert last("INJECTOR_FAULT clog cyl3 0.15").identified_injectors == [3]


def test_fuel_system_fault_rail_sag_no_injector_blamed() -> None:
    end = last("FUEL_SYSTEM_FAULT 1.0")
    assert end.rail_residual_kpa < -30.0
    assert end.fuel_system_status in ("WARNING", "CRITICAL")
    assert end.identified_injectors == []
    assert abs(end.fuel_delivery_ratio - 1.0) < 0.03  # meter agrees with the rail-corrected command


def test_knock_from_ignition_retard_not_vibration() -> None:
    end, nom = last("DETONATION_KNOCK cyl2 1.0"), last("NOMINAL")
    assert end.knock_suspected == [False, True, False, False]
    assert end.knock_status == "CRITICAL"
    assert end.vhi_band == nom.vhi_band and end.overall_rms == pytest.approx(nom.overall_rms, rel=0.05)
    assert end.identified_injectors == []


def test_confirmed_misfire_not_blamed_on_the_injector() -> None:
    end = last("MISFIRE 1.0")
    assert end.identified_injectors == []
    assert "SRD-FUN-044" in end.injector_flow_reason[0]


# --------------------------------------------------------------------------
# Sensor dropout, missing burst, CSV replay
# --------------------------------------------------------------------------

def test_egt_dropout_channel_invalid_and_in_coverage() -> None:
    end = last("SENSOR dropout EGT1")
    assert end.egt1_valid is False
    assert end.egt_coverage == pytest.approx(0.75)
    assert "egt_cyl_1" in end.invalid_channels and end.record_coverage < 1.0
    assert end.overall() != "NORMAL"


def test_map_dropout_channel_invalid_and_in_coverage() -> None:
    """Was a strict xfail (OI-12: MAP accepted as valid 0 Pa)."""
    end = last("SENSOR dropout MAP")
    assert end.map_valid is False
    assert {"map_pressure", "boost_pressure"} <= set(end.invalid_channels)
    assert end.cht_residual is None and end.air_kg_s is None  # no silent 101325 Pa substitution
    assert end.overall() != "NORMAL"


@pytest.mark.parametrize("name", ["NOMINAL, missing burst", "NOMINAL, CSV replay"])
def test_missing_burst_never_normal_with_full_coverage(name: str) -> None:
    end = last(name)
    assert end.verdict == "INVALID"
    assert end.combustion_status == "INVALID"
    assert end.vhi is None and end.vhi_band == "UNKNOWN"
    assert end.csi is None and end.csi_band == "UNKNOWN"
    assert end.overall() != "NORMAL"
    assert end.lhi_coverage < 1.0


# --------------------------------------------------------------------------
# API serialisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["NOMINAL", "MISFIRE 1.0", "NOMINAL, missing burst", "SENSOR dropout EGT1"])
def test_api_json_has_no_nan_and_invalid_is_null(name: str) -> None:
    for ep, (code, text) in api_responses(list(records(name))).items():
        assert code == 200, (ep, code)
        assert not json_has_nan(text), ep
        assert invalid_values_are_null(text) == [], (ep, invalid_values_are_null(text)[:3])


def test_normalized_record_json_mode_is_strict_json() -> None:
    """Was a strict xfail (OI-15). Invalid channel values serialise as null."""
    import json

    norm = convert_raw_to_engineering_state(records("NOMINAL")[-1])
    json.dumps(norm.model_dump(mode="json"), allow_nan=False)


# --------------------------------------------------------------------------
# Replay pipeline
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def replay(name: str):
    return tuple(run_replay_pipeline(list(records(name))))


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_replay_pipeline_accepts_and_carries_derived_state(name: str) -> None:
    steps = replay(name)
    assert all(s.accepted for s in steps)
    assert steps[-1].derived_state is not None


@pytest.mark.parametrize("name", ["MISFIRE 1.0", "OIL_DEGRADATION 1.0"])
def test_replay_health_index_drops_under_fault(name: str) -> None:
    """Health index (stateful pipeline, all L2 states reach L3 since OI-20):
    NOMINAL 0.899, MISFIRE 1.0 0.410, OIL_DEGRADATION 0.725 (was 0.905 / 0.825 / 0.789)."""
    assert replay(name)[-1].health_index < replay("NOMINAL")[-1].health_index


@pytest.mark.xfail(strict=True, reason=(
    "OI-16: the L3 fault classifier predicts NOMINAL (anomaly score 0.000) for every "
    "scenario, including MISFIRE 1.0, BEARING_WEAR and OIL_DEGRADATION."))
def test_replay_classifier_names_misfire() -> None:
    assert replay("MISFIRE 1.0")[-1].predicted_fault_class.name == "MISFIRE"


# ==========================================================================
# Full taxonomy (Prompt 18): all 14 classes and the 3 condition flags.
#
# Each class runs 120 s (1 Hz, onset 30 s, seed 42, the fault settings of
# tests/scientific/test_fault_taxonomy.py) through L1 (HMAC sign + validator)
# -> L2 -> L3 at the operating point where it is observable: cruise
# 4000 rpm / 110 kPa, except BATTERY_DEGRADATION at idle (1800 rpm / 60 kPa;
# R_int is hidden while the regulator holds the bus) and INJECTOR_FAULT at
# rated power (5800 rpm / 140 kPa; at cruise EGT is near its peak, OI-21).
# Evidence = mean of the ML feature over the last 10 records, fault vs NOMINAL
# at the same operating point and seed. The class is the RULE-BASED class at
# the last record (the demo default: ML off). Measured values are in
# docs/PHYSICS_AUDIT_RESULTS.md. No threshold was tuned for these tests.
# ==========================================================================

import math  # noqa: E402

import numpy as np  # noqa: E402

from src.core.provenance import CONDITION_FLAGS, FaultClass, FlightPhase  # noqa: E402
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner  # noqa: E402
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter  # noqa: E402
from tests.scientific.test_fault_taxonomy import FAULTS, OFF_DIAGONAL  # noqa: E402

CRUISE = {"rpm": 4000.0, "map_pa": 110000.0}
OBSERVABLE_AT = {FaultClass.BATTERY_DEGRADATION: {"rpm": 1800.0, "map_pa": 60000.0},
                 FaultClass.INJECTOR_FAULT: {"rpm": 5800.0, "map_pa": 140000.0}}

# class -> (evidence feature, direction, minimum change); the floor is a band
# edge where one exists, otherwise about half the measured change.
EVIDENCE: dict[FaultClass, tuple[str, int, float]] = {
    FaultClass.MISFIRE: ("comb_crank_cov", +1, 0.5),                  # 0.013 -> 1.007
    FaultClass.DETONATION_KNOCK: ("inj_ign_residual_cyl2", -1, 4.0),  # 0.0 -> -8.0 deg (ignition retard)
    FaultClass.EXHAUST_VALVE_LEAK: ("inj_egt_diff_cyl1", -1, 80.0),   # +12.8 -> -167.2 K (cold cylinder)
    FaultClass.INTAKE_BOOST_LEAK: ("twin_engine_load", -1, 0.10),     # 0.499 -> 0.305
    FaultClass.OIL_DEGRADATION: ("lub_lhi", -1, 0.5),                 # 1.002 -> 0.274
    FaultClass.COOLING_FAULT: ("res_coolant", +1, 8.0),               # -6.6 -> +3.6 K (+10.3 K; 8 K band)
    FaultClass.BEARING_WEAR: ("vib_envelope_rms", +1, 0.5),           # 0.072 -> 1.218
    FaultClass.INJECTOR_FAULT: ("res_injector_flow_ratio_cyl3", -1, 0.07),  # +0.013 -> -0.136 (rated)
    FaultClass.FUEL_SYSTEM_FAULT: ("res_fuel_rail_pressure", -1, 45000.0),  # -326 -> -90 436 Pa (rail sag)
    FaultClass.IMBALANCE: ("vib_prop1x_vel_frac_lat", +1, 0.5),       # 0.0002 -> 0.775 (dominance 0.5)
    FaultClass.CHARGING_FAULT: ("elec_charging_residual", -1, 0.7),   # -0.05 -> -1.50 V
    FaultClass.BATTERY_DEGRADATION: ("elec_r_int", +1, 7.0),          # 15.3 -> 30.1 mOhm (idle)
}


def _profile(fc: FaultClass) -> dict:
    return OBSERVABLE_AT.get(fc, CRUISE)


@lru_cache(maxsize=None)
def full_run(fc: FaultClass, profile_key: tuple):
    kw = FAULTS[fc]
    faults = None if kw is None else [FaultScenarioConfig(fault_class=fc, onset_time_s=30.0, duration_s=1e4, **kw)]
    recs, _, _ = ScenarioRunner(seed=42).run_scenario(duration_s=120.0, dt_s=1.0, operating_profile=[dict(profile_key)],
                                                       fault_scenarios=faults)
    return tuple(PipelineReplayAdapter().process_sequence(recs))


def taxonomy_run(fc: FaultClass):
    return full_run(fc, tuple(_profile(fc).items()))


def nominal_at(fc: FaultClass):
    return full_run(FaultClass.NOMINAL, tuple(_profile(fc).items()))


def feature_mean(steps, name: str) -> float | None:
    v = [s.ml_features.get(name) for s in steps[-10:]]
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else None


def _class_params():
    out = []
    for fc in FaultClass:
        if fc in OFF_DIAGONAL and fc not in OBSERVABLE_AT:
            got, why = OFF_DIAGONAL[fc]
            out.append(pytest.param(fc, marks=pytest.mark.xfail(strict=True, reason=f"measured {fc.name} -> "
                                                                f"{got.name}: {why}"), id=fc.name))
        else:
            out.append(pytest.param(fc, id=fc.name))
    return out


@pytest.mark.parametrize("fc", list(FaultClass), ids=[f.name for f in FaultClass])
def test_taxonomy_run_every_record_accepted_and_in_flight(fc: FaultClass) -> None:
    """L1: every record signed, verified and accepted; the derived flight
    phase is never GROUND in flight (OI-23)."""
    steps = taxonomy_run(fc)
    assert len(steps) == 120 and all(s.accepted for s in steps)
    assert all(s.mission_state.flight_phase != FlightPhase.GROUND for s in steps)


@pytest.mark.parametrize("fc", list(EVIDENCE), ids=[f.name for f in EVIDENCE])
def test_taxonomy_intended_evidence_moves(fc: FaultClass) -> None:
    name, sign, floor = EVIDENCE[fc]
    base, fault = feature_mean(nominal_at(fc), name), feature_mean(taxonomy_run(fc), name)
    assert base is not None and fault is not None, (name, base, fault)
    assert sign * (fault - base) >= floor, (name, base, fault)


def test_taxonomy_sensor_fault_evidence_is_the_invalid_channel() -> None:
    """SENSOR_FAULT (EGT1 dropout): the EGT1 residual becomes not derivable
    (NaN / valid=False), never a plausible constant."""
    assert feature_mean(nominal_at(FaultClass.SENSOR_FAULT), "res_egt_cyl1") is not None
    assert feature_mean(taxonomy_run(FaultClass.SENSOR_FAULT), "res_egt_cyl1") is None
    assert taxonomy_run(FaultClass.SENSOR_FAULT)[-1].residual_state.residuals["egt_cyl1"].valid is False


@pytest.mark.parametrize("fc", _class_params())
def test_taxonomy_intended_class_is_produced(fc: FaultClass) -> None:
    rule = taxonomy_run(fc)[-1].rule_fault_result
    assert rule.predicted_class == fc, (rule.class_name, rule.evidence)


# Condition flags: the run where each is intended to fire.
FLAG_RUNS = {"overheating_trend": FaultClass.COOLING_FAULT,
             "combustion_instability": FaultClass.DETONATION_KNOCK,
             "lubrication_degraded": FaultClass.OIL_DEGRADATION}


@pytest.mark.parametrize("flag", sorted(FLAG_RUNS))
def test_condition_flag_fires_on_its_run(flag: str) -> None:
    assert set(FLAG_RUNS) == set(CONDITION_FLAGS)
    assert taxonomy_run(FLAG_RUNS[flag])[-1].fault_result.condition_flags[flag] is True
    assert nominal_at(FLAG_RUNS[flag])[-1].fault_result.condition_flags[flag] is False


@pytest.mark.parametrize("profile", [CRUISE, *OBSERVABLE_AT.values()], ids=["cruise", "idle", "rated"])
def test_taxonomy_nominal_raises_nothing(profile: dict) -> None:
    """NOMINAL: from 30 s on, rule class NOMINAL, no condition flag, health
    NORMAL at every record, and no anomaly alarm (ML off: not evaluated)."""
    steps = full_run(FaultClass.NOMINAL, tuple(profile.items()))[30:]
    assert all(s.rule_fault_result.predicted_class == FaultClass.NOMINAL for s in steps)
    assert all(not any(v is True for v in s.fault_result.condition_flags.values()) for s in steps)
    assert all(s.health_state.health_status.value == "NORMAL" for s in steps)
    assert not any(s.anomaly_result.is_anomaly for s in steps)
