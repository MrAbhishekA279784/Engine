"""
Prompt 13: injection and ignition timing, injector and fuel-system diagnosis.

Simulator (injection.py): ECU pulse width from the trapped air and target λ,
SOI / advance maps, fuel rail with pump and regulator; each cylinder burns the
fuel its injector delivers, so EGT moves through the energy balance. Faults
are physical: injector flow coefficient (clog / leak), pump capacity, ECU knock
retard. L2 (injection_model.py) never reads the simulator's constants.

Operating points: rated power 5800 rpm / 140 kPa (rich of peak EGT: the
EGT-based flow ratio is defined there), cruise 4000 rpm / 110 kPa.
"""

from __future__ import annotations

from functools import lru_cache
from statistics import median

import pytest

from src.core.config import get_settings
from src.l1_data.simulator.fault_injection import FaultMode
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter
from src.l2_digital_twin.injection_model import InjectionModel, interval_us_to_deg
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, operating_point_from_record
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import evaluate_digital_twin
from tests.unit.test_vibration_burst import _base_raw

RATED = (5800.0, 140000.0)
CRUISE = (4000.0, 110000.0)
ONSET = 30.0
DURATION = 90.0


def _fault(mode: FaultMode, severity: float, cylinder: int | None = None, sub_mode: str | None = None):
    return (FaultScenarioConfig(fault_class=mode, severity=severity, affected_cylinder=cylinder,
                                sub_mode=sub_mode, onset_time_s=ONSET, duration_s=1e4),)


@lru_cache(maxsize=None)
def _run(point: tuple[float, float], fault: str | None = None):
    recs, gts, _ = ScenarioRunner(seed=7).run_scenario(
        duration_s=DURATION, dt_s=1.0, operating_profile=[{"rpm": point[0], "map_pa": point[1]}],
        fault_scenarios=list(FAULTS[fault]) if fault else None)
    return recs, gts, PipelineReplayAdapter().process_sequence(recs)


def _last(point, faults=None):
    return _run(point, faults)[2][-1]


CLOG3, LEAK3, PUMP, KNOCK2 = "clog3", "leak3", "pump", "knock2"
FAULTS = {
    CLOG3: _fault(FaultMode.INJECTOR_FAULT, 0.15, 3, "clog"),
    LEAK3: _fault(FaultMode.INJECTOR_FAULT, 0.15, 3, "leak"),
    PUMP: _fault(FaultMode.FUEL_SYSTEM_FAULT, 0.7),
    KNOCK2: _fault(FaultMode.DETONATION_KNOCK, 1.0, 2),
}


# ---------------------------------------------------------------------------
# Timing conversion
# ---------------------------------------------------------------------------

def test_interval_to_angle_1000_us_at_5000_rpm_is_30_deg() -> None:
    assert interval_us_to_deg(1000.0, 5000.0) == pytest.approx(30.0, abs=1e-12)


def test_sensor_inverse_converts_with_the_record_rpm() -> None:
    raw = _base_raw(crank_period_us=12000.0,  # 5000 rpm
                    ign_delay_us=(1000.0, 1000.0, 1000.0, 1000.0),
                    inj_soi_delay_us=(11000.0, 11000.0, 11000.0, 11000.0),
                    inj_pw_us=(5000.0, 5000.0, 5000.0, 5000.0))
    norm = convert_raw_to_engineering_state(raw)
    assert norm.ignition_timing_cyl_1.valid and norm.ignition_timing_cyl_1.value == pytest.approx(30.0)
    assert norm.injection_soi_deg_cyl_4.value == pytest.approx(330.0)
    assert norm.injection_timing_cyl_2.value == pytest.approx(0.011)      # interval kept, in s
    assert norm.injector_pulse_width_cyl_3.value == pytest.approx(0.005)


def test_not_instrumented_injection_is_invalid_everywhere() -> None:
    norm = convert_raw_to_engineering_state(_base_raw())
    for name in ("fuel_pressure", "ignition_timing_cyl_1", "injector_pulse_width_cyl_1", "injection_soi_deg_cyl_1"):
        ch = getattr(norm, name)
        assert ch.valid is False and ch.value is None and "not instrumented" in ch.fault_flag
    s = InjectionModel().evaluate(norm, evaluate_digital_twin(norm))
    for tv in (s.rail_dp_kpa, s.rail_pressure_residual_kpa, s.fuel_cmd_total_kg_s, s.fuel_delivery_ratio,
               s.ign_timing_residual_deg[0], s.injector_duty_pct[0], s.injector_flow_ratio[0]):
        assert tv.valid is False and tv.value is None and tv.fault_flag
    assert s.status.value == "INVALID"


# ---------------------------------------------------------------------------
# Healthy engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("point", [RATED, CRUISE], ids=["rated", "cruise"])
def test_healthy_flow_and_delivery_ratios_within_3_percent(point) -> None:
    s = _last(point).injection_state
    ratios = [tv.value for tv in s.injector_flow_ratio_median]
    assert all(tv.valid for tv in s.injector_flow_ratio_median)
    assert all(abs(r - 1.0) <= 0.03 for r in ratios), ratios
    assert s.fuel_delivery_ratio_median.valid and abs(s.fuel_delivery_ratio_median.value - 1.0) <= 0.03
    assert s.identified_injectors == []
    assert abs(s.rail_pressure_residual_median_kpa.value) < 5.0
    assert s.knock_status.value == "NORMAL" and s.status.value == "NORMAL"


def test_duty_cycle_at_rated_power_below_saturation() -> None:
    """Worst case: 5800 rpm at the simulator's maximum MAP (richest schedule)."""
    s = _last((5800.0, 160000.0)).injection_state
    duties = [tv.value for tv in s.injector_duty_pct]
    limit = get_settings().injection.duty_saturation_pct
    assert all(tv.valid for tv in s.injector_duty_pct)
    assert max(duties) < limit, f"duty {max(duties):.1f} % vs limit {limit} %"
    assert not any(s.injector_duty_saturated)


# ---------------------------------------------------------------------------
# INJECTOR_FAULT
# ---------------------------------------------------------------------------

def test_clogged_injector_cylinder_3_identified_hot_others_in_band() -> None:
    s = _last(RATED, CLOG3).injection_state
    assert s.identified_injectors == [3]
    assert s.egt_differential_k[2].value > 0.0          # hot for the fuel it was commanded
    assert s.injector_flow_ratio_median[2].value == pytest.approx(0.85, abs=0.05)
    for i in (0, 1, 3):
        assert s.injector_status[i].value == "NORMAL"
        assert abs(s.injector_flow_ratio_median[i].value - 1.0) < get_settings().injection.injector_flow_warning
    # engine-level consistency: measured flow falls below the command (one injector 15 % short)
    assert s.fuel_delivery_ratio_median.value == pytest.approx(1.0 - 0.15 / 4, abs=0.01)


ABS_EGT_REASON = (
    "OI-5: the absolute M-11 EGT residual is biased on a healthy engine (cyl1-4 -259.9 / -266.1 / "
    "-271.7 / -273.7 K at 5800 rpm / 140 kPa, seed 7), so cylinder 3's absolute residual rises by 96 K "
    "under a 15 % clog but stays negative (measured cyl1-4 {}). The diagnosis uses the residual "
    "relative to the other cylinders.")


@lru_cache(maxsize=None)
def _abs_egt(point, faults):
    last = _last(point, faults)
    norm = convert_raw_to_engineering_state(last.record)
    exp = HealthyExpectationModel().expected_egt_per_cylinder_k(operating_point_from_record(norm))
    egts = [norm.egt_cyl_1.value, norm.egt_cyl_2.value, norm.egt_cyl_3.value, norm.egt_cyl_4.value]
    return [round(e - x, 1) for e, x in zip(egts, exp)]


@pytest.mark.xfail(strict=True, reason=ABS_EGT_REASON.format("-259.9 / -266.1 / -175.5 / -273.7 K"))
def test_clogged_injector_absolute_egt_residual_positive() -> None:
    r = _abs_egt(RATED, CLOG3)
    assert r[2] > 0.0, r


def test_leaking_injector_cold_but_not_a_misfire() -> None:
    last = _last(RATED, LEAK3)
    s, comb = last.injection_state, last.combustion_state
    assert s.egt_differential_k[2].value < 0.0
    assert s.identified_injectors == [3]
    assert s.injector_flow_ratio_median[2].value == pytest.approx(1.15, abs=0.05)
    # the dual-channel gate (M-06) does not confirm, so no misfire is declared
    assert comb.misfire_verdict != "CONFIRMED"
    assert comb.misfire_detected == [False, False, False, False]


def test_confirmed_misfire_makes_the_cold_cylinder_non_firing_not_rich() -> None:
    """SRD-FUN-044: a cold cylinder is read as rich only if the misfire gate does
    not confirm a misfire on it; confirmed -> flow ratio invalid (non-firing)."""
    recs, _, steps = _run(RATED, LEAK3)
    model = InjectionModel()
    for rec, step in zip(recs, steps):
        comb = step.combustion_state.model_copy(update={"misfire_detected": [False, False, True, False]})
        norm = convert_raw_to_engineering_state(rec)
        s = model.evaluate(norm, evaluate_digital_twin(norm), comb)
    assert s.injector_flow_ratio[2].valid is False and s.injector_flow_ratio[2].value is None
    assert "SRD-FUN-044" in s.injector_flow_ratio[2].fault_flag
    assert 3 not in s.identified_injectors


# ---------------------------------------------------------------------------
# FUEL_SYSTEM_FAULT
# ---------------------------------------------------------------------------

def test_fuel_system_fault_rail_sags_and_all_cylinders_lean_together() -> None:
    nom_recs, nom_gts, nom_steps = _run(RATED)
    recs, gts, steps = _run(RATED, PUMP)
    s, s_nom = steps[-1].injection_state, nom_steps[-1].injection_state
    warn_kpa = get_settings().injection.rail_residual_warning_pa / 1000.0
    assert s.rail_pressure_residual_median_kpa.value < -warn_kpa
    assert s.fuel_system_status.value in ("WARNING", "CRITICAL")
    # every cylinder is commanded-and-delivered less fuel: rail mixture factor < 1 for all
    assert s.rail_mixture_factor.value < 0.97
    for i in range(4):
        assert s.fuel_cmd_per_cycle_mg[i].value < 0.97 * s_nom.fuel_cmd_per_cycle_mg[i].value
    assert any("all cylinders lean together" in e for e in s.evidence)
    # ...and all four run hotter together (lean of the rich schedule)
    norm, norm_nom = convert_raw_to_engineering_state(recs[-1]), convert_raw_to_engineering_state(nom_recs[-1])
    for ch in ("egt_cyl_1", "egt_cyl_2", "egt_cyl_3", "egt_cyl_4"):
        assert getattr(norm, ch).value > getattr(norm_nom, ch).value + 10.0
    # no single injector is blamed and the meter agrees with the (rail-corrected) command
    assert s.identified_injectors == []
    assert abs(s.fuel_delivery_ratio_median.value - 1.0) < 0.03


# ---------------------------------------------------------------------------
# Ignition timing and knock
# ---------------------------------------------------------------------------

def test_knock_is_ignition_retard_plus_egt_pattern() -> None:
    s = _last(CRUISE, KNOCK2).injection_state
    retard = get_settings().simulator.injection.knock_max_retard_deg
    assert s.ign_timing_residual_median_deg[1].value == pytest.approx(-retard, abs=0.1)
    for i in (0, 2, 3):
        assert abs(s.ign_timing_residual_median_deg[i].value) < 0.1
    assert s.knock_suspected == [False, True, False, False]
    assert s.knock_status.value == "CRITICAL"          # retard + that cylinder runs hot
    assert s.egt_differential_k[1].value > get_settings().injection.knock_egt_support_k
    # the retarded cylinder's EGT is not read as an injector problem
    assert s.injector_flow_ratio[1].valid is False and "ignition retarded" in s.injector_flow_ratio[1].fault_flag
    assert s.identified_injectors == []


def test_knock_leaves_no_trace_in_the_accelerometer_band() -> None:
    """Knock energy is above the 1024 Hz Nyquist limit of the 2048 Hz burst:
    the simulator adds none, and nothing in L2 reads knock from vibration."""
    nom, knock = _run(CRUISE)[1][-1], _run(CRUISE, KNOCK2)[1][-1]
    assert knock.vibration_rms_m_s2 == nom.vibration_rms_m_s2
    assert knock.vibration_exc_g == nom.vibration_exc_g


# ---------------------------------------------------------------------------
# Residuals and the expectation guard
# ---------------------------------------------------------------------------

def test_residual_state_carries_fuel_system_residuals() -> None:
    recs, _, _ = _run(RATED, CLOG3)
    from src.l2_digital_twin.residual_engine import ResidualEngine

    model, engine = InjectionModel(), ResidualEngine()
    for rec in recs:
        norm = convert_raw_to_engineering_state(rec)
        derived = evaluate_digital_twin(norm)
        state, exp = engine.evaluate(norm, derived, injection_state=model.evaluate(norm, derived))
    for key in ("fuel_rail_pressure", "fuel_delivery_ratio", *(f"injector_flow_ratio_cyl{i}" for i in range(1, 5))):
        assert key in state.residuals and key in exp.expected_values
    assert state.residuals["injector_flow_ratio_cyl3"].value == pytest.approx(-0.15, abs=0.05)
    assert state.residuals["fuel_delivery_ratio"].value < -0.02


def test_rail_expectation_does_not_read_the_rail_pressure() -> None:
    norm = convert_raw_to_engineering_state(_run(RATED)[0][-1])
    changed = norm.model_copy(update={"fuel_pressure": norm.fuel_pressure.model_copy(update={"value": 150000.0})})
    a, b = InjectionModel().evaluate(norm), InjectionModel().evaluate(changed)
    assert a.expected_rail_dp_kpa.value == b.expected_rail_dp_kpa.value
    assert b.rail_pressure_residual_kpa.value < a.rail_pressure_residual_kpa.value - 100.0


def test_cylinder_reference_excludes_the_cylinder_it_judges() -> None:
    """Changing cylinder 3's EGT moves its differential residual by exactly the
    change: its reference (median of the other cylinders) does not read it."""
    norm = convert_raw_to_engineering_state(_run(RATED)[0][-1])
    changed = norm.model_copy(update={"egt_cyl_3": norm.egt_cyl_3.model_copy(
        update={"value": norm.egt_cyl_3.value + 40.0})})
    a, b = InjectionModel().evaluate(norm), InjectionModel().evaluate(changed)
    assert b.egt_differential_k[2].value - a.egt_differential_k[2].value == pytest.approx(40.0)


def test_simulated_record_carries_raw_timing_and_counts_only() -> None:
    r = _run(RATED)[0][-1]
    assert len(r.inj_pw_us) == 4 and len(r.inj_soi_delay_us) == 4 and len(r.ign_delay_us) == 4
    assert isinstance(r.fuel_press_counts, int) and 0 < r.fuel_press_counts < 4095
    # intervals in microseconds, consistent with the ECU angle at this rpm
    assert interval_us_to_deg(r.ign_delay_us[0], 60e6 / r.crank_period_us) == pytest.approx(
        _run(RATED)[2][-1].injection_state.scheduled_advance_deg.value, abs=0.2)
