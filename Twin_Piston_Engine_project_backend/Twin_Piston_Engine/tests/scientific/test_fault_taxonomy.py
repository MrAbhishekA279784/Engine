"""
Prompt 15: unified 14-class fault taxonomy with condition flags.

Every simulator fault mode runs 120 s at the audit point (4000 rpm, 110 kPa,
1 Hz, onset 30 s, seed 42) through the real pipeline; the rule fallback's class
at the last record forms the confusion matrix below. Rules were not tuned to
force the diagonal: the off-diagonal cases are strict xfails with the measured
class and the physical reason (docs/FAULT_TAXONOMY.md).
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.core.provenance import CONDITION_FLAGS, FAULT_CLASS_COUNT, FaultClass
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine, RunningPipeline
from src.l3_ml.advisory_explainability import (
    CONDITION_FLAG_TEMPLATES,
    FAULT_ADVISORY_TEMPLATES,
    condition_flag_advisories,
    fault_class_advisory,
)
from src.l3_ml.anomaly_fault import FaultClassifier, NineClassFaultClassifier, evaluate_condition_flags

CRUISE = {"rpm": 4000.0, "map_pa": 110000.0}
FAULTS: dict[FaultClass, dict | None] = {
    FaultClass.NOMINAL: None,
    FaultClass.MISFIRE: dict(severity=1.0),
    FaultClass.DETONATION_KNOCK: dict(severity=1.0, affected_cylinder=2),
    FaultClass.EXHAUST_VALVE_LEAK: dict(severity=1.0),
    FaultClass.INTAKE_BOOST_LEAK: dict(severity=1.0),
    FaultClass.OIL_DEGRADATION: dict(severity=1.0),
    FaultClass.COOLING_FAULT: dict(severity=1.0),
    FaultClass.BEARING_WEAR: dict(severity=0.05),
    FaultClass.SENSOR_FAULT: dict(severity=1.0, affected_channel="egt_cyl1_hot_uv"),
    FaultClass.INJECTOR_FAULT: dict(severity=0.15, affected_cylinder=3, sub_mode="clog"),
    FaultClass.FUEL_SYSTEM_FAULT: dict(severity=1.0),
    FaultClass.IMBALANCE: dict(severity=1.0),
    FaultClass.CHARGING_FAULT: dict(severity=1.0, sub_mode="setpoint_drift"),
    FaultClass.BATTERY_DEGRADATION: dict(severity=1.0),
}

# Measured at the audit point (see module docstring). Off-diagonal entries and why.
OFF_DIAGONAL = {
    FaultClass.EXHAUST_VALVE_LEAK: (FaultClass.INJECTOR_FAULT,
        "a cold cylinder without a confirmed misfire reads as a rich (leaking) injector; EGT alone cannot "
        "separate a valve leak from an injector leak (EXHAUST_VALVE_LEAK is listed in also_consistent_with)"),
    FaultClass.INTAKE_BOOST_LEAK: (FaultClass.NOMINAL,
        "not evaluable: no boost reference (wastegate or throttle position) is instrumented, so a low MAP "
        "looks like a lower power setting"),
    FaultClass.COOLING_FAULT: (FaultClass.NOMINAL,
        "the thermostat absorbs the radiator blockage at cruise: coolant residual +7.5 K at steady state, "
        "below the 8 K band (OI-22); the overheating_trend flag does fire"),
    FaultClass.INJECTOR_FAULT: (FaultClass.NOMINAL,
        "at cruise EGT is near its peak, so the clog is only bounded (OI-21); identified at rated power"),
    FaultClass.BATTERY_DEGRADATION: (FaultClass.NOMINAL,
        "R_int is not observable while the regulator holds the bus (4000 rpm, above cut-in; Prompt 12); "
        "identified at idle"),
}


def _run(fault: FaultClass, profile: dict, duration_s: float = 120.0):
    kw = FAULTS[fault]
    faults = None if kw is None else [FaultScenarioConfig(fault_class=fault, onset_time_s=30.0, duration_s=1e4, **kw)]
    recs, _, _ = ScenarioRunner(seed=42).run_scenario(duration_s=duration_s, dt_s=1.0, operating_profile=[profile],
                                                       fault_scenarios=faults)
    return recs, PipelineReplayAdapter().process_sequence(recs)


@lru_cache(maxsize=None)
def _last(fault: FaultClass):
    return _run(fault, CRUISE)[1][-1]


def _params():
    out = []
    for fc in FAULTS:
        if fc in OFF_DIAGONAL:
            got, why = OFF_DIAGONAL[fc]
            out.append(pytest.param(fc, marks=pytest.mark.xfail(
                strict=True, reason=f"measured {fc.name} -> {got.name}: {why}"), id=fc.name))
        else:
            out.append(pytest.param(fc, id=fc.name))
    return out


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_taxonomy_ids_and_alias() -> None:
    assert FAULT_CLASS_COUNT == 14 and [m.value for m in FaultClass] == list(range(14))
    assert NineClassFaultClassifier is FaultClassifier
    assert set(CONDITION_FLAGS) == {"overheating_trend", "combustion_instability", "lubrication_degraded"}
    assert set(FAULT_ADVISORY_TEMPLATES) == {FaultClass(i) for i in range(9, 14)}
    assert set(CONDITION_FLAG_TEMPLATES) == set(CONDITION_FLAGS)


@pytest.mark.parametrize("n", [9, 14])
def test_ml_probability_vectors_of_legacy_and_unified_length_are_accepted(n: int) -> None:
    probs = [0.0] * n
    probs[n - 1] = 1.0
    prob_dict, conf, cls = FaultClassifier()._parse_probabilities(probs)
    assert prob_dict is not None and cls == FaultClass(n - 1) and conf == 1.0
    assert FaultClassifier()._parse_probabilities([0.1] * 10)[0] is None


# ---------------------------------------------------------------------------
# Rule fallback: each fault mode -> its own class (120 s, audit point)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault", _params())
def test_fault_mode_produces_its_own_class(fault: FaultClass) -> None:
    rule = _last(fault).rule_fault_result
    assert rule.is_ml is False and rule.confidence is None
    assert rule.predicted_class == fault, (rule.class_name, rule.evidence)


def test_confusion_matrix_is_the_documented_one() -> None:
    """Row = injected, predicted class at the last record. 9 of 14 on the
    diagonal; the documented off-diagonal cases are exactly OFF_DIAGONAL."""
    predicted = {fc: _last(fc).rule_fault_result.predicted_class for fc in FAULTS}
    expected = {fc: OFF_DIAGONAL.get(fc, (fc, ""))[0] for fc in FAULTS}
    assert predicted == expected
    assert sum(p == fc for fc, p in predicted.items()) == 9


def test_observable_conditions_identify_battery_and_injector() -> None:
    """The two observability cases are identified where they are observable."""
    idle = {"rpm": 1800.0, "map_pa": 60000.0}
    rated = {"rpm": 5800.0, "map_pa": 140000.0}
    assert _run(FaultClass.BATTERY_DEGRADATION, idle)[1][-1].rule_fault_result.predicted_class == \
        FaultClass.BATTERY_DEGRADATION
    assert _run(FaultClass.NOMINAL, idle)[1][-1].rule_fault_result.predicted_class == FaultClass.NOMINAL
    assert _run(FaultClass.INJECTOR_FAULT, rated)[1][-1].rule_fault_result.predicted_class == \
        FaultClass.INJECTOR_FAULT


def test_imbalance_is_1x_velocity_dominant_and_lateral() -> None:
    step = _last(FaultClass.IMBALANCE)
    ev = step.rule_fault_result.evidence
    assert ev["prop_1x_hz"] == pytest.approx(4000.0 / 60.0 / 2.54, rel=1e-3)  # from the measured crank period
    assert ev["prop_1x_velocity_fraction_lateral"] > 0.5 and ev["prop_1x_lateral_vertical_ratio"] > 1.0
    nominal = _last(FaultClass.NOMINAL)
    assert nominal.rule_fault_result.predicted_class == FaultClass.NOMINAL


# ---------------------------------------------------------------------------
# Condition flags: symptoms, independent of the class
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "OI-11: CSI stays NORMAL on the misfire run (0.0242 at 119 s, severity 1.0; NOMINAL 0.0142; band "
    "WARNING from 0.15): the crank term is CoV/100, so combustion_instability does not fire."))
def test_misfire_run_raises_combustion_instability() -> None:
    step = _last(FaultClass.MISFIRE)
    assert step.rule_fault_result.predicted_class == FaultClass.MISFIRE
    assert step.fault_result.condition_flags["combustion_instability"] is True


def test_flags_fire_independently_of_the_class() -> None:
    # A flag changes, the class does not: set the misfire run's CSI band to WARNING.
    mis = _last(FaultClass.MISFIRE)
    comb = mis.combustion_state.model_copy(update={"csi_band": "WARNING"})
    flags, _ = evaluate_condition_flags(mis.overheat_state, comb, None)
    assert flags["combustion_instability"] is True
    rule = FaultClassifier().classify_fault_rule_fallback(comb_state=comb, norm_record=None)
    assert rule.predicted_class == FaultClass.MISFIRE and rule.condition_flags["combustion_instability"] is True
    # Measured co-occurrences on the real runs:
    assert _last(FaultClass.COOLING_FAULT).fault_result.condition_flags["overheating_trend"] is True
    assert _last(FaultClass.COOLING_FAULT).rule_fault_result.predicted_class == FaultClass.NOMINAL
    oil = _last(FaultClass.OIL_DEGRADATION)
    assert oil.fault_result.condition_flags["lubrication_degraded"] is True
    assert oil.rule_fault_result.predicted_class == FaultClass.OIL_DEGRADATION
    knock = _last(FaultClass.DETONATION_KNOCK)
    assert knock.fault_result.condition_flags["combustion_instability"] is True
    assert knock.rule_fault_result.predicted_class == FaultClass.DETONATION_KNOCK
    nominal = _last(FaultClass.NOMINAL).fault_result.condition_flags
    assert nominal == {f: False for f in CONDITION_FLAGS}


def test_flag_not_evaluable_is_none_not_false() -> None:
    flags, ev = evaluate_condition_flags(None, None, None)
    assert flags == {f: None for f in CONDITION_FLAGS}
    assert all(isinstance(ev[f], str) for f in CONDITION_FLAGS)


# ---------------------------------------------------------------------------
# Advisory templates (US-801) and API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault", [FaultClass.FUEL_SYSTEM_FAULT, FaultClass.IMBALANCE, FaultClass.CHARGING_FAULT])
def test_class_advisory_gives_cause_evidence_and_action(fault: FaultClass) -> None:
    adv = fault_class_advisory(_last(fault).rule_fault_result)
    tpl = FAULT_ADVISORY_TEMPLATES[fault]
    assert adv is not None and "consistent with" in adv.message and tpl["action"] in adv.message
    for key in tpl["evidence"]:
        assert f"{key} = " in adv.message


def test_flag_advisories_for_set_flags_only() -> None:
    advs = condition_flag_advisories(_last(FaultClass.OIL_DEGRADATION).fault_result)
    titles = {a.title for a in advs}
    assert "Condition: lubrication degraded" in titles
    assert all("Action:" in a.message for a in advs)
    assert condition_flag_advisories(_last(FaultClass.NOMINAL).fault_result) == []


def test_fault_api_carries_flags_and_rule_diagnosis() -> None:
    recs, _ = _run(FaultClass.IMBALANCE, CRUISE)
    engine = ReplayEngine("tx")
    engine.load_scenario("tx", recs, [])
    pipeline = RunningPipeline(PipelineReplayAdapter())
    app = create_app()
    app.dependency_overrides[get_replay_engine] = lambda: engine
    app.dependency_overrides[get_running_pipeline] = lambda: pipeline
    with TestClient(app) as client:
        body = client.get("/api/v1/diagnostics/fault").json()
    assert body["status"] == "MODEL_UNAVAILABLE"            # the ML result stays honest
    assert set(body["condition_flags"]) == set(CONDITION_FLAGS)
    assert body["rule_based"]["predicted_class"] == "IMBALANCE" and body["rule_based"]["is_ml"] is False
    assert body["rule_based"]["class_id"] == FaultClass.IMBALANCE.value
