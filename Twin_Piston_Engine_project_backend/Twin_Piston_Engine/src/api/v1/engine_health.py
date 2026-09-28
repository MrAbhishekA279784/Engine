"""
Engine Health Router — Module 14 Health Index & Degradation Supervision Endpoints (Module 20).

Exposes supervisory HealthIndex, degradation state, trend, and component scores.
Calls Module 14 HealthSupervisionEngine; NEVER recalculates Health Index inside API route.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    get_replay_engine,
    get_running_pipeline,
)
from src.api.v1.schemas import EngineHealthResponse
from src.core.provenance import Provenance
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, RunningPipeline
from src.l3_ml.advisory_explainability import electrical_advisories, overheat_advisories

ELECTRICAL_FIELDS = (
    "bus_voltage_v", "alternator_current_a", "battery_current_a", "charging_residual_v",
    "battery_resistance_mohm", "voltage_ripple_pct", "ripple_dominant_frequency_hz",
)


def electrical_summary(elec) -> dict:
    """Flatten an ElectricalState for API / WebSocket payloads: valid values
    as floats, invalid ones as None with their reason."""
    if elec is None:
        return {"electrical_status": "INVALID",
                "electrical_invalid_reasons": {"electrical": "no electrical state"}}
    out: dict = {}
    reasons: dict[str, str] = {}
    for name in ELECTRICAL_FIELDS:
        tv = getattr(elec, "charging_residual_median_v" if name == "charging_residual_v" else name)
        out[name] = tv.value if tv.valid else None
        if not tv.valid:
            reasons[name] = tv.fault_flag or "not derivable"
    out["electrical_health_index"] = elec.ehi.value if elec.ehi.valid else None
    out["electrical_health_band"] = elec.ehi_band
    out["electrical_status"] = elec.status.value
    out["electrical_invalid_reasons"] = reasons
    out["electrical_advisories"] = [a.message for a in electrical_advisories(elec)]
    return out

def cooling_summary(coolant, overheat) -> dict:
    """Coolant state and overheating trends for API / WebSocket payloads.
    Invalid values are None with their reason; nothing is a constant."""
    out: dict = {}
    reasons: dict[str, str] = {}
    if coolant is None:
        reasons["coolant"] = "no coolant state"
    else:
        for api_name, attr in (("coolant_temp_c", "coolant_temp_c"),
                               ("coolant_residual_k", "coolant_residual_median_k"),
                               ("coolant_cht_delta_k", "coolant_cht_delta_k")):
            tv = getattr(coolant, attr)
            out[api_name] = tv.value if tv.valid else None
            if not tv.valid:
                reasons[api_name] = tv.fault_flag or "not derivable"
        out["coolant_status"] = coolant.status.value
    out["coolant_invalid_reasons"] = reasons
    trends: dict[str, dict] = {}
    if overheat is not None:
        for name, t in overheat.channels.items():
            ttl = t.time_to_limit_s
            trends[name] = {
                "current_c": t.current_c.value if t.current_c.valid else None,
                "limit_c": t.limit_c,
                "slope_residual_k_min": t.slope_residual_k_min.value if t.slope_residual_k_min.valid else None,
                "slope_residual_ci_k_min": list(t.slope_residual_ci_k_min) if t.slope_residual_ci_k_min else None,
                "time_to_limit_s": ttl.value if ttl.valid else None,
                "time_to_limit_range_s": list(t.time_to_limit_range_s) if (ttl.valid and t.time_to_limit_range_s)
                else None,
                "reason": None if ttl.valid else (ttl.fault_flag or "no trend"),
                "status": t.status.value,
            }
        out["overheat_status"] = overheat.status.value
        out["overheat_advisories"] = [a.message for a in overheat_advisories(overheat)]
    out["overheat_trends"] = trends
    return out


HEALTH_COMPONENTS = ("thermal_health", "lubrication_health", "vibration_health", "combustion_health",
                     "performance_health", "anomaly_fault_health", "electrical_health")


def health_summary(health) -> dict:
    """Flatten an L3 HealthState: nothing is a constant. Components without
    valid data are None with their reason."""
    if health is None:
        return {"health_index": None, "health_index_reason": "no health state", "degradation_state": "UNKNOWN",
                "trend": "UNKNOWN", "health_rate_per_s": None,
                "component_health": {c: None for c in HEALTH_COMPONENTS},
                "component_health_reasons": {c: "no health state" for c in HEALTH_COMPONENTS},
                "coverage": None, "status": "INVALID", "quality": 0.0}
    hi = health.health_index
    valid = bool(hi.valid and hi.value is not None)
    components: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    for c in HEALTH_COMPONENTS:
        if c in health.component_health:
            components[c] = health.component_health[c]
        else:
            components[c] = None
            reasons[c] = health.component_unavailable.get(c, "no valid data")
    return {
        "health_index": hi.value if valid else None,
        "health_index_reason": None if valid else (
            f"insufficient evidence quality {health.quality:.2f}"),
        "degradation_state": health.degradation_state.value if valid else "UNKNOWN",
        "trend": health.trend if valid else "UNKNOWN",
        "health_rate_per_s": health.health_rate if valid else None,
        "baseline_version": health.baseline_version, "adapted_baseline": health.adapted_baseline,
        "model_version": health.model_version, "hours_since_baseline": health.hours_since_baseline,
        "component_health": components,
        "component_health_reasons": reasons,
        "coverage": health.evidence.get("coverage", {}).get("data_coverage"),
        "status": health.health_status.value,
        "quality": health.quality,
    }


router = APIRouter(prefix="/engine", tags=["Engine Health"])


@router.get(
    "/health",
    response_model=EngineHealthResponse,
    summary="Get Engine Health Index & Degradation State",
    description="Exposes Module 14 HealthSupervisionEngine aggregate engine health state, degradation classification, and trend.",
)
def get_engine_health(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    pipeline: RunningPipeline = Depends(get_running_pipeline),
) -> EngineHealthResponse:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry records available to compute engine health.")

    # Running pipeline: only new records are stepped; windows, trend and R_int
    # history persist across calls exactly as in the pipeline (OI-20).
    latest = pipeline.advance(records, source_key=(id(replay_engine), replay_engine.scenario_id))
    derived = latest.derived_state

    def _value(tv):  # valid tagged value -> float, otherwise None
        return tv.value if (tv is not None and tv.valid) else None

    return EngineHealthResponse(
        **health_summary(latest.health_state),
        records_in_window=pipeline.records_processed,
        provenance=Provenance.DERIVED,
        timestamp=latest.timestamp,
        sfc_kg_kwh=_value(derived.sfc_kg_kwh) if derived else None,
        sfc_band=derived.sfc_band if derived else "UNKNOWN",
        sfc_degradation_pct=_value(derived.sfc_degradation_pct) if derived else None,
        engine_load=_value(derived.engine_load) if derived else None,
        load_band=derived.load_band if derived else "UNKNOWN",
        pressure_altitude_m=_value(derived.pressure_altitude_m) if derived else None,
        isa_deviation_k=_value(derived.isa_deviation_k) if derived else None,
        density_altitude_m=_value(derived.density_altitude_m) if derived else None,
        **electrical_summary(latest.electrical_state),
        **cooling_summary(latest.coolant_state, latest.overheat_state),
    )
