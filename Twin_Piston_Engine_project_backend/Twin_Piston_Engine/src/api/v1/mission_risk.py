"""
Mission Risk Router — Module 16 Flight Phase & Analytical Risk Endpoints (Module 20).

Exposes the MissionRiskEngine output of the running pipeline (history kept
across calls): flight phase, risk score, level and trend as L3 computed them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.api.v1.schemas import MissionRiskResponse
from src.core.provenance import Provenance
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, RunningPipeline

router = APIRouter(prefix="/mission", tags=["Mission Risk"])


@router.get(
    "/risk",
    response_model=MissionRiskResponse,
    summary="Get Mission Phase & Risk Assessment",
    description="Exposes Module 16 MissionRiskEngine phase context and relative analytical risk score.",
)
def get_mission_risk(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    pipeline: RunningPipeline = Depends(get_running_pipeline),
) -> MissionRiskResponse:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    latest = pipeline.advance(records, source_key=(id(replay_engine), replay_engine.scenario_id))
    m = latest.mission_state
    phase = m.flight_phase.value if hasattr(m.flight_phase, "value") else str(m.flight_phase)
    return MissionRiskResponse(
        flight_phase=phase,
        risk_score=m.risk_score,
        risk_level=m.risk_level,
        risk_trend=m.risk_trend,
        health_index=m.health_index,
        rul_hours=m.rul_hours,
        contributing_fault=m.contributing_fault.name if m.contributing_fault is not None else None,
        quality=m.quality,
        provenance=Provenance.DERIVED,
        timestamp=latest.timestamp,
        records_in_window=pipeline.records_processed,
        rul_validated=bool(latest.rul_state is not None and latest.rul_state.validated),
    )
