"""
RUL Router — Module 15 Remaining Useful Life Endpoints (Module 20).

Exposes the RULEstimator output of the running pipeline (history kept across
calls): hours remaining, bounds, trend and status as L3 computed them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.api.v1.schemas import RULResponse
from src.core.provenance import InferenceStatus, Provenance
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, RunningPipeline

router = APIRouter(prefix="/engine", tags=["Remaining Useful Life"])


@router.get(
    "/rul",
    response_model=RULResponse,
    summary="Get Remaining Useful Life Estimate",
    description="Exposes Module 15 RULEstimator hours remaining and confidence bounds.",
)
def get_rul_estimate(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    pipeline: RunningPipeline = Depends(get_running_pipeline),
) -> RULResponse:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    latest = pipeline.advance(records, source_key=(id(replay_engine), replay_engine.scenario_id))
    rul = latest.rul_state
    ok = rul.status == InferenceStatus.SUCCESS
    return RULResponse(
        hours_remaining=rul.hours_remaining if ok else None,
        lower_bound_hours=rul.lower_bound_hours if ok else None,
        upper_bound_hours=rul.upper_bound_hours if ok else None,
        unit=rul.unit,
        trend=rul.trend,
        operating_assumption=rul.operating_assumption,
        # an unvalidated estimate is never presented as a plain SUCCESS
        status=rul.status.value if (rul.validated or not ok) else "UNVALIDATED",
        validated=bool(rul.validated),
        validation_note=rul.validation_note or None,
        inference_status=rul.status.value,
        interval_calibrated=bool(rul.interval_calibrated),
        interval_note=rul.interval_note or None,
        quality=rul.quality,
        provenance=Provenance.MODEL_OUTPUT,
        timestamp=latest.timestamp,
        records_in_window=pipeline.records_processed,
    )
