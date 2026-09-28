"""
Fault & Anomaly Router — Module 13 ML Inference Endpoints (Module 20).

Exposes the AnomalyDetector and FaultClassifier outputs of the running
pipeline (history kept across calls), with their model status: when a model
is unavailable its outputs are None and the status says so (no substitutes).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.api.v1.schemas import AnomalyResponse, FaultClassificationResponse
from src.core.provenance import InferenceStatus, Provenance
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, RunningPipeline

router = APIRouter(prefix="/diagnostics", tags=["Faults & Anomalies"])


def _latest(replay_engine: ReplayEngine, pipeline: RunningPipeline):
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    return pipeline.advance(records, source_key=(id(replay_engine), replay_engine.scenario_id))


@router.get(
    "/anomaly",
    response_model=AnomalyResponse,
    summary="Get Anomaly Detection Status & Score",
    description="Exposes Module 13 AnomalyDetector output.",
)
def get_anomaly_status(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    pipeline: RunningPipeline = Depends(get_running_pipeline),
) -> AnomalyResponse:
    latest = _latest(replay_engine, pipeline)
    a = latest.anomaly_result
    ok = a.status == InferenceStatus.SUCCESS
    return AnomalyResponse(
        is_anomaly=a.is_anomaly if ok else None,
        anomaly_score=a.anomaly_score if ok else None,
        threshold=a.threshold if ok else None,
        status=("ANOMALY_DETECTED" if a.is_anomaly else "NORMAL") if ok else a.status.value,
        evidence=dict(a.evidence),
        quality=a.quality,
        provenance=Provenance.MODEL_OUTPUT,
        timestamp=latest.timestamp,
        records_in_window=pipeline.records_processed,
    )


@router.get(
    "/fault",
    response_model=FaultClassificationResponse,
    summary="Get Fault Classifier Prediction and Condition Flags",
    description="Exposes the Module 13 FaultClassifier prediction over the unified 14-class taxonomy, the condition flags, and the rule-based diagnosis.",
)
def get_fault_classification(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    pipeline: RunningPipeline = Depends(get_running_pipeline),
) -> FaultClassificationResponse:
    latest = _latest(replay_engine, pipeline)
    f = latest.fault_result
    r = latest.rule_fault_result
    rule = None if r is None else {
        "predicted_class": r.class_name, "class_id": r.class_id, "is_ml": r.is_ml,
        "provenance": r.provenance.value, "evidence": r.evidence or {},
    }
    return FaultClassificationResponse(
        predicted_class=f.class_name,
        class_id=f.class_id,
        confidence=f.confidence,
        probabilities=f.probabilities,
        status=f.status.value,
        quality=f.quality,
        provenance=Provenance.MODEL_OUTPUT,
        timestamp=latest.timestamp,
        records_in_window=pipeline.records_processed,
        baseline_version=f.baseline_version,
        adapted_baseline=f.adapted_baseline,
        model_version=f.model_version,
        hours_since_baseline=f.hours_since_baseline,
        condition_flags=dict(f.condition_flags),
        condition_evidence=dict(f.condition_evidence),
        rule_based=rule,
    )
