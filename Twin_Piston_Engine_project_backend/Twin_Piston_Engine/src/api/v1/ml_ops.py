"""
ML operations (Prompt 17): model versions and metrics, residual drift, and
rollback of the model or a per-engine baseline.

POST /ml/rollback changes state, so it requires a bearer token: the token is
read from the environment variable named by security.api_bearer_token_env_var.
If that variable is unset the endpoint refuses (fail closed).
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_app_settings, get_replay_engine, get_running_pipeline
from src.core.config import AppSettings
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, RunningPipeline
from src.l3_ml.adaptation import AdaptationLog, BaselineStore, ModelRegistry

router = APIRouter(prefix="/ml", tags=["ML Operations"])


def require_bearer_token(authorization: str | None = Header(default=None),
                         settings: AppSettings = Depends(get_app_settings)) -> None:
    expected = os.environ.get(settings.security.api_bearer_token_env_var)
    if not expected:
        raise HTTPException(status_code=503, detail="bearer token not configured; state-changing ML endpoints are disabled")
    if not authorization or not authorization.startswith("Bearer ") or \
            not hmac.compare_digest(authorization[7:].encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


def _log(settings: AppSettings) -> AdaptationLog:
    return AdaptationLog(Path(settings.adaptation.store_dir) / "adaptation_log.jsonl")


@router.get("/models", summary="Model versions, metrics and active flags")
def get_models(settings: AppSettings = Depends(get_app_settings)) -> dict[str, Any]:
    root = Path(settings.model_dir)
    reg = ModelRegistry(root).load()
    versions = {}
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            card = d / "model_card.json"
            versions[d.name] = {
                "files": sorted(f.name for f in d.iterdir() if f.suffix in (".joblib", ".json")),
                "model_card": json.loads(card.read_text()) if card.exists() else None,
                "registry": reg["versions"].get(d.name),
            }
    active = settings.ml.model_version
    if active == "registry":
        active = reg["history"][-1] if reg["history"] else None
    return {"configured_model_version": settings.ml.model_version, "active_model_version": active,
            "models_loaded": bool(active), "rul_model_version": settings.ml.rul_model_version,
            "registry_history": reg["history"], "versions": versions,
            "note": "all metrics measured on SIMULATED data"}


@router.get("/drift", summary="Residual drift (PSI) per channel for the replayed stream")
def get_drift(replay_engine: ReplayEngine = Depends(get_replay_engine),
              pipeline: RunningPipeline = Depends(get_running_pipeline)) -> dict[str, Any]:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    latest = pipeline.advance(records, source_key=(id(replay_engine), replay_engine.scenario_id))
    d = latest.drift_state
    hs = latest.health_state
    return {"drift": d.model_dump(mode="json") if d is not None else None,
            "baseline_version": hs.baseline_version, "adapted_baseline": hs.adapted_baseline,
            "records_in_window": pipeline.records_processed}


class RollbackRequest(BaseModel):
    target: Literal["model", "baseline"]
    engine_id: str | None = None


@router.post("/rollback", summary="Restore the previous model or per-engine baseline version",
             dependencies=[Depends(require_bearer_token)])
def post_rollback(req: RollbackRequest, settings: AppSettings = Depends(get_app_settings)) -> dict[str, Any]:
    log = _log(settings)
    if req.target == "model":
        reg = ModelRegistry(settings.model_dir)
        before = reg.active()
        after = reg.rollback()
        log.append("model", "rollback", [], from_version=before, to_version=after)
        return {"target": "model", "from_version": before, "to_version": after}
    if not req.engine_id:
        raise HTTPException(status_code=422, detail="engine_id is required for a baseline rollback")
    store = BaselineStore(settings.adaptation.store_dir)
    before = store.active(req.engine_id).version
    after = store.rollback(req.engine_id)
    log.append("baseline", "rollback", [], engine_id=req.engine_id, from_version=before, to_version=after.version)
    return {"target": "baseline", "engine_id": req.engine_id, "from_version": before, "to_version": after.version}
