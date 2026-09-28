"""
Telemetry Router — Raw Telemetry Ingestion and Query Endpoints (Module 20).

Enforces L1 Telemetry Security & Validation without bypassing Module 3.
"""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import (
    get_packet_signer,
    get_telemetry_repository,
    get_telemetry_validator,
)
from src.api.v1.schemas import (
    PaginatedResponse,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
)
from src.l1_data.raw_repository import RawTelemetryRepositoryProtocol
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.post(
    "",
    response_model=TelemetryIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Raw Telemetry Packet",
    description="Validates and persists incoming raw acquisition packet. Rejects dropped or tampered packets via Module 3 security rules.",
)
def ingest_telemetry(
    request: TelemetryIngestRequest,
    validator: TelemetryValidator = Depends(get_telemetry_validator),
    signer: PacketSigner = Depends(get_packet_signer),
    repository: RawTelemetryRepositoryProtocol = Depends(get_telemetry_repository),
) -> TelemetryIngestResponse:
    ts = request.timestamp or datetime.now(timezone.utc)

    record = RawSignalRecord(
        timestamp=ts,
        sequence_number=request.sequence_number,
        source_type=request.source_type,
        egt_cyl1_hot_uv=request.egt_cyl1_hot_uv,
        egt_cyl2_hot_uv=request.egt_cyl2_hot_uv,
        egt_cyl3_hot_uv=request.egt_cyl3_hot_uv,
        egt_cyl4_hot_uv=request.egt_cyl4_hot_uv,
        egt_cold_c=request.egt_cold_c,
        cht_hot_uv=request.cht_hot_uv,
        cht_cold_c=request.cht_cold_c,
        oil_rtd_ohms=request.oil_rtd_ohms,
        oil_p_counts=request.oil_p_counts,
        map_counts=request.map_counts,
        adc_vref_counts=request.adc_vref_counts,
        crank_period_us=request.crank_period_us,
        fuel_pulse_hz=request.fuel_pulse_hz,
        accel_counts_xyz=request.accel_counts_xyz,
        ambient_temp_c=request.ambient_temp_c,
        ambient_press_pa=request.ambient_press_pa,
        accel_burst_counts_x=tuple(request.accel_burst_counts_x),
        accel_burst_counts_y=tuple(request.accel_burst_counts_y),
        accel_burst_counts_z=tuple(request.accel_burst_counts_z),
        accel_burst_fs_hz=request.accel_burst_fs_hz,
        crank_period_burst_us=tuple(request.crank_period_burst_us),
        bus_v_counts=request.bus_v_counts,
        alt_i_counts=request.alt_i_counts,
        batt_i_counts=request.batt_i_counts,
        bus_v_burst_counts=tuple(request.bus_v_burst_counts) if request.bus_v_burst_counts is not None else None,
        bus_v_burst_fs_hz=request.bus_v_burst_fs_hz,
        inj_pw_us=tuple(request.inj_pw_us) if request.inj_pw_us is not None else None,
        inj_soi_delay_us=tuple(request.inj_soi_delay_us) if request.inj_soi_delay_us is not None else None,
        ign_delay_us=tuple(request.ign_delay_us) if request.ign_delay_us is not None else None,
        fuel_press_counts=request.fuel_press_counts,
        coolant_ntc_ohms=request.coolant_ntc_ohms,
    )

    # Use signature from request or sign using signer
    sig = request.signature or signer.sign_record(record)
    val_result = validator.validate_packet(record, signature=sig)

    if val_result.accepted and val_result.record:
        repository.save(val_result.record)

    return TelemetryIngestResponse(
        accepted=val_result.accepted,
        sequence_number=request.sequence_number,
        rejection_reason=val_result.rejection_reason,
        security_verified=val_result.security_verified,
        invalid_channels=val_result.invalid_channels,
        quality=val_result.record.signal_quality if val_result.record else SignalQuality(valid=False, score=0.0),
        timestamp=ts,
    )


@router.get(
    "/latest",
    response_model=dict[str, Any],
    summary="Get Latest Raw Telemetry Record",
    description="Returns the most recently ingested canonical RawSignalRecord.",
)
def get_latest_telemetry(
    repository: RawTelemetryRepositoryProtocol = Depends(get_telemetry_repository),
) -> dict[str, Any]:
    latest = repository.get_latest()
    if not latest:
        raise HTTPException(
            status_code=status.HTTP_444_NO_RESPONSE if hasattr(status, "HTTP_444_NO_RESPONSE") else 404,
            detail="No telemetry records available in repository.",
        )
    return latest.model_dump()


@router.get(
    "/history",
    response_model=PaginatedResponse[dict[str, Any]],
    summary="Get Historical Telemetry Records",
    description="Returns paginated raw telemetry records sorted chronologically.",
)
def get_telemetry_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    repository: RawTelemetryRepositoryProtocol = Depends(get_telemetry_repository),
) -> PaginatedResponse[dict[str, Any]]:
    offset = (page - 1) * page_size
    records = repository.get_history(limit=page_size, offset=offset)
    total = repository.count()

    total_pages = max(1, (total + page_size - 1) // page_size)
    items = [r.model_dump() for r in records]

    return PaginatedResponse(
        items=items,
        total_items=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
