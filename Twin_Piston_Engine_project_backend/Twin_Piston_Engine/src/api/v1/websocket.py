"""
WebSocket Real-Time Router — /api/v1/ws/engine Stream Endpoint (Module 20 & 23).

Provides structured real-time diagnostic event streaming.
DISCONNECT-SAFE, EXCEPTION-ISOLATED, READ-ONLY DIAGNOSTIC STREAM.
RESILIENT: Bounded client capacity, slow-client isolation, non-blocking broadcast timeout.
DOES NOT LEAK RAW SIMULATOR GROUND TRUTH OR PRIVATE INTERNALS.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.dependencies import (
    get_advisory_engine,
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.engine_health import cooling_summary, electrical_summary
from src.api.v1.schemas import EventEnvelope
from src.core.logging import get_logger
from src.core.provenance import Provenance
from src.l1_data.telemetry_validator import TelemetryValidator
from src.performance_resilience.tracker import tracker_instance

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["Real-Time Streaming"])


class ConnectionManager:
    """Resilient read-only WebSocket connection manager with slow-client isolation."""

    def __init__(self, max_clients: int = 100, send_timeout_s: float = 2.0) -> None:
        self.active_connections: list[WebSocket] = []
        self._max_clients = max_clients
        self._send_timeout_s = send_timeout_s
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        async with self._lock:
            if len(self.active_connections) >= self._max_clients:
                logger.warning(f"WebSocket client connection rejected: Max capacity ({self._max_clients}) reached.")
                await websocket.close(code=1013, reason="Server busy / Max connections reached")
                return False

            await websocket.accept()
            self.active_connections.append(websocket)
            tracker_instance.set_active_ws_clients(len(self.active_connections))
            logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")
            return True

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            tracker_instance.set_active_ws_clients(len(self.active_connections))
            logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def send_envelope(self, websocket: WebSocket, envelope: EventEnvelope) -> None:
        """Send envelope to a single WebSocket client with timeout isolation."""
        try:
            payload_json = envelope.model_dump(mode="json")
            await asyncio.wait_for(
                websocket.send_json(payload_json),
                timeout=self._send_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timeout (slow client). Disconnecting client.")
            tracker_instance.increment_dropped_events()
            self.disconnect(websocket)
        except Exception as e:
            logger.warning(f"Failed to send envelope to WebSocket client: {e}")
            tracker_instance.increment_dropped_events()
            self.disconnect(websocket)


manager = ConnectionManager()


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    """Returns when the client disconnects (clients send nothing on this stream)."""
    try:
        while True:
            await websocket.receive_text()
    except Exception:  # WebSocketDisconnect or a closed transport
        return


@router.websocket("/engine")
async def websocket_engine_endpoint(websocket: WebSocket) -> None:
    """Real-time engine diagnostic event streaming endpoint."""
    connected = await manager.connect(websocket)
    if not connected:
        return

    replay_engine = get_replay_engine()
    adapter = get_pipeline_adapter()

    try:
        # Emit initial system status event envelope
        init_envelope = EventEnvelope(
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            sequence_number=0,
            payload={"status": "CONNECTED", "stream": "realtime_engine_telemetry"},
            quality=1.0,
            provenance=Provenance.DERIVED,
        )
        await manager.send_envelope(websocket, init_envelope)

        records = replay_engine.get_all_records()
        if not records:
            close_envelope = EventEnvelope(
                event_type="system_status",
                timestamp=datetime.now(timezone.utc),
                sequence_number=0,
                payload={"status": "NO_TELEMETRY_AVAILABLE"},
                quality=0.0,
                provenance=Provenance.DERIVED,
            )
            await manager.send_envelope(websocket, close_envelope)
            await websocket.close()
            return

        # Every record is streamed (Prompt 18, OI-23; was records[:5]), each
        # through one stateful pipeline run with its own validator, processed
        # as it is sent.
        run = adapter.new_run(validator=TelemetryValidator(settings=adapter._settings))
        # A client that closes is only seen on receive: watch for it
        # concurrently so the stream stops instead of running to the end.
        client_gone = asyncio.create_task(_wait_for_disconnect(websocket))
        sent = 0
        for rec in records:
            if websocket not in manager.active_connections or client_gone.done():
                break
            step = run.step(rec)
            sent += 1

            # 1. Telemetry event envelope
            t_env = EventEnvelope(
                event_type="telemetry_update",
                timestamp=step.timestamp,
                sequence_number=step.sequence_number,
                payload={
                    "rpm": step.derived_rpm,
                    "power_kw": step.derived_power_kw,
                },
                quality=1.0 if step.accepted else 0.5,
                provenance=Provenance.SIMULATED,
            )
            await manager.send_envelope(websocket, t_env)

            # 2. Health event envelope
            h_env = EventEnvelope(
                event_type="health_update",
                timestamp=step.timestamp,
                sequence_number=step.sequence_number,
                payload={
                    "health_index": step.health_index,
                    "is_anomaly": step.is_anomaly,
                    "anomaly_score": step.anomaly_score,
                    "predicted_fault_class": step.predicted_fault_class.name if hasattr(step.predicted_fault_class, "name") else str(step.predicted_fault_class),
                    "rul_hours": step.rul_hours,
                    "rul_validated": bool(step.rul_state is not None and step.rul_state.validated),
                    "mission_risk_score": step.mission_risk_score,
                    "electrical": electrical_summary(step.electrical_state),
                    "cooling": cooling_summary(step.coolant_state, step.overheat_state),
                },
                quality=1.0 if step.accepted else 0.5,
                provenance=Provenance.DERIVED,
            )
            await manager.send_envelope(websocket, h_env)

            await asyncio.sleep(0.05)

        stopped_early = client_gone.done()
        client_gone.cancel()
        if websocket in manager.active_connections and not stopped_early:
            await manager.send_envelope(websocket, EventEnvelope(
                event_type="system_status",
                timestamp=datetime.now(timezone.utc),
                sequence_number=sent,
                payload={"status": "STREAM_COMPLETE", "records_streamed": sent, "records_available": len(records)},
                quality=1.0,
                provenance=Provenance.DERIVED,
            ))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect event caught cleanly.")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket handler exception: {e}")
        manager.disconnect(websocket)
    finally:
        # the stream ended (complete or client gone): release the slot (Prompt 18;
        # before, a normally completed stream stayed in active_connections)
        manager.disconnect(websocket)
