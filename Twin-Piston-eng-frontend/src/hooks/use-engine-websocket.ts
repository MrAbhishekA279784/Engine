import { useEffect, useRef, useState, useCallback } from "react";
import type {
  AdvisoryResponse,
  EngineHealthResponse,
  EventEnvelope,
  FaultClassificationResponse,
  MissionRiskResponse,
  RULResponse,
} from "@/types/backend-api";

const WS_URL = (import.meta.env.VITE_WS_URL as string) || "ws://localhost:8000/api/v1/ws/engine";

export interface StreamState {
  isConnected: boolean;
  connectionState: "connected" | "connecting" | "disconnected" | "fallback_mock";
  lastTimestamp: string | null;
  sequenceNumber: number;
  quality: number;
  liveTelemetry: Record<string, any> | null;
  liveHealth: EngineHealthResponse | null;
  liveFault: FaultClassificationResponse | null;
  liveRul: RULResponse | null;
  liveMission: MissionRiskResponse | null;
  liveAdvisories: AdvisoryResponse[];
}

export function useEngineWebSocket() {
  const [streamState, setStreamState] = useState<StreamState>({
    isConnected: false,
    connectionState: "connecting",
    lastTimestamp: null,
    sequenceNumber: 0,
    quality: 1.0,
    liveTelemetry: null,
    liveHealth: null,
    liveFault: null,
    liveRul: null,
    liveMission: null,
    liveAdvisories: [],
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    try {
      setStreamState((prev) => ({ ...prev, connectionState: "connecting" }));
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setStreamState((prev) => ({
          ...prev,
          isConnected: true,
          connectionState: "connected",
        }));
      };

      ws.onmessage = (event) => {
        try {
          const envelope: EventEnvelope = JSON.parse(event.data);
          const { event_type, timestamp, sequence_number, quality, payload } = envelope;

          setStreamState((prev) => {
            const next = {
              ...prev,
              lastTimestamp: timestamp,
              sequenceNumber: sequence_number,
              quality: quality ?? 1.0,
            };

            switch (event_type) {
              case "telemetry_update":
                next.liveTelemetry = payload;
                break;
              case "health_update":
                next.liveHealth = payload as EngineHealthResponse;
                break;
              case "fault_update":
                next.liveFault = payload as FaultClassificationResponse;
                break;
              case "advisory_update":
                if (Array.isArray(payload.advisories)) {
                  next.liveAdvisories = payload.advisories;
                }
                break;
              default:
                break;
            }

            return next;
          });
        } catch {
          // Ignore invalid WS payload
        }
      };

      ws.onerror = () => {
        setStreamState((prev) => ({
          ...prev,
          isConnected: false,
          connectionState: "fallback_mock",
        }));
      };

      ws.onclose = () => {
        setStreamState((prev) => ({
          ...prev,
          isConnected: false,
          connectionState: "fallback_mock",
        }));
        // Attempt reconnect after 5s
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, 5000);
      };
    } catch {
      setStreamState((prev) => ({
        ...prev,
        isConnected: false,
        connectionState: "fallback_mock",
      }));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return streamState;
}
