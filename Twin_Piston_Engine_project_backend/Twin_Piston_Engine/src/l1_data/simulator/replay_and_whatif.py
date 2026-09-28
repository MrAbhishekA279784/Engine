"""
Simulation Replay and What-if Engine — Original Module 18.

Provides:
- Deterministic simulation replay with pause/resume, seek, restart, and state tracking.
- Controlled What-if scenario cloning and parameter modification with strict validation.
- Pipeline replay adapter executing replayed RawSignalRecord sequences through L1 -> L2 -> L3.
- Deterministic scenario comparison (baseline vs. what-if deltas in Health Index, RUL, Anomaly Score, Fault Classification, Mission Risk).
- Post-inference ground-truth accuracy validation (Ground truth isolated from L2/L3 inference).

STRICT RULES:
- Ground truth physical state remains strictly isolated during pipeline inference.
- What-if cloning NEVER mutates the baseline scenario.
- RawSignalRecord remains canonical.
- Invalid what-if parameters trigger explicit validation errors (ValueError).
- 100% deterministic outputs given identical scenario, modification, and seed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import FlightPhase, FaultClass, InferenceStatus, Provenance
from src.core.schemas import DerivedEngineState, SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    ScenarioRunner,
    SensorForwardModel,
    SimulationGroundTruth,
    SimulationMetadata,
)

logger = get_logger(__name__)


class ReplayStatus(str, Enum):
    """Execution status of the scenario replay engine."""

    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


@dataclass
class ReplayState:
    """Current state of scenario replay engine."""

    scenario_id: str
    replay_position: int
    total_records: int
    status: ReplayStatus
    current_timestamp: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    sequence_number: int = 0
    speed_multiplier: float = 1.0
    provenance: Provenance = Provenance.SIMULATED


@dataclass
class WhatIfScenario:
    """Representation of a cloned and modified what-if scenario definition."""

    what_if_id: str
    scenario_id: str
    parent_scenario_id: str
    modifications: dict[str, Any]
    resulting_configuration: dict[str, Any]
    seed: int
    provenance: Provenance = Provenance.SIMULATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MetricDelta:
    """Comparison metric delta between baseline and what-if scenarios."""

    metric_name: str
    baseline_value: float | None
    what_if_value: float | None
    delta: float | None
    pct_change: float | None = None


@dataclass
class ScenarioComparison:
    """Structured result of baseline vs what-if scenario evaluation comparison."""

    baseline_id: str
    what_if_id: str
    metric_deltas: dict[str, MetricDelta]
    state_changes: dict[str, Any]
    timestamp_alignment_info: dict[str, Any]
    ground_truth_validation: dict[str, Any] | None = None
    quality: SignalQuality = field(default_factory=lambda: SignalQuality(valid=True, score=1.0))
    provenance: Provenance = Provenance.SIMULATED


class ReplayEngine:
    """Deterministic replay engine for simulated telemetry sequences."""

    def __init__(self, scenario_id: str = "default_scenario") -> None:
        self._scenario_id = scenario_id
        self._records: list[RawSignalRecord] = []
        self._ground_truths: list[SimulationGroundTruth] = []
        self._position: int = 0
        self._status: ReplayStatus = ReplayStatus.STOPPED
        self._speed_multiplier: float = 1.0

    def load_scenario(
        self,
        scenario_id: str,
        records: list[RawSignalRecord],
        ground_truths: list[SimulationGroundTruth] | None = None,
    ) -> None:
        """Load a sequence of RawSignalRecords for replay."""
        if not records:
            raise ValueError("Cannot load empty telemetry record sequence for replay.")

        self._scenario_id = scenario_id
        self._records = list(records)
        self._ground_truths = list(ground_truths) if ground_truths else []
        self._position = 0
        self._status = ReplayStatus.STOPPED

    @property
    def is_loaded(self) -> bool:
        return len(self._records) > 0

    def play(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("No scenario loaded to play.")
        if self._position >= len(self._records):
            self._position = 0
        self._status = ReplayStatus.PLAYING

    def pause(self) -> None:
        if self._status == ReplayStatus.PLAYING:
            self._status = ReplayStatus.PAUSED

    def resume(self) -> None:
        if self._status == ReplayStatus.PAUSED:
            self._status = ReplayStatus.PLAYING

    def stop(self) -> None:
        self._status = ReplayStatus.STOPPED
        self._position = 0

    def restart(self) -> None:
        self._position = 0
        self._status = ReplayStatus.PLAYING

    def seek(self, position: int) -> RawSignalRecord | None:
        """Seek to a specific sample index in the scenario."""
        if not self.is_loaded:
            raise RuntimeError("No scenario loaded to seek.")
        if position < 0 or position >= len(self._records):
            raise IndexError(f"Seek position {position} out of bounds [0, {len(self._records) - 1}]")

        self._position = position
        return self._records[self._position]

    def seek_to_timestamp(self, target_time: datetime) -> RawSignalRecord | None:
        """Seek to the nearest sample corresponding to target_time."""
        if not self.is_loaded:
            raise RuntimeError("No scenario loaded to seek.")

        best_idx = 0
        min_diff = float("inf")
        for idx, rec in enumerate(self._records):
            diff = abs((rec.timestamp - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_idx = idx

        self._position = best_idx
        return self._records[self._position]

    def step(self) -> RawSignalRecord | None:
        """Advance replay by one timestep and return the RawSignalRecord."""
        if not self.is_loaded or self._status in (ReplayStatus.STOPPED, ReplayStatus.PAUSED):
            return None

        if self._position >= len(self._records):
            self._status = ReplayStatus.COMPLETED
            return None

        record = self._records[self._position]
        self._position += 1

        if self._position >= len(self._records):
            self._status = ReplayStatus.COMPLETED

        return record

    def get_state(self) -> ReplayState:
        """Get current ReplayState status snapshot."""
        current_ts = self._records[self._position].timestamp if self.is_loaded and self._position < len(self._records) else None
        start_ts = self._records[0].timestamp if self.is_loaded else None
        end_ts = self._records[-1].timestamp if self.is_loaded else None
        seq_num = self._records[self._position].sequence_number if self.is_loaded and self._position < len(self._records) else 0

        return ReplayState(
            scenario_id=self._scenario_id,
            replay_position=self._position,
            total_records=len(self._records),
            status=self._status,
            current_timestamp=current_ts,
            start_time=start_ts,
            end_time=end_ts,
            sequence_number=seq_num,
            speed_multiplier=self._speed_multiplier,
            provenance=Provenance.SIMULATED,
        )

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    def get_all_records(self) -> list[RawSignalRecord]:
        return list(self._records)

    def get_all_ground_truths(self) -> list[SimulationGroundTruth]:
        """Exposed ONLY for metadata identification and post-inference comparison."""
        return list(self._ground_truths)


class WhatIfEngine:
    """Engine for controlled scenario cloning, parameter modification, and execution."""

    ALLOWED_MODIFICATION_KEYS: set[str] = {
        "rpm_profile",
        "throttle_profile",
        "operating_profile",
        "fault_scenarios",
        "seed",
        "ambient_temp_k",
        "ambient_pressure_pa",
        "isa_deviation_k",
    }

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._runner = ScenarioRunner(settings=self._settings)

    def validate_modifications(self, modifications: dict[str, Any]) -> None:
        """Validate proposed what-if parameter modifications strictly.

        Raises ValueError on invalid keys, types, bounds, or out-of-range parameters.
        """
        if not isinstance(modifications, dict):
            raise ValueError(f"Modifications must be a dictionary, got {type(modifications)}")

        for key, val in modifications.items():
            if key not in self.ALLOWED_MODIFICATION_KEYS:
                raise ValueError(f"Unsupported what-if parameter '{key}'. Allowed keys: {sorted(self.ALLOWED_MODIFICATION_KEYS)}")

            if key in ("rpm_profile", "throttle_profile") and val is not None:
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"'{key}' must be a list of numbers")
                for item in val:
                    if not isinstance(item, (int, float)):
                        raise ValueError(f"Invalid item in '{key}': {item}")

            if key == "fault_scenarios" and val is not None:
                if not isinstance(val, list):
                    raise ValueError("'fault_scenarios' must be a list of FaultScenarioConfig objects")
                for f_cfg in val:
                    if not isinstance(f_cfg, FaultScenarioConfig):
                        raise ValueError(f"Item in fault_scenarios is not FaultScenarioConfig: {type(f_cfg)}")
                    if not (0.0 <= f_cfg.severity <= 1.0):
                        raise ValueError(f"Fault severity must be in [0.0, 1.0], got {f_cfg.severity}")

            if key == "seed" and val is not None:
                if not isinstance(val, int):
                    raise ValueError(f"Seed must be an integer, got {type(val)}")

            if key in ("ambient_temp_k", "ambient_pressure_pa") and val is not None:
                if not isinstance(val, (int, float)) or val <= 0:
                    raise ValueError(f"'{key}' must be a positive number, got {val}")

            if key == "isa_deviation_k" and val is not None:
                if not isinstance(val, (int, float)) or not -60.0 <= val <= 60.0:
                    raise ValueError(f"'isa_deviation_k' must be a number in [-60, 60] K, got {val}")

    def create_what_if_scenario(
        self,
        baseline_scenario_id: str,
        baseline_config: dict[str, Any],
        modifications: dict[str, Any],
    ) -> WhatIfScenario:
        """Clone a baseline scenario and apply validated modifications without mutating baseline."""
        self.validate_modifications(modifications)

        # Deepcopy baseline config so parent is NEVER mutated
        new_config = copy.deepcopy(baseline_config)
        new_config.update(modifications)

        # Generate deterministic what_if_id
        mod_json = json.dumps(modifications, sort_keys=True, default=str)
        hash_str = hashlib.sha256(mod_json.encode("utf-8")).hexdigest()[:8]
        what_if_id = f"whatif_{baseline_scenario_id}_{hash_str}"

        active_seed = modifications.get("seed", baseline_config.get("seed", 42))

        return WhatIfScenario(
            what_if_id=what_if_id,
            scenario_id=what_if_id,
            parent_scenario_id=baseline_scenario_id,
            modifications=modifications,
            resulting_configuration=new_config,
            seed=active_seed,
            provenance=Provenance.SIMULATED,
        )

    def execute_what_if(
        self,
        what_if_scenario: WhatIfScenario,
        duration_s: float = 60.0,
        dt_s: float = 0.1,
    ) -> tuple[list[RawSignalRecord], list[SimulationGroundTruth], SimulationMetadata]:
        """Execute what-if scenario deterministically using ScenarioRunner."""
        cfg = what_if_scenario.resulting_configuration

        # Explicit ambient overrides apply to every profile segment (they were
        # accepted but not applied before Prompt 14).
        profile = cfg.get("operating_profile")
        overrides = {k: cfg[k] for k in ("ambient_temp_k", "ambient_pressure_pa") if cfg.get(k) is not None}
        if overrides:
            profile = [dict(seg, **overrides) for seg in (profile or [{}])]

        records, ground_truths, metadata = self._runner.run_scenario(
            duration_s=duration_s,
            dt_s=dt_s,
            rpm_profile=cfg.get("rpm_profile"),
            throttle_profile=cfg.get("throttle_profile"),
            operating_profile=profile,
            fault_scenarios=cfg.get("fault_scenarios"),
            seed=what_if_scenario.seed,
            isa_deviation_k=float(cfg.get("isa_deviation_k") or 0.0),
        )

        return records, ground_truths, metadata


@dataclass
class PipelineStepResult:
    """Result of running one RawSignalRecord through the complete L1 -> L2 -> L3 pipeline."""

    timestamp: datetime
    sequence_number: int
    record: RawSignalRecord
    accepted: bool
    derived_rpm: float = 0.0
    derived_power_kw: float = 0.0
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    predicted_fault_class: FaultClass = FaultClass.NOMINAL
    health_index: float = 100.0
    rul_hours: float = 1000.0
    mission_risk_score: float = 0.0
    derived_state: DerivedEngineState | None = None  # L2 twin output for this step
    electrical_state: Any = None  # L2 ElectricalState for this step (Prompt 12)
    injection_state: Any = None   # L2 InjectionState for this step (Prompt 13)
    combustion_state: Any = None  # L2 CombustionStabilityState (misfire gate, CSI)
    coolant_state: Any = None     # L2 CoolantState (Prompt 14)
    overheat_state: Any = None    # L2 OverheatTrendState (Prompt 14)
    health_state: Any = None      # L3 HealthState for this step (component health, trend, coverage)
    rul_state: Any = None         # L3 RULState
    mission_state: Any = None     # L3 MissionState (risk)
    anomaly_result: Any = None    # L3 AnomalyResult
    fault_result: Any = None      # L3 FaultClassificationResult (ML; carries the condition flags)
    rule_fault_result: Any = None  # L3 rule-based FaultClassificationResult (is_ml=False)
    # L2 states and the ML feature row (schema 2.0.0) for dataset generation / explanation
    residual_state: Any = None
    vibration_state: Any = None
    lubrication_state: Any = None
    egt_result: Any = None
    ml_features: dict | None = None
    expectation_result: Any = None  # HealthyExpectationResult (fleet and per-engine expectations)
    drift_state: Any = None         # DriftState (Prompt 17)


class PipelineRun:
    """One stateful L1 -> L2 -> L3 chain over a record stream.

    Every L2 model and L3 engine is built once, so their windows (CSI, LHI,
    misfire gate, electrical residual median and R_int steps, health trend)
    accumulate across step() calls exactly as in a live pipeline.
    """

    def __init__(self, settings: AppSettings, validator: TelemetryValidator, signer: Any,
                 baseline: Any = None) -> None:
        # Dynamic import of L2 and L3 components to ensure zero static AST import violations
        l2 = importlib.import_module("src.l2_digital_twin")
        l3 = importlib.import_module("src.l3_ml")
        self._settings = settings
        self._baseline = baseline  # per-engine EngineBaseline (Prompt 17) or None = fleet-0
        self._validator = validator
        self._signer = signer
        # one stateful sensor-inverse model per run (engine hours accumulate across records)
        self._sensor_inverse = getattr(l2, "SensorInverseModel")(settings)
        self._twin = getattr(l2, "ThermodynamicMechanicalTwin")(settings)
        self._egt = getattr(l2, "EGTDiagnosticsEngine")(settings)
        self._lub = getattr(l2, "LubricationModel")(settings)
        self._vib = getattr(l2, "VibrationProcessor")(settings)
        self._misfire = getattr(l2, "MisfireDetector")(settings)
        self._residual = getattr(l2, "ResidualEngine")(settings)
        if baseline is not None and baseline.corrections:
            self._residual.set_baseline(baseline.corrections_tuple(), baseline.version)
        drift_mod = importlib.import_module("src.l3_ml.drift_monitor")
        self._adapt_mod = importlib.import_module("src.l3_ml.adaptation")
        ref = drift_mod.DriftReference.from_dict(baseline.reference) if (baseline is not None and baseline.reference) \
            else None
        self._drift = drift_mod.DriftMonitor(settings, ref)
        self._model_version = settings.ml.model_version or "rules-only"
        self._electrical = getattr(l2, "ElectricalModel")(settings=settings)
        self._injection = getattr(l2, "InjectionModel")(settings)
        self._coolant = getattr(l2, "CoolantModel")(settings)
        self._overheat = getattr(l2, "OverheatTrendModel")(settings)

        self._builder = getattr(l3, "MLFeatureVectorBuilder")(settings=settings)
        self._ml_features = importlib.import_module("src.l3_ml.ml_features")
        self._anomaly = getattr(l3, "AnomalyDetector")(settings=settings)
        self._classifier = getattr(l3, "FaultClassifier")(settings=settings)
        self._condition_flags = getattr(l3, "evaluate_condition_flags")
        self._health = getattr(l3, "HealthSupervisionEngine")(settings=settings)
        self._rul = getattr(l3, "RULEstimator")(settings=settings)
        self._lifetime_rul = None
        if settings.ml.rul_model_version:
            from pathlib import Path as _Path
            path = _Path(settings.model_dir) / settings.ml.rul_model_version / "rul_lifetime.joblib"
            if path.exists():
                import joblib as _joblib
                lifetime = importlib.import_module("src.l3_ml.lifetime_rul")
                self._lifetime_rul = lifetime.LifetimeRULEstimator(_joblib.load(path))
        self._risk = getattr(l3, "MissionRiskEngine")(settings=settings)
        # Flight phase (Prompt 18, OI-23): derived per record from rpm, MAP,
        # throttle, pressure altitude and its vertical rate; stateful debounce.
        self._phase = getattr(l3, "MissionPhaseClassifier")(settings=settings)
        self._alt_hist: list[tuple[float, float]] = []
        self._last_phase: Any = None

    def _vertical_rate(self, ts: Any, alt: float | None) -> float | None:
        """Least-squares slope [m/s] of pressure altitude over the configured
        window; None when altitude is invalid or the window is too short."""
        cfg = self._settings.mission
        if alt is None or ts is None:
            self._alt_hist.clear()
            return None
        t = ts.timestamp()
        self._alt_hist.append((t, float(alt)))
        self._alt_hist = [(a, b) for a, b in self._alt_hist if t - a <= cfg.vertical_rate_window_s]
        if len(self._alt_hist) < cfg.vertical_rate_min_points:
            return None
        ts_, hs = zip(*self._alt_hist)
        mt, mh = sum(ts_) / len(ts_), sum(hs) / len(hs)
        sxx = sum((a - mt) ** 2 for a in ts_)
        return sum((a - mt) * (b - mh) for a, b in zip(ts_, hs)) / sxx if sxx > 0.0 else None

    def _flight_phase(self, norm_rec: Any, twin_res: Any) -> tuple[Any, float, str | None]:
        def v(ch):
            return ch.value if (ch is not None and ch.valid and ch.value is not None) else None
        pa = twin_res.pressure_altitude_m
        alt = pa.value if (pa.valid and pa.value is not None) else None
        return self._phase.classify_phase(
            rpm=v(norm_rec.rpm), map_pa=v(norm_rec.map_pressure), throttle_pct=v(norm_rec.throttle_position),
            altitude_m=alt, vertical_rate_m_s=self._vertical_rate(norm_rec.timestamp, alt),
            timestamp=norm_rec.timestamp)

    def _lifetime_rul_state(self, engine_hours: float, ml_row: dict, baseline: Any) -> Any:
        """Adopted lifetime RUL (validated on held-out simulated engines,
        reports/rul_metrics.md) replaces the unvalidated baseline number."""
        r = self._lifetime_rul.observe(float(engine_hours), ml_row)
        est = r["status"] == "ESTIMATED"
        note = ("Lifetime RUL model (validated on held-out SIMULATED engines; reports/rul_metrics.md). "
                if est else f"Lifetime RUL status {r['status']} after {r['windows']} monitoring windows: no number. ")
        return baseline.model_copy(update={
            "hours_remaining": r["rul_h"] if est else 0.0,
            "lower_bound_hours": r["rul_lo_h"] if est else None,
            "upper_bound_hours": r["rul_hi_h"] if est else None,
            "uncertainty_available": est,
            "status": InferenceStatus.SUCCESS if est else InferenceStatus.INSUFFICIENT_HISTORY,
            "model_name": "lifetime_rul", "is_ml": True, "validated": est,
            "validation_note": note + "Interval from validation-engine error quantiles.",
            "interval_calibrated": bool(est and r["rul_h"] < 150.0),
            "interval_note": ("90 % interval; measured coverage 0.87-0.88 on held-out engines below 150 h"
                              if est and r["rul_h"] < 150.0 else
                              "interval NOT calibrated: measured 90 % coverage 0.52 for RUL beyond 150 h "
                              "(reports/rul_metrics.md)" if est else ""),
            "evidence": {"lifetime_status": r["status"], "severity_estimate": r["severity_estimate"],
                         "monitoring_windows": r["windows"]},
        })

    def step(self, rec: RawSignalRecord) -> PipelineStepResult:
        # 1. L1 Validation with HMAC signing
        sig = self._signer.sign_record(rec)
        accepted = self._validator.validate_packet(rec, signature=sig).accepted

        # 2. L2 Digital Twin Processing (stateful models)
        norm_rec = self._sensor_inverse.convert_raw_to_engineering(rec)  # this run's settings and state
        twin_res = self._twin.evaluate(norm_rec)
        _, egt_res = self._egt.evaluate(norm_rec)
        lub_state, _ = self._lub.evaluate(norm_rec, twin_res)
        vib_state, vib_res = self._vib.process_record(norm_rec)
        comb_state, _ = self._misfire.evaluate(norm_rec, twin_res, egt_res, vib_res)
        inj_state = self._injection.evaluate(norm_rec, twin_res, comb_state)
        res_state, exp_res = self._residual.evaluate(norm_rec, twin_res, injection_state=inj_state)
        elec_state = self._electrical.evaluate(norm_rec)
        coolant_state = self._coolant.evaluate(norm_rec)
        overheat_state = self._overheat.evaluate(norm_rec)

        # 3. L3 Feature Construction & Inference: ML feature schema 2.0.0, built
        #    only from the L2/L3 states above (NaN where not derivable).
        ml_row = self._ml_features.extract_features(
            residual_state=res_state, derived_state=twin_res, vib_state=vib_state, comb_state=comb_state,
            lub_state=lub_state, elec_state=elec_state, injection_state=inj_state, coolant_state=coolant_state,
            overheat_state=overheat_state, egt_result=egt_res)
        feature_vector = self._ml_features.feature_vector_from_row(ml_row, rec.timestamp)
        anomaly_res = self._anomaly.detect_anomaly(feature_vector)
        fault_res = self._classifier.classify_fault(feature_vector)
        # Condition flags (symptoms) come from L2 states, whatever the classifier says
        flags, flag_ev = self._condition_flags(overheat_state, comb_state, lub_state, self._settings)
        fault_res = fault_res.model_copy(update={"condition_flags": flags, "condition_evidence": flag_ev})
        # Rule-based diagnosis over the L2 evidence (is_ml=False), reported alongside the ML result
        rule_res = self._classifier.classify_fault_rule_fallback(
            residual_state=res_state, egt_diag=egt_res, vib_state=vib_state, comb_state=comb_state,
            lub_state=lub_state, injection_state=inj_state, elec_state=elec_state,
            coolant_state=coolant_state, overheat_state=overheat_state, norm_record=norm_rec,
            timestamp=rec.timestamp)

        # Health Supervision & RUL: every L2 output the pipeline has
        health_res = self._health.evaluate_health(
            residual_state=res_state,
            anomaly_result=anomaly_res,
            fault_result=fault_res,
            derived_state=twin_res,
            egt_diag=egt_res,
            lub_state=lub_state,
            vib_state=vib_state,
            comb_state=comb_state,
            elec_state=elec_state,
        )
        rul_res = self._rul.estimate_rul(health_state=health_res)
        if self._lifetime_rul is not None and norm_rec.engine_hours.valid:
            rul_res = self._lifetime_rul_state(norm_rec.engine_hours.value, ml_row, rul_res)
        phase, phase_q, phase_tr = self._flight_phase(norm_rec, twin_res)
        if not isinstance(phase, FlightPhase) and self._last_phase is not None:
            # UNKNOWN (e.g. rpm invalid): MissionState would read it as GROUND;
            # carry the last derived phase with quality 0 instead.
            phase, phase_q, phase_tr = self._last_phase, 0.0, "phase not derivable; last derived phase kept"
        risk_res = self._risk.evaluate_mission_risk(
            flight_phase=phase,
            health_state=health_res,
            rul_state=rul_res,
            fault_result=fault_res,
            phase_quality=phase_q,
            previous_phase=self._last_phase,
            phase_transition=phase_tr,
        )
        self._last_phase = phase

        rpm_val = norm_rec.rpm.value if (norm_rec.rpm.valid and norm_rec.rpm.value is not None) else 0.0
        pwr_val = twin_res.brake_power_kw.value if hasattr(twin_res.brake_power_kw, "value") else float(twin_res.brake_power_kw)

        # Versioning (Prompt 17): which baseline and model produced these outputs
        b = self._baseline
        hours = norm_rec.engine_hours.value if norm_rec.engine_hours.valid else None
        vers = {
            "baseline_version": self._residual.baseline_version,
            "adapted_baseline": bool(b is not None and b.corrections),
            "model_version": self._model_version,
            "hours_since_baseline": (hours - b.created_at_hours) if (hours is not None and b is not None
                                                                     and b.created_at_hours is not None) else None,
        }
        health_res = health_res.model_copy(update=vers)
        fault_res = fault_res.model_copy(update=vers)
        rule_res = rule_res.model_copy(update={**vers, "model_version": "rules-only"})

        result = PipelineStepResult(
            timestamp=rec.timestamp,
            sequence_number=rec.sequence_number,
            record=rec,
            accepted=accepted,
            derived_rpm=rpm_val,
            derived_power_kw=pwr_val,
            anomaly_score=anomaly_res.anomaly_score,
            is_anomaly=anomaly_res.is_anomaly,
            predicted_fault_class=getattr(fault_res, "predicted_class", FaultClass.NOMINAL),
            health_index=health_res.health_index.value if hasattr(health_res.health_index, "value") else float(health_res.health_index),
            rul_hours=rul_res.hours_remaining,
            mission_risk_score=risk_res.risk_score,
            derived_state=twin_res,
            electrical_state=elec_state,
            injection_state=inj_state,
            combustion_state=comb_state,
            coolant_state=coolant_state,
            overheat_state=overheat_state,
            health_state=health_res,
            rul_state=rul_res,
            mission_state=risk_res,
            anomaly_result=anomaly_res,
            fault_result=fault_res,
            rule_fault_result=rule_res,
            residual_state=res_state,
            vibration_state=vib_state,
            lubrication_state=lub_state,
            egt_result=egt_res,
            ml_features=ml_row,
            expectation_result=exp_res,
        )
        # drift: the reference comes from the baseline, or is self-commissioned
        # from records the adaptation gate judges healthy
        healthy = not self._adapt_mod.record_eligibility(result, self._settings)
        self._drift.observe(res_state, healthy=healthy)
        result.drift_state = self._drift.evaluate(rec.timestamp)
        return result


class PipelineReplayAdapter:
    """Adapter executing replayed RawSignalRecords through existing L1 -> L2 -> L3 modules."""

    def __init__(self, settings: AppSettings | None = None, baseline: Any = None) -> None:
        self._settings = settings or get_settings()
        self._validator = TelemetryValidator(settings=self._settings)
        from src.l1_data.telemetry_security import PacketSigner
        self._signer = PacketSigner(settings=self._settings)
        self.baseline = baseline  # per-engine EngineBaseline (Prompt 17)

    def new_run(self, validator: TelemetryValidator | None = None) -> PipelineRun:
        """A fresh stateful chain. By default it shares this adapter's validator
        (replay protection across calls); a stream gets its own via `validator`."""
        return PipelineRun(self._settings, validator or self._validator, self._signer, self.baseline)

    def process_sequence(
        self,
        records: list[RawSignalRecord],
    ) -> list[PipelineStepResult]:
        """Execute a sequence of RawSignalRecords through L1 validation -> L2 twin -> L3 supervision,
        stateful across the sequence."""
        run = self.new_run()
        return [run.step(rec) for rec in records]


class RunningPipeline:
    """Keeps one PipelineRun alive across API calls over a growing record
    stream, so every call sees the same history windows as the pipeline
    (OI-20). Only records not yet processed are stepped; the run restarts
    when the stream changes (another source, scenario reload, or rewind)."""

    def __init__(self, adapter: PipelineReplayAdapter) -> None:
        self._adapter = adapter
        self._run: PipelineRun | None = None
        self._key: tuple | None = None
        self._processed: list[RawSignalRecord] = []
        self._latest: PipelineStepResult | None = None

    def advance(self, records: Sequence[RawSignalRecord], source_key: Any = None) -> PipelineStepResult | None:
        if not records:
            return None
        n = len(self._processed)
        key = (source_key, records[0].sequence_number, records[0].timestamp)
        same_stream = (key == self._key and len(records) >= n
                       and (n == 0 or records[n - 1] is self._processed[-1] or records[n - 1] == self._processed[-1]))
        if not same_stream:
            self._run = self._adapter.new_run(validator=TelemetryValidator(settings=self._adapter._settings))
            self._key, self._processed, self._latest = key, [], None
            n = 0
        for rec in records[n:]:
            self._latest = self._run.step(rec)
            self._processed.append(rec)
        return self._latest

    @property
    def records_processed(self) -> int:
        return len(self._processed)


DEFAULT_MISSION_WAYPOINTS: list[dict[str, float]] = [
    # climb power from sea level to 1500 m, then a 60 s power reduction to cruise
    {"t_s": 0.0, "rpm": 5500.0, "map_pa": 135000.0, "altitude_m": 0.0, "airspeed_m_s": 38.0},
    {"t_s": 300.0, "rpm": 5500.0, "map_pa": 135000.0, "altitude_m": 1500.0, "airspeed_m_s": 38.0},
    {"t_s": 360.0, "rpm": 4000.0, "map_pa": 110000.0, "altitude_m": 1500.0, "airspeed_m_s": 50.0},
    {"t_s": 900.0, "rpm": 4000.0, "map_pa": 110000.0, "altitude_m": 1500.0, "airspeed_m_s": 50.0},
]


def mission_profile_from_waypoints(waypoints: list[dict[str, float]], step_s: float = 10.0) -> list[dict[str, float]]:
    """Operating-profile segments of step_s, linearly interpolated between
    waypoints (rpm, map_pa, altitude_m, airspeed_m_s), so power and altitude
    change gradually rather than in one step."""
    keys = ("rpm", "map_pa", "altitude_m", "airspeed_m_s")
    segs: list[dict[str, float]] = []
    t_end = waypoints[-1]["t_s"]
    t = 0.0
    while t < t_end:
        nxt = next(i for i, w in enumerate(waypoints) if w["t_s"] > t)
        a, b = waypoints[nxt - 1], waypoints[nxt]
        f = (t - a["t_s"]) / (b["t_s"] - a["t_s"])
        seg = {k: a[k] + f * (b[k] - a[k]) for k in keys}
        segs.append({"start_time_s": t, "end_time_s": t + step_s, **seg})
        t += step_s
    segs[-1]["end_time_s"] = 1e9
    return segs


def hot_weather_mission_profile(
    isa_deviation_k: float = 30.0,
    waypoints: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Mission profile with an ISA temperature deviation (Prompt 14; SRD-FUN-154).

    Returns a what-if configuration for WhatIfEngine: the profile (rpm, MAP,
    altitude, airspeed) is flown with ambient = ISA(altitude) + isa_deviation_k.
    Run it with WhatIfEngine.execute_what_if and the records through
    PipelineReplayAdapter, the same stack as any other scenario. Default: 5 min
    climb at climb power to 1500 m, 1 min power reduction, cruise.
    """
    profile = mission_profile_from_waypoints(waypoints or DEFAULT_MISSION_WAYPOINTS)
    return {"operating_profile": profile, "isa_deviation_k": float(isa_deviation_k)}


class ScenarioComparator:
    """Deterministic comparator for baseline vs. what-if scenario pipeline outputs."""

    def compare_runs(
        self,
        baseline_results: list[PipelineStepResult],
        what_if_results: list[PipelineStepResult],
        baseline_id: str = "baseline",
        what_if_id: str = "what_if",
        ground_truths: list[SimulationGroundTruth] | None = None,
    ) -> ScenarioComparison:
        """Compare baseline and what-if pipeline outputs with timestamp alignment."""
        if not baseline_results or not what_if_results:
            raise ValueError("Baseline and What-If results must both be non-empty for comparison.")

        # 1. Align time series (match by sequence number or nearest timestamp)
        aligned_pairs: list[tuple[PipelineStepResult, PipelineStepResult]] = []

        min_len = min(len(baseline_results), len(what_if_results))
        for i in range(min_len):
            aligned_pairs.append((baseline_results[i], what_if_results[i]))

        # 2. Calculate summary metric deltas across final/mean values
        base_last = baseline_results[-1]
        what_if_last = what_if_results[-1]

        deltas: dict[str, MetricDelta] = {}

        metrics_to_compare = [
            ("health_index", base_last.health_index, what_if_last.health_index),
            ("anomaly_score", base_last.anomaly_score, what_if_last.anomaly_score),
            ("rul_hours", base_last.rul_hours, what_if_last.rul_hours),
            ("mission_risk_score", base_last.mission_risk_score, what_if_last.mission_risk_score),
            ("derived_power_kw", base_last.derived_power_kw, what_if_last.derived_power_kw),
        ]

        for m_name, b_val, w_val in metrics_to_compare:
            delta_val = w_val - b_val
            pct = (delta_val / abs(b_val) * 100.0) if b_val != 0.0 else 0.0
            deltas[m_name] = MetricDelta(
                metric_name=m_name,
                baseline_value=b_val,
                what_if_value=w_val,
                delta=delta_val,
                pct_change=pct,
            )

        state_changes = {
            "baseline_fault_class": base_last.predicted_fault_class.value,
            "what_if_fault_class": what_if_last.predicted_fault_class.value,
            "fault_class_changed": base_last.predicted_fault_class != what_if_last.predicted_fault_class,
            "baseline_anomaly": base_last.is_anomaly,
            "what_if_anomaly": what_if_last.is_anomaly,
        }

        align_info = {
            "baseline_sample_count": len(baseline_results),
            "what_if_sample_count": len(what_if_results),
            "aligned_sample_count": len(aligned_pairs),
            "alignment_strategy": "sequence_index",
        }

        # 3. Optional post-run ground-truth validation (Ground truth kept hidden during inference)
        gt_validation = None
        if ground_truths and len(ground_truths) >= len(what_if_results):
            correct_fault_predictions = 0
            for step_res, gt in zip(what_if_results, ground_truths):
                if step_res.predicted_fault_class == gt.active_fault:
                    correct_fault_predictions += 1

            accuracy = correct_fault_predictions / float(len(what_if_results))
            gt_validation = {
                "ground_truth_samples": len(ground_truths),
                "fault_classification_accuracy": accuracy,
                "isolated_from_inference": True,
            }

        return ScenarioComparison(
            baseline_id=baseline_id,
            what_if_id=what_if_id,
            metric_deltas=deltas,
            state_changes=state_changes,
            timestamp_alignment_info=align_info,
            ground_truth_validation=gt_validation,
            provenance=Provenance.SIMULATED,
        )
