"""
End-to-end harness for the physics audit (tests and report only; no source).

Real pipeline, stateful per scenario, one record per second:
    ForwardPhysicsModel/SensorForwardModel (ScenarioRunner)
      -> RawSignalRecord -> PacketSigner.sign_record -> TelemetryValidator.validate_packet
         (HMAC verify, replay protection, range and stale checks)
      -> sensor_inverse -> ThermodynamicMechanicalTwin -> EGTDiagnosticsEngine
      -> LubricationModel -> VibrationProcessor -> MisfireDetector (gate + CSI)
      -> InjectionModel (Prompt 13) -> ResidualEngine -> ElectricalModel (Prompt 12)
Replay pipeline: PipelineReplayAdapter.process_sequence over the same records.
API: FastAPI app with the replay engine loaded with the scenario records.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.provenance import FaultClass
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.fault_injection import FaultMode
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticsEngine
from src.l2_digital_twin.cooling_model import CoolantModel
from src.l2_digital_twin.electrical_model import ElectricalModel
from src.l2_digital_twin.injection_model import InjectionModel
from src.l2_digital_twin.lubrication_model import LubricationModel
from src.l2_digital_twin.misfire_classifier import MisfireDetector
from src.l2_digital_twin.residual_engine import ResidualEngine
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import ThermodynamicMechanicalTwin
from src.l2_digital_twin.vibration_processor import VibrationProcessor

DURATION_S = 120
FAULT_ONSET_S = 30.0
RPM = 4000.0
MAP_PA = 110000.0
BAD = ("WARNING", "ALARM", "CRITICAL")


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: FaultClass | FaultMode | None = None  # FaultMode for the simulator-only electrical faults
    severity: float = 0.0
    channel: str | None = None       # SENSOR_FAULT channel
    sub_mode: str | None = None      # CHARGING_FAULT / INJECTOR_FAULT sub-mode
    cylinder: int | None = None      # affected cylinder (default 1)
    strip_bursts: bool = False       # missing burst
    via_csv: bool = False            # CSV replay (OI-3)


SCENARIOS = [
    Scenario("NOMINAL"),
    Scenario("MISFIRE 0.3", FaultClass.MISFIRE, 0.3),
    Scenario("MISFIRE 1.0", FaultClass.MISFIRE, 1.0),
    Scenario("BEARING_WEAR 0.05", FaultClass.BEARING_WEAR, 0.05),
    Scenario("OIL_DEGRADATION 1.0", FaultClass.OIL_DEGRADATION, 1.0),
    Scenario("COOLING_FAULT 1.0", FaultClass.COOLING_FAULT, 1.0),
    Scenario("INTAKE_BOOST_LEAK 1.0", FaultClass.INTAKE_BOOST_LEAK, 1.0),
    Scenario("SENSOR dropout EGT1", FaultClass.SENSOR_FAULT, 1.0, channel="egt_cyl1_hot_uv"),
    Scenario("SENSOR dropout MAP", FaultClass.SENSOR_FAULT, 1.0, channel="map_counts"),
    Scenario("CHARGING_FAULT setpoint_drift 1.0", FaultMode.CHARGING_FAULT, 1.0, sub_mode="setpoint_drift"),
    Scenario("CHARGING_FAULT open_diode 1.0", FaultMode.CHARGING_FAULT, 1.0, sub_mode="open_diode"),
    Scenario("BATTERY_DEGRADATION 1.0", FaultMode.BATTERY_DEGRADATION, 1.0),
    Scenario("INJECTOR_FAULT clog cyl3 0.15", FaultMode.INJECTOR_FAULT, 0.15, sub_mode="clog", cylinder=3),
    Scenario("FUEL_SYSTEM_FAULT 1.0", FaultMode.FUEL_SYSTEM_FAULT, 1.0),
    Scenario("DETONATION_KNOCK cyl2 1.0", FaultClass.DETONATION_KNOCK, 1.0, cylinder=2),
    Scenario("IMBALANCE 1.0", FaultClass.IMBALANCE, 1.0),
    Scenario("NOMINAL, missing burst", strip_bursts=True),
    Scenario("NOMINAL, CSV replay", via_csv=True),
]


def simulate(sc: Scenario) -> list[RawSignalRecord]:
    faults = None
    if sc.fault is not None:
        faults = [FaultScenarioConfig(fault_class=sc.fault, severity=sc.severity, onset_time_s=FAULT_ONSET_S,
                                      duration_s=10_000.0, affected_channel=sc.channel,
                                      sub_mode=sc.sub_mode, affected_cylinder=sc.cylinder)]
    records, _, _ = ScenarioRunner(seed=42).run_scenario(
        duration_s=float(DURATION_S), dt_s=1.0,
        operating_profile=[{"rpm": RPM, "map_pa": MAP_PA, "throttle_pct": 60.0}],
        fault_scenarios=faults,
    )
    if sc.strip_bursts:
        records = [_strip(r) for r in records]
    if sc.via_csv:
        records = _csv_round_trip(records)
    return records


def _strip(rec: RawSignalRecord) -> RawSignalRecord:
    rec = rec.model_copy(update={"accel_burst_counts_x": (), "accel_burst_counts_y": (), "accel_burst_counts_z": (),
                                 "accel_burst_fs_hz": 0.0, "crank_period_burst_us": ()})
    return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})


def _csv_round_trip(records: list[RawSignalRecord]) -> list[RawSignalRecord]:
    """Write the records as CSV (the CSV format has no burst columns) and read
    them back through the real CSVReplayAdapter row parser."""
    from src.l1_data.adapters.csv_replay_adapter import CSVReplayAdapter

    cols = ["timestamp", "sequence_number", "egt_cyl1_hot_uv", "egt_cyl2_hot_uv", "egt_cyl3_hot_uv",
            "egt_cyl4_hot_uv", "egt_cold_c", "cht_hot_uv", "cht_cold_c", "oil_rtd_ohms", "oil_p_counts",
            "map_counts", "adc_vref_counts", "crank_period_us", "fuel_pulse_hz", "accel_x", "accel_y",
            "accel_z", "ambient_temp_c", "ambient_press_pa"]
    path = Path(tempfile.mkdtemp()) / "replay.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            d = r.model_dump()
            d["timestamp"] = r.timestamp.isoformat()
            d["accel_x"], d["accel_y"], d["accel_z"] = r.accel_counts_xyz
            w.writerow([d[c] for c in cols])
    adapter = CSVReplayAdapter(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [adapter._row_to_raw_record(row, int(row["sequence_number"])) for row in rows]


def _v(tv) -> float | None:
    return tv.value if (tv is not None and tv.valid and tv.value is not None) else None


@dataclass
class Snapshot:
    """Indicators at one record."""

    accepted: bool = False
    verdict: str = ""
    misfire_detected: list[bool] = field(default_factory=list)
    combustion_status: str = ""
    csi: float | None = None
    csi_band: str = ""
    csi_coverage: float | None = None
    lhi: float | None = None
    lhi_band: str = ""
    lhi_coverage: float | None = None
    vhi: float | None = None
    vhi_band: str = ""
    envelope_rms: float | None = None
    overall_rms: float | None = None
    egt_status: str = ""
    lub_status: str = ""
    cht_residual: float | None = None
    map_pa: float | None = None
    map_valid: bool = True
    air_kg_s: float | None = None
    sfc: float | None = None
    sfc_band: str = ""
    egt1_valid: bool = True
    record_coverage: float = 1.0
    invalid_channels: list[str] = field(default_factory=list)
    egt_coverage: float = 1.0
    charging_status: str = ""
    ripple_status: str = ""
    battery_status: str = ""
    ehi_band: str = ""
    charging_residual_median: float | None = None
    ripple_pct: float | None = None
    r_int_mohm: float | None = None
    r_int_reason: str = ""
    injection_status: str = ""
    fuel_system_status: str = ""
    knock_status: str = ""
    identified_injectors: list[int] = field(default_factory=list)
    knock_suspected: list[bool] = field(default_factory=list)
    rail_residual_kpa: float | None = None
    fuel_delivery_ratio: float | None = None
    injector_flow_ratio: list[float | None] = field(default_factory=list)
    egt_differential_k: list[float | None] = field(default_factory=list)
    injector_flow_reason: list[str] = field(default_factory=list)
    coolant_temp_c: float | None = None
    coolant_residual_k: float | None = None
    coolant_status: str = ""

    def overall(self) -> str:
        bands = [self.combustion_status, self.csi_band, self.lhi_band, self.vhi_band,
                 self.egt_status, self.lub_status, self.ehi_band]
        if any(b in ("ALARM", "CRITICAL") for b in bands):
            return "ALARM"
        if any(b == "WARNING" for b in bands):
            return "WARNING"
        if any(b in ("INVALID", "UNKNOWN") for b in bands):
            return "INVALID/REDUCED"
        if self.record_coverage < 1.0 or self.egt_coverage < 1.0:
            return "REDUCED COVERAGE"
        return "NORMAL"

    def bad_bands(self) -> list[str]:
        named = {"combustion": self.combustion_status, "csi": self.csi_band, "lhi": self.lhi_band,
                 "vhi": self.vhi_band, "egt": self.egt_status, "lubrication": self.lub_status,
                 "electrical": self.ehi_band,
                 "sfc": self.sfc_band if self.sfc_band not in ("BELOW_EXPECTED",) else ""}
        return [f"{k}={v}" for k, v in named.items() if v in BAD]


def run_real_pipeline(records: list[RawSignalRecord]) -> list[Snapshot]:
    signer, validator = PacketSigner(), TelemetryValidator()
    twin, egt_eng, lub, vib, mis, res = (ThermodynamicMechanicalTwin(), EGTDiagnosticsEngine(), LubricationModel(),
                                        VibrationProcessor(), MisfireDetector(), ResidualEngine())
    elec = ElectricalModel()
    inj_model = InjectionModel()
    coolant_model = CoolantModel()
    out: list[Snapshot] = []
    for rec in records:
        val = validator.validate_packet(rec, signature=signer.sign_record(rec))
        snap = Snapshot(accepted=val.accepted)
        if not val.accepted or val.record is None:
            out.append(snap)
            continue
        norm = convert_raw_to_engineering_state(val.record)
        derived = twin.evaluate(norm)
        egt_state, egt_res = egt_eng.evaluate(norm)
        lub_state, _ = lub.evaluate(norm, derived)
        vib_state, vib_res = vib.process_record(norm)
        comb_state, comb_res = mis.evaluate(norm, derived, egt_res, vib_res)
        inj = inj_model.evaluate(norm, derived, comb_state)
        _, exp = res.evaluate(norm, derived, injection_state=inj)
        thermo = twin.thermo_twin.compute(norm)
        snap.verdict = comb_state.misfire_verdict
        snap.misfire_detected = list(comb_state.misfire_detected)
        snap.combustion_status = comb_state.overall_combustion_status.value
        snap.csi, snap.csi_band = _v(comb_state.csi_value), comb_state.csi_band
        snap.csi_coverage = comb_res.csi_terms.get("coverage")
        snap.lhi, snap.lhi_band, snap.lhi_coverage = _v(lub_state.lhi), lub_state.lhi_band, lub_state.lhi_coverage
        snap.vhi, snap.vhi_band = _v(vib_state.vibration_health_index), vib_state.vhi_band
        snap.envelope_rms, snap.overall_rms = _v(vib_state.envelope_rms_m_s2), _v(vib_state.overall_rms_m_s2)
        snap.egt_status = egt_res.overall_status.value
        snap.lub_status = lub_state.status.value
        cht = exp.residuals.get("cht")
        snap.cht_residual = cht.raw_residual if (cht is not None and cht.valid) else None
        snap.map_pa = norm.map_pressure.value if norm.map_pressure.valid else None
        snap.map_valid = norm.map_pressure.valid
        snap.air_kg_s = _v(thermo.air_mass_flow_kg_s)
        snap.sfc, snap.sfc_band = _v(derived.sfc_kg_kwh), derived.sfc_band
        snap.egt1_valid = norm.egt_cyl_1.valid
        snap.record_coverage = norm.channel_coverage
        snap.invalid_channels = list(norm.invalid_channels)
        snap.egt_coverage = egt_res.coverage
        el = elec.evaluate(norm)
        snap.charging_status, snap.ripple_status = el.charging_status.value, el.ripple_status.value
        snap.battery_status, snap.ehi_band = el.battery_status.value, el.ehi_band
        snap.charging_residual_median = _v(el.charging_residual_median_v)
        snap.ripple_pct = _v(el.voltage_ripple_pct)
        snap.r_int_mohm = _v(el.battery_resistance_mohm)
        snap.r_int_reason = el.battery_resistance_mohm.fault_flag or ""
        snap.injection_status, snap.fuel_system_status = inj.status.value, inj.fuel_system_status.value
        snap.knock_status, snap.knock_suspected = inj.knock_status.value, list(inj.knock_suspected)
        snap.identified_injectors = list(inj.identified_injectors)
        snap.rail_residual_kpa = _v(inj.rail_pressure_residual_median_kpa)
        snap.fuel_delivery_ratio = _v(inj.fuel_delivery_ratio_median)
        snap.injector_flow_ratio = [_v(tv) for tv in inj.injector_flow_ratio_median]
        snap.egt_differential_k = [_v(tv) for tv in inj.egt_differential_k]
        snap.injector_flow_reason = [tv.fault_flag or "" for tv in inj.injector_flow_ratio]
        cs = coolant_model.evaluate(norm)
        snap.coolant_temp_c, snap.coolant_residual_k = _v(cs.coolant_temp_c), _v(cs.coolant_residual_median_k)
        snap.coolant_status = cs.status.value
        out.append(snap)
    return out


def run_replay_pipeline(records: list[RawSignalRecord]):
    return PipelineReplayAdapter().process_sequence(records)


API_ENDPOINTS = ["/api/v1/diagnostics", "/api/v1/diagnostics/egt", "/api/v1/diagnostics/lubrication",
                 "/api/v1/diagnostics/vibration", "/api/v1/diagnostics/combustion", "/api/v1/engine/health"]


def api_responses(records: list[RawSignalRecord]) -> dict[str, tuple[int, str]]:
    """Real FastAPI app, replay engine loaded with this scenario."""
    from fastapi.testclient import TestClient

    from src.api.app import create_app
    from src.api.dependencies import get_replay_engine

    engine = ReplayEngine(scenario_id="audit")
    engine.load_scenario("audit", records, [])
    app = create_app()
    app.dependency_overrides[get_replay_engine] = lambda: engine
    out = {}
    with TestClient(app, raise_server_exceptions=False) as client:
        for ep in API_ENDPOINTS:
            r = client.get(ep)
            out[ep] = (r.status_code, r.text)
    return out


def json_has_nan(text: str) -> bool:
    """True if the payload carries NaN/Infinity (literal tokens or floats)."""
    if any(tok in text for tok in ("NaN", "Infinity")):
        return True
    try:
        data = json.loads(text)
    except ValueError:
        return False

    def walk(x: Any) -> bool:
        if isinstance(x, float):
            return not math.isfinite(x)
        if isinstance(x, dict):
            return any(walk(v) for v in x.values())
        if isinstance(x, list):
            return any(walk(v) for v in x)
        return False

    return walk(data)


def invalid_values_are_null(text: str) -> list[str]:
    """Paths of tagged values with valid=false but a non-null value."""
    bad: list[str] = []

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            if x.get("valid") is False and "value" in x and x["value"] is not None:
                bad.append(f"{path} value={x['value']}")
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    try:
        walk(json.loads(text), "$")
    except ValueError:
        pass
    return bad
