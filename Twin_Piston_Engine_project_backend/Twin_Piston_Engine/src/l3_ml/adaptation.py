"""
Gated adaptive learning (Prompt 17; SIH section D).

The fleet expectation models (M-11, SRD-FUN-105) and the frozen model bundles
(Prompt 16, SRD-FUN-110) are NEVER modified. Adaptation acts only on
PER-ENGINE baselines: a bounded correction of each residual channel's
expectation, expected' = expected + a + b (expected - ref), stored as a new
baseline version with the old ones kept.

TARGET (Prompt 17b): the correction moves the engine's residual towards the
FLEET's healthy residual at the same operating point
(src/l3_ml/fleet_reference.py), not towards zero. The fleet residual carries
the operating-point dependent expectation bias (OI-5) that the frozen models
learned as healthy; zeroing it pushed their inputs out of distribution (OI-25).
Drift is measured on the deviation from that reference, so a change of
operating point is not drift.

Loop (post-flight, one flight = the records of one flight or monitoring window):

  a. ELIGIBILITY GATE (the most important part). A flight is used only if, for
     EVERY record: the record was accepted by L1; no anomaly alarm; health
     status NORMAL and degradation state HEALTHY or WATCH; rule-based class
     NOMINAL; ML class NOMINAL or its confidence at or below the threshold; no
     condition flag set; health index in the top band; every core residual
     channel valid. Otherwise adaptation is frozen for that flight and the
     reasons are logged. A degrading engine must never be learned as healthy.
  b. BASELINE REFIT. Commissioning: the first eligible flights (at least
     `commissioning_flights` and drift.reference_min_records records) fit (a, b) per channel (a within commissioning_max_offset, |b| <=
     max_linear_gain); the reference drift histograms come from the corrected
     residuals. Afterwards, on an eligible flight with MODERATE drift, only a
     is updated: at most rate_limit_per_flight per flight and at most
     max_total_from_commissioning from the commissioning value.
  c. CHALLENGER after `challenger_after_flights` eligible flights (or labelled
     maintenance findings): retrained with the Prompt 16 pipeline, the new
     rows added to TRAIN only.
  d. PROMOTION GATE on the FROZEN Prompt 16 test split: macro-F1 and PR-AUC
     may not drop by more than the configured margins and FPR must stay
     <= promotion_max_fpr; otherwise rejected with the reasons logged.
  e. ROLLBACK: one call restores the previous baseline or model version.
     Every decision goes to an append-only, hash-chained log.
  f. SIGNIFICANT drift (eligible or not) never adapts: an advisory is raised
     ("data distribution has shifted; possible sensor drift or unmodelled
     condition"), cross-checked against SENSOR_FAULT evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.provenance import DegradationState, DiagnosticStatus, FaultClass, InferenceStatus
from collections import deque

from src.l3_ml.drift_monitor import DRIFT_CHANNELS, DriftReference, overall_score, psi
from src.l3_ml.fleet_reference import FleetResidualReference, load_for_settings, op_vector

ADAPTED_CHANNELS: tuple[str, ...] = ("egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4", "cht", "coolant", "fuel_flow",
                                     "oil_pressure", "oil_temp", "vibration_rms", "brake_power_kw")
CORE_CHANNELS: tuple[str, ...] = ("egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4", "cht", "oil_pressure", "oil_temp",
                                  "fuel_flow")
DRIFT_ADVISORY = "Data distribution has shifted; possible sensor drift or unmodelled condition."


def channel_group(ch: str) -> str:
    return "egt" if ch.startswith("egt_") else ch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Append-only decision log (SRD-QUA-003 style)
# ---------------------------------------------------------------------------

class AdaptationLog:
    """JSON-lines, append-only; each entry carries the hash of the previous one."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, kind: str, decision: str, reasons: list[str] | None = None, **details: Any) -> dict[str, Any]:
        prev = self.entries()
        entry = {"seq": len(prev), "time_utc": _now(), "kind": kind, "decision": decision,
                 "reasons": list(reasons or []), "details": details,
                 "prev_hash": prev[-1]["hash"] if prev else "0" * 64}
        entry["hash"] = hashlib.sha256(json.dumps({k: v for k, v in entry.items() if k != "hash"},
                                                  sort_keys=True, default=str).encode()).hexdigest()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def verify(self) -> bool:
        prev = "0" * 64
        for e in self.entries():
            h = hashlib.sha256(json.dumps({k: v for k, v in e.items() if k != "hash"}, sort_keys=True,
                                          default=str).encode()).hexdigest()
            if e["prev_hash"] != prev or e["hash"] != h:
                return False
            prev = e["hash"]
        return True


# ---------------------------------------------------------------------------
# Per-engine baselines
# ---------------------------------------------------------------------------

@dataclass
class EngineBaseline:
    engine_id: str
    version: str
    corrections: dict[str, list[float]] = field(default_factory=dict)   # channel -> [a, b, ref]
    commissioning: dict[str, float] = field(default_factory=dict)       # channel -> a at commissioning
    reference: dict[str, Any] | None = None                            # DriftReference.to_dict()
    created_at_hours: float | None = None
    parent: str | None = None
    kind: str = "fleet"                                                 # fleet / commissioning / adapted

    def corrections_tuple(self) -> dict[str, tuple[float, float, float]]:
        return {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in self.corrections.items()}

    @property
    def adapted(self) -> bool:
        return bool(self.corrections)


def fleet_baseline(engine_id: str) -> EngineBaseline:
    return EngineBaseline(engine_id=engine_id, version="fleet-0")


class BaselineStore:
    """Versioned per-engine baselines on disk; old versions are kept and the
    active pointer can be rolled back."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, engine_id: str) -> Path:
        return self.root / f"baselines_{engine_id}.json"

    def _load(self, engine_id: str) -> dict[str, Any]:
        p = self._path(engine_id)
        return json.loads(p.read_text()) if p.exists() else {"versions": {}, "history": []}

    def _save(self, engine_id: str, d: dict[str, Any]) -> None:
        self._path(engine_id).write_text(json.dumps(d, indent=1, default=str))

    def active(self, engine_id: str) -> EngineBaseline:
        d = self._load(engine_id)
        if not d["history"]:
            return fleet_baseline(engine_id)
        return EngineBaseline(**d["versions"][d["history"][-1]])

    def versions(self, engine_id: str) -> list[str]:
        return list(self._load(engine_id)["versions"])

    def history(self, engine_id: str) -> list[str]:
        return list(self._load(engine_id)["history"])

    def add(self, baseline: EngineBaseline) -> None:
        d = self._load(baseline.engine_id)
        assert baseline.version not in d["versions"], "baseline versions are immutable"
        d["versions"][baseline.version] = asdict(baseline)
        d["history"].append(baseline.version)
        self._save(baseline.engine_id, d)

    def rollback(self, engine_id: str) -> EngineBaseline:
        d = self._load(engine_id)
        if d["history"]:
            d["history"].pop()
        self._save(engine_id, d)
        return self.active(engine_id)


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def record_eligibility(step: Any, settings: AppSettings) -> list[str]:
    """Reasons this record is NOT healthy (empty list = healthy)."""
    reasons: list[str] = []
    cfg = settings.adaptation
    if not getattr(step, "accepted", True):
        reasons.append("record rejected by L1 validation")
    an = step.anomaly_result
    if an is not None and an.status == InferenceStatus.SUCCESS and an.is_anomaly:
        reasons.append("anomaly alarm")
    hs = step.health_state
    if hs is None or not hs.health_index.valid:
        reasons.append("health index not valid")
    else:
        if hs.health_status != DiagnosticStatus.NORMAL or hs.degradation_state not in (
                DegradationState.HEALTHY, DegradationState.WATCH):
            reasons.append(f"alert above WATCH ({hs.health_status.value}/{hs.degradation_state.value})")
        if hs.health_index.value < settings.health.healthy_threshold:
            reasons.append(f"health index {hs.health_index.value:.3f} below the top band "
                           f"({settings.health.healthy_threshold})")
    rule = step.rule_fault_result
    if rule is not None and rule.predicted_class != FaultClass.NOMINAL:
        reasons.append(f"rule-based class {rule.class_name}")
    ml = step.fault_result
    if ml is not None and ml.status == InferenceStatus.SUCCESS and ml.predicted_class != FaultClass.NOMINAL \
            and (ml.confidence or 0.0) > cfg.fault_confidence_threshold:
        reasons.append(f"ML class {ml.class_name} (confidence {ml.confidence:.2f})")
    flags = ml.condition_flags if ml is not None else {}
    for name, v in flags.items():
        if v is True:
            reasons.append(f"condition flag {name}")
    res = step.residual_state.residuals if step.residual_state is not None else {}
    bad = [ch for ch in CORE_CHANNELS if not (res.get(ch) is not None and res[ch].valid)]
    if bad:
        reasons.append(f"sensor channels invalid: {bad}")
    return reasons


def flight_eligibility(steps: list[Any], settings: AppSettings) -> tuple[bool, list[str]]:
    counts: dict[str, int] = {}
    for st in steps:
        for r in record_eligibility(st, settings):
            key = r.split(" (")[0] if r.startswith("health index ") else r
            counts[key] = counts.get(key, 0) + 1
    reasons = [f"{k} ({n} of {len(steps)} records)" for k, n in counts.items()]
    return (not reasons and len(steps) > 0), reasons


# ---------------------------------------------------------------------------
# Baseline fitting
# ---------------------------------------------------------------------------

def _reference_values(steps: list[Any], fleet_ref: FleetResidualReference | None,
                      channels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Fleet healthy residual per record and channel (0 without a reference,
    NaN where the operating point is incomplete)."""
    n = len(steps)
    if fleet_ref is None:
        return {ch: np.zeros(n) for ch in channels}
    ops = np.full((n, 5), np.nan)
    for i, st in enumerate(steps):
        er = st.expectation_result
        v = op_vector(er.operating_point) if er is not None else None
        if v is not None:
            ops[i] = v
    return {ch: fleet_ref.predict_many(ops, ch) if ch in fleet_ref.coefs else np.zeros(n) for ch in channels}


def _fleet_residuals(steps: list[Any], fleet_ref: FleetResidualReference | None = None
                     ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """channel -> (raw fleet expectation, observed, fleet healthy residual at
    the record's operating point) over the flight's records."""
    ref = _reference_values(steps, fleet_ref, ADAPTED_CHANNELS)
    out: dict[str, tuple[list[float], list[float], list[float]]] = {ch: ([], [], []) for ch in ADAPTED_CHANNELS}
    for i, st in enumerate(steps):
        er = st.expectation_result
        if er is None:
            continue
        for ch in ADAPTED_CHANNELS:
            q = er.residuals.get(ch)
            e = er.raw_expected_values.get(ch)
            f = ref[ch][i]
            if q is not None and q.valid and q.observed is not None and e is not None and math.isfinite(e) \
                    and math.isfinite(f):
                out[ch][0].append(float(e))
                out[ch][1].append(float(q.observed))
                out[ch][2].append(float(f))
    return {ch: (np.asarray(e), np.asarray(o), np.asarray(f)) for ch, (e, o, f) in out.items() if e}


def fit_commissioning(flights: list[list[Any]], engine_id: str, hours: float | None, settings: AppSettings,
                      version: str, fleet_ref: FleetResidualReference | None = None) -> EngineBaseline:
    """(a, b) per channel so that the corrected residual matches the fleet's
    healthy residual at the same operating point: d = observed - expected -
    fleet_reference(op); b = slope of d on (expected - ref), a = median rest."""
    cfg = settings.adaptation
    pooled: dict[str, list[list[np.ndarray]]] = {}
    for steps in flights:
        for ch, (e, o, f) in _fleet_residuals(steps, fleet_ref).items():
            pooled.setdefault(ch, [[], [], []])
            pooled[ch][0].append(e)
            pooled[ch][1].append(o)
            pooled[ch][2].append(f)
    corrections: dict[str, list[float]] = {}
    for ch, (es, os_, fs) in pooled.items():
        e, o, f = np.concatenate(es), np.concatenate(os_), np.concatenate(fs)
        d = o - e - f
        ref = float(np.median(e))
        de = e - ref
        b = 0.0
        if np.std(de) > 1e-9 and len(e) >= 10:
            b = float(np.clip(np.polyfit(de, d, 1)[0], -cfg.max_linear_gain, cfg.max_linear_gain))
        bound = cfg.commissioning_max_offset.get(channel_group(ch), 0.0)
        a = float(np.clip(np.median(d - b * de), -bound, bound))
        corrections[ch] = [a, b, ref]
    base = EngineBaseline(engine_id=engine_id, version=version, corrections=corrections,
                          commissioning={ch: c[0] for ch, c in corrections.items()}, created_at_hours=hours,
                          kind="commissioning")
    base.reference = reference_from(flights, base, settings, fleet_ref).to_dict()
    return base


def corrected_residuals(steps: list[Any], baseline: EngineBaseline,
                        fleet_ref: FleetResidualReference | None = None) -> dict[str, list[float]]:
    """Deviation of every drift channel from the fleet's healthy residual at the
    record's operating point, under this baseline (plain residuals when there
    is no fleet reference)."""
    out: dict[str, list[float]] = {ch: [] for ch in DRIFT_CHANNELS}
    corr = baseline.corrections_tuple()
    ref = _reference_values(steps, fleet_ref, DRIFT_CHANNELS)
    for i, st in enumerate(steps):
        er = st.expectation_result
        res = st.residual_state.residuals if st.residual_state is not None else {}
        for ch in DRIFT_CHANNELS:
            f = ref[ch][i]
            if not math.isfinite(f):
                continue
            if ch in ADAPTED_CHANNELS and er is not None:
                q, e = er.residuals.get(ch), er.raw_expected_values.get(ch)
                if q is None or not q.valid or q.observed is None or e is None:
                    continue
                a, b, r0 = corr.get(ch, (0.0, 0.0, 0.0))
                out[ch].append(float(q.observed) - (e + a + b * (e - r0)) - f)
            else:
                tv = res.get(ch)
                if tv is not None and tv.valid and tv.value is not None and math.isfinite(tv.value):
                    out[ch].append(float(tv.value) - f)
    return out


def reference_from(flights: list[list[Any]], baseline: EngineBaseline, settings: AppSettings,
                   fleet_ref: FleetResidualReference | None = None) -> DriftReference:
    pooled: dict[str, list[float]] = {ch: [] for ch in DRIFT_CHANNELS}
    for steps in flights:
        for ch, v in corrected_residuals(steps, baseline, fleet_ref).items():
            pooled[ch].extend(v)
    return DriftReference.fit(pooled, settings.drift.bins, baseline.version, settings.drift.reference_min_records)


# ---------------------------------------------------------------------------
# Adaptation manager (post-flight)
# ---------------------------------------------------------------------------

@dataclass
class AdaptationDecision:
    engine_id: str
    decision: str                   # commissioning_pending / commissioned / adapted / no_change / refused / frozen
    eligible: bool
    reasons: list[str]
    baseline_version: str
    drift_psi: dict[str, float] = field(default_factory=dict)
    drift_band: str = "UNKNOWN"
    advisory: str | None = None
    challenger_due: bool = False


class AdaptationManager:
    def __init__(self, store: BaselineStore, log: AdaptationLog, settings: AppSettings | None = None,
                 fleet_reference: FleetResidualReference | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.adaptation
        self.fleet_ref = fleet_reference if fleet_reference is not None else load_for_settings(self.settings)
        self.store, self.log = store, log
        self._commissioning: dict[str, list[list[Any]]] = {}
        self._eligible_since_model: dict[str, int] = {}
        self._recent: dict[str, deque] = {}

    def _drift(self, steps: list[Any], base: EngineBaseline, engine_id: str) -> tuple[dict[str, float], str]:
        """PSI of the recent flights (this one plus earlier ones until the window
        has drift.window_records records) under the current baseline vs its reference."""
        recent = self._recent.setdefault(engine_id, deque(maxlen=200))
        recent.append(steps)
        if not base.reference:
            return {}, "UNKNOWN"
        ref = DriftReference.from_dict(base.reference)
        pooled: dict[str, list[float]] = {ch: [] for ch in DRIFT_CHANNELS}
        n = 0
        for fl in reversed(recent):
            for ch, vals in corrected_residuals(fl, base, self.fleet_ref).items():
                pooled[ch].extend(vals)
            n += len(fl)
            if n >= self.settings.drift.window_records:
                break
        if n < self.settings.drift.min_window_records:
            return {}, "UNKNOWN"
        scores = {ch: psi(ref.probs[ch], ref.edges[ch], vals, self.settings.drift.epsilon)
                  for ch, vals in pooled.items() if ch in ref.probs and vals}
        worst = overall_score(scores)
        if worst is None:
            return scores, "UNKNOWN"
        band = "STABLE" if worst < self.settings.drift.psi_moderate else (
            "MODERATE" if worst < self.settings.drift.psi_significant else "SIGNIFICANT")
        return scores, band

    def process_flight(self, engine_id: str, steps: list[Any], engine_hours: float | None = None) -> AdaptationDecision:
        eligible, reasons = flight_eligibility(steps, self.settings)
        base = self.store.active(engine_id)
        scores, band = self._drift(steps, base, engine_id)
        sensor_evidence = any(st.rule_fault_result is not None and st.rule_fault_result.predicted_class
                              == FaultClass.SENSOR_FAULT for st in steps)
        dec = AdaptationDecision(engine_id, "refused", eligible, reasons, base.version, scores, band)
        if eligible:
            self._eligible_since_model[engine_id] = self._eligible_since_model.get(engine_id, 0) + 1
            dec.challenger_due = self._eligible_since_model[engine_id] >= self.cfg.challenger_after_flights

        if base.kind == "fleet":                                  # commissioning phase
            if not eligible:
                self.log.append("baseline", "refused", reasons, engine_id=engine_id, phase="commissioning",
                                engine_hours=engine_hours)
                return dec
            if self.fleet_ref is None:
                why = ["no fleet healthy-residual reference (adaptation.fleet_reference_version): cannot commission"]
                dec.reasons = why
                self.log.append("baseline", "refused", why, engine_id=engine_id, phase="commissioning",
                                engine_hours=engine_hours)
                return dec
            flights = self._commissioning.setdefault(engine_id, [])
            flights.append(steps)
            if len(flights) < self.cfg.commissioning_flights or \
                    sum(len(f) for f in flights) < self.settings.drift.reference_min_records:
                dec.decision = "commissioning_pending"
                self.log.append("baseline", "commissioning_pending", [], engine_id=engine_id,
                                eligible_flights=len(flights), engine_hours=engine_hours)
                return dec
            new = fit_commissioning(flights, engine_id, engine_hours, self.settings, f"{engine_id}-v1",
                                    self.fleet_ref)
            self.store.add(new)
            self._commissioning.pop(engine_id, None)
            dec.decision, dec.baseline_version = "commissioned", new.version
            self.log.append("baseline", "commissioned", [], engine_id=engine_id, version=new.version,
                            corrections=new.corrections, engine_hours=engine_hours)
            return dec

        if band == "SIGNIFICANT":                               # never adapt on significant drift
            dec.decision = "frozen"
            dec.advisory = DRIFT_ADVISORY + (" SENSOR_FAULT evidence present in this flight: sensor drift is likely."
                                             if sensor_evidence else " No SENSOR_FAULT evidence in this flight.")
            self.log.append("baseline", "frozen", reasons + ["significant drift"], engine_id=engine_id,
                            drift_psi=scores, eligible=eligible, advisory=dec.advisory, engine_hours=engine_hours)
            return dec
        if not eligible:
            self.log.append("baseline", "refused", reasons, engine_id=engine_id, drift_band=band,
                            engine_hours=engine_hours)
            return dec
        if band != "MODERATE":
            dec.decision = "no_change"
            self.log.append("baseline", "no_change", [f"drift {band}"], engine_id=engine_id, engine_hours=engine_hours)
            return dec

        # eligible + moderate drift: rate-limited, capped offset update
        fleet = _fleet_residuals(steps, self.fleet_ref)
        corrections = {ch: list(v) for ch, v in base.corrections.items()}
        changes = {}
        for ch, (e, o, f) in fleet.items():
            if ch not in corrections:
                continue
            a, b, ref = corrections[ch]
            target = float(np.median(o - e - f - b * (e - ref)))
            g = channel_group(ch)
            step = float(np.clip(target - a, -self.cfg.rate_limit_per_flight[g], self.cfg.rate_limit_per_flight[g]))
            a0 = base.commissioning.get(ch, a)
            cap = self.cfg.max_total_from_commissioning[g]
            new_a = float(np.clip(a + step, a0 - cap, a0 + cap))
            if new_a != a:
                corrections[ch] = [new_a, b, ref]
                changes[ch] = new_a - a
        if not changes:
            dec.decision = "no_change"
            self.log.append("baseline", "no_change", ["moderate drift but corrections at their caps"],
                            engine_id=engine_id, engine_hours=engine_hours)
            return dec
        n = len(self.store.versions(engine_id)) + 1
        new = EngineBaseline(engine_id=engine_id, version=f"{engine_id}-v{n}", corrections=corrections,
                             commissioning=dict(base.commissioning), reference=base.reference,
                             created_at_hours=engine_hours, parent=base.version, kind="adapted")
        self.store.add(new)
        dec.decision, dec.baseline_version = "adapted", new.version
        self.log.append("baseline", "adapted", [f"drift {band}"], engine_id=engine_id, version=new.version,
                        parent=base.version, changes=changes, drift_psi=scores, engine_hours=engine_hours)
        return dec

    def rollback_baseline(self, engine_id: str) -> EngineBaseline:
        before = self.store.active(engine_id).version
        after = self.store.rollback(engine_id)
        self.log.append("baseline", "rollback", [], engine_id=engine_id, from_version=before, to_version=after.version)
        return after


# ---------------------------------------------------------------------------
# Models: registry, challenger, promotion gate, rollback
# ---------------------------------------------------------------------------

class ModelRegistry:
    """models/registry.json: active model version, activation history, and the
    metrics of every evaluated version. Bundles themselves are never modified."""

    def __init__(self, model_root: str | Path) -> None:
        self.root = Path(model_root)
        self.path = self.root / "registry.json"

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text()) if self.path.exists() else {"history": [], "versions": {}}

    def _save(self, d: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=1, default=str))

    def active(self) -> str | None:
        h = self.load()["history"]
        return h[-1] if h else None

    def record(self, version: str, metrics: dict[str, Any], status: str) -> None:
        d = self.load()
        d["versions"][version] = {"metrics": metrics, "status": status, "recorded_utc": _now()}
        self._save(d)

    def activate(self, version: str) -> None:
        d = self.load()
        d["history"].append(version)
        self._save(d)

    def rollback(self) -> str | None:
        d = self.load()
        if len(d["history"]) > 1:
            d["history"].pop()
        self._save(d)
        return self.active()


def promotion_gate(champion: dict[str, float], challenger: dict[str, float], settings: AppSettings) -> tuple[bool, list[str]]:
    cfg = settings.adaptation
    reasons = []
    if challenger["macro_f1"] < champion["macro_f1"] - cfg.promotion_max_f1_drop:
        reasons.append(f"macro-F1 {challenger['macro_f1']:.3f} < champion {champion['macro_f1']:.3f} - "
                       f"{cfg.promotion_max_f1_drop}")
    if challenger["pr_auc"] < champion["pr_auc"] - cfg.promotion_max_pr_auc_drop:
        reasons.append(f"PR-AUC {challenger['pr_auc']:.3f} < champion {champion['pr_auc']:.3f} - "
                       f"{cfg.promotion_max_pr_auc_drop}")
    if challenger["fpr"] > cfg.promotion_max_fpr:
        reasons.append(f"FPR {challenger['fpr']:.3f} > {cfg.promotion_max_fpr}")
    return (not reasons), reasons


def frozen_test_metrics(model_dir: str | Path, test_df) -> dict[str, float]:
    """Anomaly PR-AUC / FPR and classifier macro-F1 of a bundle on the frozen test rows."""
    import joblib
    from sklearn.metrics import average_precision_score, f1_score

    from src.core.provenance import FAULT_CLASS_COUNT
    from src.l3_ml.ml_features import ML_FEATURES
    d = Path(model_dir)
    an, fc = joblib.load(d / "anomaly_detector.joblib"), joblib.load(d / "fault_classifier.joblib")
    x, y = test_df[list(ML_FEATURES)].to_numpy(float), test_df["label"].to_numpy(int)
    s = -an["estimator"].score_samples(x)
    pos = y != 0
    pred = np.asarray(fc["estimator"].classes_)[fc["estimator"].predict_proba(x).argmax(1)]
    return {"pr_auc": float(average_precision_score(pos, s)), "fpr": float(np.mean(s[~pos] >= an["threshold"])),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(FAULT_CLASS_COUNT)),
                                       zero_division=0))}


def train_challenger(train_df, val_df, extra_rows, model_root: str | Path, version: str, seed: int,
                     provenance: dict[str, Any]) -> Path:
    """Retrain with the Prompt 16 pipeline; extra (eligible-flight / maintenance)
    rows are added to TRAIN only and may not come from the frozen test split."""
    import joblib
    import pandas as pd

    from src.l3_ml import training as T
    from src.l3_ml.ml_features import ML_FEATURES
    if extra_rows is not None and len(extra_rows):
        assert "split" not in extra_rows or not (extra_rows["split"] == "test").any(), \
            "adaptation must never use the frozen test split"
        train_df = pd.concat([train_df, extra_rows], ignore_index=True)
    out = Path(model_root) / version
    assert not out.exists(), "model versions are immutable; choose a new version"
    out.mkdir(parents=True)
    an = T.train_anomaly_detector(train_df, val_df, seed)
    joblib.dump(T.make_bundle("anomaly", "anomaly_detector", version, an["estimator"], ML_FEATURES,
                              {"threshold": an["threshold"], "group_missing_rate": an["group_missing_rate"]},
                              provenance, an["val_metrics"], "challenger"), out / "anomaly_detector.joblib")
    fc = T.train_fault_classifier(train_df, val_df, seed)
    joblib.dump(T.make_bundle("classifier", "fault_classifier", version, fc["estimator"], ML_FEATURES,
                              {"importance": fc["importance"], "group_missing_rate": fc["group_missing_rate"]},
                              provenance, fc["val_metrics"], "challenger"), out / "fault_classifier.joblib")
    return out


def evaluate_and_gate(registry: ModelRegistry, log: AdaptationLog, champion: str, challenger: str,
                      model_root: str | Path, frozen_test_df, settings: AppSettings | None = None) -> bool:
    settings = settings or get_settings()
    m_champ = frozen_test_metrics(Path(model_root) / champion, frozen_test_df)
    m_chall = frozen_test_metrics(Path(model_root) / challenger, frozen_test_df)
    ok, reasons = promotion_gate(m_champ, m_chall, settings)
    registry.record(champion, m_champ, "champion")
    registry.record(challenger, m_chall, "promoted" if ok else "rejected")
    if ok:
        registry.activate(challenger)
    log.append("model", "promoted" if ok else "rejected", reasons, champion=champion, challenger=challenger,
               champion_metrics=m_champ, challenger_metrics=m_chall)
    return ok
