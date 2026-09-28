"""
Prompt 17: drift monitoring and gated adaptive learning with versioning and rollback.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.dependencies import get_app_settings, get_replay_engine, get_running_pipeline  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.core.provenance import FaultClass, Provenance  # noqa: E402
from src.core.schemas import OperatingPoint, ResidualState, make_tagged  # noqa: E402
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner  # noqa: E402
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine, RunningPipeline  # noqa: E402
from src.l3_ml import adaptation as A  # noqa: E402
from src.l3_ml.drift_monitor import DRIFT_CHANNELS, DriftMonitor  # noqa: E402
from src.l3_ml.fleet_reference import OP_KEYS, FleetResidualReference  # noqa: E402

OP = OperatingPoint(rpm=4000.0, map_pressure_pa=110000.0, altitude_m=0.0, ambient_temp_k=288.15,
                    ambient_pressure_pa=101325.0)
CRUISES = [(4000.0, 110000.0), (4200.0, 115000.0), (3800.0, 105000.0)]


def _flight(settings, seed: int, rpm: float = 4000.0, map_pa: float = 110000.0, duration: float = 80.0,
            baseline=None, faults=None):
    recs, _, _ = ScenarioRunner(settings=settings, seed=seed).run_scenario(
        duration_s=duration, dt_s=1.0, operating_profile=[{"rpm": rpm, "map_pa": map_pa}], fault_scenarios=faults)
    return recs, PipelineReplayAdapter(settings, baseline=baseline).process_sequence(recs)


def _median_res(steps, ch: str) -> float:
    return float(np.median([s.residual_state.residuals[ch].value for s in steps if s.residual_state.residuals[ch].valid]))


@lru_cache(maxsize=None)
def _fleet_ref() -> FleetResidualReference:
    """Fleet healthy-residual reference from reference-engine (fleet-0) flights
    at the test cruise points (Prompt 17b: the adaptation target)."""
    ops, res = [], {ch: [] for ch in DRIFT_CHANNELS}
    for k, (rpm, mp) in enumerate(CRUISES):
        for st in _flight(get_settings(), 100 + k, rpm, mp)[1]:
            op = st.expectation_result.operating_point
            ops.append([getattr(op, key) for key in OP_KEYS])
            for ch in DRIFT_CHANNELS:
                tv = st.residual_state.residuals.get(ch)
                res[ch].append(tv.value if tv is not None and tv.valid else np.nan)
    return FleetResidualReference.fit(np.asarray(ops), {ch: np.asarray(v, float) for ch, v in res.items()})


def _manager(tmp: Path, settings=None, fleet_reference="default"):
    s = settings or get_settings()
    ref = _fleet_ref() if fleet_reference == "default" else fleet_reference
    return A.AdaptationManager(A.BaselineStore(tmp / "store"), A.AdaptationLog(tmp / "store" / "log.jsonl"), s,
                               fleet_reference=ref)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def _rs(value: float) -> ResidualState:
    return ResidualState(operating_point=OP, residuals={"egt_cyl1": make_tagged(value, Provenance.DERIVED)})


def test_mean_shift_on_one_residual_raises_psi_above_0p25() -> None:
    rng = np.random.default_rng(1)
    mon = DriftMonitor()
    for v in rng.normal(0.0, 1.0, 300):
        mon.observe(_rs(float(v)), healthy=True)          # self-commissioned reference
    assert mon.reference is not None and mon.self_commissioned
    stable = DriftMonitor(reference=mon.reference)
    shifted = DriftMonitor(reference=mon.reference)
    for v in rng.normal(0.0, 1.0, 250):
        stable.observe(_rs(float(v)))
    for v in rng.normal(1.5, 1.0, 250):
        shifted.observe(_rs(float(v)))
    s_ok, s_bad = stable.evaluate(), shifted.evaluate()
    assert s_ok.channels["egt_cyl1"].psi < 0.10 and s_ok.overall_band == "STABLE"
    assert s_bad.channels["egt_cyl1"].psi > 0.25 and s_bad.overall_band == "SIGNIFICANT"


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def test_fault_flight_is_refused_and_the_log_records_why(tmp_path) -> None:
    s = get_settings()
    _, steps = _flight(s, 11, faults=[FaultScenarioConfig(fault_class=FaultClass.MISFIRE, severity=1.0,
                                                          onset_time_s=0.0, duration_s=1e4)])
    mgr = _manager(tmp_path)
    dec = mgr.process_flight("E1", steps, engine_hours=10.0)
    assert dec.eligible is False and dec.decision == "refused"
    last = mgr.log.entries()[-1]
    assert last["decision"] == "refused" and any("rule-based class MISFIRE" in r for r in last["reasons"])
    assert mgr.store.active("E1").version == "fleet-0"          # nothing learned
    assert mgr.log.verify()


def test_healthy_flight_is_eligible() -> None:
    _, steps = _flight(get_settings(), 12)
    ok, reasons = A.flight_eligibility(steps, get_settings())
    assert ok, reasons


# ---------------------------------------------------------------------------
# Baseline refit: engine-to-engine offset absorbed
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _offset_settings():
    s = get_settings().model_copy(deep=True)
    s.simulator.physics.egt_cylinder_offsets_k = [0.0, 8.0, 0.0, 0.0]
    return s


def _commission(tmp: Path, settings):
    mgr = _manager(tmp, settings)
    for k, (rpm, mp) in enumerate(CRUISES):
        dec = mgr.process_flight("E2", _flight(settings, 20 + k, rpm, mp)[1], engine_hours=5.0 * (k + 1))
    return mgr, dec


def test_engine_offset_of_8_k_is_absorbed_by_the_baseline_refit(tmp_path) -> None:
    s = _offset_settings()
    _, before = _flight(s, 30)
    _, ref_engine = _flight(get_settings(), 30)
    assert _median_res(before, "egt_cyl2") - _median_res(ref_engine, "egt_cyl2") == pytest.approx(8.0, abs=0.5)
    mgr, dec = _commission(tmp_path, s)
    assert dec.decision == "commissioned"
    base = mgr.store.active("E2")
    assert base.version == "E2-v1" and base.adapted
    _, after = _flight(s, 30, baseline=base)
    # Prompt 17b: the target is the fleet's healthy residual at this operating
    # point (here: the reference engine's), not zero (was |median| < 1.5 K)
    gap = _median_res(after, "egt_cyl2") - _median_res(ref_engine, "egt_cyl2")
    assert abs(gap) < 1.5, gap
    assert abs(_median_res(after, "egt_cyl2")) > 50.0      # the OI-5 fleet bias is kept, as the models expect
    assert after[-1].health_state.baseline_version == "E2-v1" and after[-1].health_state.adapted_baseline is True


def test_no_fleet_reference_means_no_commissioning(tmp_path) -> None:
    mgr = _manager(tmp_path, get_settings(), fleet_reference=None)
    dec = mgr.process_flight("E3", _flight(get_settings(), 21, *CRUISES[0])[1], engine_hours=5.0)
    assert dec.decision == "refused" and "no fleet healthy-residual reference" in dec.reasons[0]
    assert mgr.store.active("E3").version == "fleet-0"


def test_significant_drift_freezes_adaptation_and_raises_the_advisory(tmp_path) -> None:
    s = _offset_settings()
    mgr, _ = _commission(tmp_path, s)
    hot = s.model_copy(deep=True)
    hot.simulator.physics.egt_cylinder_offsets_k = [60.0, 68.0, 60.0, 60.0]   # large unexplained shift
    decisions = [mgr.process_flight("E2", _flight(hot, 40 + k, *CRUISES[k % 3], baseline=mgr.store.active("E2"))[1],
                                    engine_hours=20.0 + 5 * k) for k in range(3)]
    assert decisions[-1].drift_band == "SIGNIFICANT" and decisions[-1].decision == "frozen"
    assert "possible sensor drift or unmodelled condition" in decisions[-1].advisory
    assert mgr.store.active("E2").version == "E2-v1"         # nothing adapted


# ---------------------------------------------------------------------------
# Challenger, promotion gate, frozen artifacts
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _rows() -> "pd.DataFrame":
    rows = []
    for tid, fc in [(0, None), (1, None), (2, FaultClass.MISFIRE), (3, FaultClass.OIL_DEGRADATION),
                    (4, FaultClass.IMBALANCE), (5, None)]:
        faults = None if fc is None else [FaultScenarioConfig(fault_class=fc, severity=1.0, onset_time_s=0.0,
                                                              duration_s=1e4)]
        recs, _, _ = ScenarioRunner(seed=tid).run_scenario(duration_s=40.0, dt_s=1.0, fault_scenarios=faults)
        for st in PipelineReplayAdapter().process_sequence(recs):
            rows.append({"trajectory_id": tid, "injected_class": int(fc or 0), "label": int(fc or 0),
                         "split": "test" if tid == 5 else "train", **st.ml_features})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    df = _rows()
    train = df[df["split"] == "train"]
    A.train_challenger(train, train, None, root, "champ", 1, {"dataset_version": "unit"})
    worse = train.copy()
    worse["label"] = np.random.default_rng(0).permutation(worse["label"].to_numpy())   # deliberately worse
    A.train_challenger(worse, worse, None, root, "worse", 1, {"dataset_version": "unit"})
    return root


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_worse_challenger_is_rejected_by_the_promotion_gate(models, tmp_path) -> None:
    frozen_test = _rows()  # stands for the frozen Prompt 16 test split (read-only here)
    reg, log = A.ModelRegistry(models), A.AdaptationLog(tmp_path / "log.jsonl")
    reg.activate("champ")
    before = {p.name: _sha(p) for p in (models / "champ").iterdir()}
    assert A.evaluate_and_gate(reg, log, "champ", "worse", models, frozen_test) is False
    assert reg.active() == "champ"
    e = log.entries()[-1]
    assert e["decision"] == "rejected" and any("macro-F1" in r for r in e["reasons"])
    assert {p.name: _sha(p) for p in (models / "champ").iterdir()} == before   # frozen bundle untouched


def test_challenger_may_not_use_the_frozen_test_split(models) -> None:
    df = _rows()
    with pytest.raises(AssertionError, match="frozen test split"):
        A.train_challenger(df[df["split"] == "train"], df[df["split"] == "train"], df[df["split"] == "test"],
                           models, "never", 1, {})


# ---------------------------------------------------------------------------
# Rollback: identical predictions
# ---------------------------------------------------------------------------

def test_baseline_rollback_restores_identical_outputs(tmp_path) -> None:
    s = _offset_settings()
    mgr, _ = _commission(tmp_path, s)
    v1 = mgr.store.active("E2")
    _, run_v1 = _flight(s, 50, baseline=v1)
    v2 = A.EngineBaseline(**{**v1.__dict__, "version": "E2-v2", "parent": "E2-v1", "kind": "adapted",
                             "corrections": {k: [c[0] + 5.0, c[1], c[2]] for k, c in v1.corrections.items()}})
    mgr.store.add(v2)
    assert mgr.store.active("E2").version == "E2-v2"
    _, run_v2 = _flight(s, 50, baseline=mgr.store.active("E2"))
    assert _median_res(run_v2, "egt_cyl1") != pytest.approx(_median_res(run_v1, "egt_cyl1"))
    restored = mgr.rollback_baseline("E2")
    assert restored.version == "E2-v1"
    _, again = _flight(s, 50, baseline=restored)
    for a, b in zip(run_v1, again):
        assert {k: v.value for k, v in a.residual_state.residuals.items()} == \
               {k: v.value for k, v in b.residual_state.residuals.items()}
        assert a.health_state.health_index.value == b.health_state.health_index.value
    assert mgr.log.entries()[-1]["decision"] == "rollback" and mgr.log.verify()


def test_model_rollback_restores_identical_predictions(models) -> None:
    from src.l3_ml.anomaly_fault import FaultClassifier
    from src.l3_ml.ml_features import ML_FEATURES, feature_vector_from_row
    reg = A.ModelRegistry(models)
    reg.activate("champ")
    reg.activate("worse")
    s = get_settings().model_copy(deep=True)
    s.model_dir, s.ml.model_version = str(models), "registry"
    row = _rows().iloc[-1].to_dict()
    fv = feature_vector_from_row({n: row[n] for n in ML_FEATURES})
    with_worse = FaultClassifier(settings=s).classify_fault(fv)
    assert reg.rollback() == "champ"
    direct = s.model_copy(deep=True)
    direct.ml.model_version = "champ"
    a = FaultClassifier(settings=s).classify_fault(fv)
    b = FaultClassifier(settings=direct).classify_fault(fv)
    assert a.probabilities == b.probabilities and a.model_metadata.version == "champ"
    assert with_worse.model_metadata.version == "worse"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_ml_endpoints_and_guarded_rollback(tmp_path, monkeypatch, models) -> None:
    s = get_settings().model_copy(deep=True)
    s.model_dir = str(models)
    s.adaptation.store_dir = str(tmp_path / "store")
    A.ModelRegistry(models).activate("champ")
    A.ModelRegistry(models).activate("worse")
    recs, _, _ = ScenarioRunner(seed=3).run_scenario(duration_s=20.0, dt_s=1.0)
    engine = ReplayEngine("m")
    engine.load_scenario("m", recs, [])
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: s
    app.dependency_overrides[get_replay_engine] = lambda: engine
    app.dependency_overrides[get_running_pipeline] = lambda: RunningPipeline(PipelineReplayAdapter())
    monkeypatch.delenv(s.security.api_bearer_token_env_var, raising=False)
    with TestClient(app) as c:
        models_body = c.get("/api/v1/ml/models").json()
        assert "champ" in models_body["versions"] and models_body["registry_history"][-2:] == ["champ", "worse"]
        drift = c.get("/api/v1/ml/drift").json()
        assert drift["drift"]["reference_status"] in ("NO_REFERENCE", "BUILDING", "READY")
        assert drift["baseline_version"] == "fleet-0" and drift["adapted_baseline"] is False
        assert c.post("/api/v1/ml/rollback", json={"target": "model"}).status_code == 503   # fail closed
        monkeypatch.setenv(s.security.api_bearer_token_env_var, "s3cret")
        assert c.post("/api/v1/ml/rollback", json={"target": "model"},
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = c.post("/api/v1/ml/rollback", json={"target": "model"}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200 and ok.json()["to_version"] == "champ"
        health = c.get("/api/v1/engine/health").json()
        assert health["baseline_version"] == "fleet-0" and health["adapted_baseline"] is False
        rul = c.get("/api/v1/engine/rul").json()
        assert rul["interval_calibrated"] is False and rul["interval_note"]
