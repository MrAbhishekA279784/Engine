"""
Prompt 17: lifetime RUL (method ported from aerotwin_ml) and the
"unvalidated" marking of RUL values that were not validated at lifetime scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.dependencies import get_replay_engine, get_running_pipeline  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.l1_data.simulator.forward_simulator import ScenarioRunner  # noqa: E402
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine, RunningPipeline  # noqa: E402
from src.l3_ml import lifetime_rul as L  # noqa: E402
from src.l3_ml.ml_features import ML_FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NOT_PRODUCED_BY_THIS_BACKEND = {"throttle_pct", "airspeed_m_s", "airspeed", "elec_load_a", "phase", "t_h",
                                "engine_hours", "hours_since_degradation"}


def test_rul_inputs_exclude_hours_and_foreign_signals() -> None:
    assert not set(L.HISTORY_COLS) & NOT_PRODUCED_BY_THIS_BACKEND
    assert not set(ML_FEATURES) & NOT_PRODUCED_BY_THIS_BACKEND
    h = L.history_features(np.linspace(0.0, 0.5, 30))
    assert list(h.columns) == L.HISTORY_COLS  # window-indexed history only


def test_fault_life_is_a_power_law_to_failure() -> None:
    life = L.FaultLife(onset_h=100.0, life_h=200.0, exponent=2.0)
    assert life.severity(50.0) == 0.0 and life.severity(100.0) == 0.0
    assert life.severity(200.0) == pytest.approx(0.25)
    assert life.severity(300.0) == pytest.approx(1.0) and life.failure_h == 300.0
    assert L.FaultLife().severity(1e6) == 0.0  # healthy


def test_band_metrics_mae_and_coverage() -> None:
    true = np.array([10.0, 40.0, 100.0, 500.0])
    pred = np.array([20.0, 40.0, 90.0, 390.0])
    m = L.band_metrics(true, pred, pred - 5.0, pred + 15.0)
    assert m["0-50h"]["n"] == 2 and m["0-50h"]["mae_h"] == pytest.approx(5.0)
    assert m["0-50h"]["coverage_90"] == pytest.approx(0.5)      # 10 not in [15, 35]
    assert m["150-400h"]["n"] == 1 and m["150-400h"]["mae_h"] == pytest.approx(10.0)  # true capped at 400


class _SevStub:
    """Severity model stand-in: reads the first feature."""

    def predict(self, x):
        return np.clip(np.nan_to_num(np.asarray(x)[:, 0]), 0.0, 1.0)


def _stub_bundle() -> dict:
    rng = np.random.default_rng(0)
    rows, ruls = [], []
    for _ in range(40):
        n = 80
        sev = np.clip(np.linspace(0.0, 1.0, n) ** 2 + rng.normal(0, 0.01, n), 0, 1)
        h = L.history_features(sev)
        rows.append(h)
        ruls.append((n - 1 - np.arange(n)) * L.MONITOR_CADENCE_H)
    h = pd.concat(rows, ignore_index=True)
    rul = L.RULModel(0).fit(h, np.concatenate(ruls))
    rul.calibrate(h, np.concatenate(ruls))
    return {"kind": "lifetime_rul", "severity_model": _SevStub(), "rul_model": rul, "features": list(ML_FEATURES),
            "monitor_cadence_h": L.MONITOR_CADENCE_H}


def test_lifetime_estimator_samples_at_cadence_and_validates_only_estimates() -> None:
    est = L.LifetimeRULEstimator(_stub_bundle())
    name = ML_FEATURES[0]
    statuses = []
    for k in range(400):
        t_h = k * 1.0                                   # one record per engine hour
        sev = min(1.0, (t_h / 400.0) ** 2)
        r = est.observe(t_h, {name: sev})
        statuses.append(r["status"])
        assert r["validated"] == (r["status"] == "ESTIMATED")
        if r["status"] != "ESTIMATED":
            assert math.isnan(r["rul_h"])
    assert est._sev.__len__() == 80                     # 400 h / 5 h cadence
    assert statuses[0] == "NO_DEGRADATION" and "INDETERMINATE" in statuses and statuses[-1] == "ESTIMATED"


def test_pipeline_uses_lifetime_rul_only_when_configured_with_engine_hours(tmp_path) -> None:
    import joblib
    (tmp_path / "lt").mkdir()
    joblib.dump(_stub_bundle(), tmp_path / "lt" / "rul_lifetime.joblib")
    s = get_settings().model_copy(deep=True)
    s.model_dir, s.ml.rul_model_version = str(tmp_path), "lt"
    recs, _, _ = ScenarioRunner(seed=3).run_scenario(duration_s=5.0, dt_s=1.0)
    # engine hours not configured -> channel invalid -> unvalidated baseline stays
    last = PipelineReplayAdapter(s).process_sequence(recs)[-1]
    assert last.rul_state.validated is False and last.rul_state.model_name != "lifetime_rul"
    s.engine.engine_hours_at_install = 250.0
    last = PipelineReplayAdapter(s).process_sequence(recs)[-1]
    assert last.rul_state.model_name == "lifetime_rul"
    assert last.rul_state.validated is False           # one window of history: not ESTIMATED
    assert last.rul_state.evidence["lifetime_status"] in ("NO_DEGRADATION", "INDETERMINATE")


def test_api_marks_baseline_rul_unvalidated() -> None:
    recs, _, _ = ScenarioRunner(seed=4).run_scenario(duration_s=30.0, dt_s=1.0)
    engine = ReplayEngine("r")
    engine.load_scenario("r", recs, [])
    pipeline = RunningPipeline(PipelineReplayAdapter())
    app = create_app()
    app.dependency_overrides[get_replay_engine] = lambda: engine
    app.dependency_overrides[get_running_pipeline] = lambda: pipeline
    with TestClient(app) as client:
        rul = client.get("/api/v1/engine/rul").json()
        risk = client.get("/api/v1/mission/risk").json()
    assert rul["validated"] is False and rul["status"] != "SUCCESS" and "UNVALIDATED" in rul["validation_note"]
    assert risk["rul_validated"] is False


def test_lifetime_fleet_split_by_whole_engine() -> None:
    path = ROOT / "data" / "datasets" / "life1" / "lifetime.parquet"
    if not path.exists():
        pytest.skip("lifetime fleet not generated (scripts/generate_lifetime_dataset.py)")
    df = pd.read_parquet(path, columns=["engine_id", "split"])
    assert int(df.groupby("engine_id")["split"].nunique().max()) == 1
