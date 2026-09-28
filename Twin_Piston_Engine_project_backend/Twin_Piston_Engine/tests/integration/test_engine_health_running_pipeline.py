"""
OI-20: /engine/health reports the real L3 HealthSupervision output of the
running pipeline (history windows kept across calls), never constants.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.api.v1.engine_health import HEALTH_COMPONENTS
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine, RunningPipeline
from tests.scientific.physics_audit_harness import SCENARIOS, simulate

BY_NAME = {s.name: s for s in SCENARIOS}


@lru_cache(maxsize=None)
def _records(name: str):
    return tuple(simulate(BY_NAME[name]))


class _Api:
    """Real app with its own replay engine and running pipeline."""

    def __init__(self, records) -> None:
        self.engine = ReplayEngine("oi20")
        self.engine.load_scenario("oi20", list(records), [])
        self.pipeline = RunningPipeline(PipelineReplayAdapter())
        self.app = create_app()
        self.app.dependency_overrides[get_replay_engine] = lambda: self.engine
        self.app.dependency_overrides[get_running_pipeline] = lambda: self.pipeline

    def health(self) -> dict:
        with TestClient(self.app) as client:
            r = client.get("/api/v1/engine/health")
        assert r.status_code == 200, r.text
        return r.json()


@lru_cache(maxsize=None)
def _health(name: str) -> dict:
    return _Api(_records(name)).health()


def test_component_health_is_derived_not_constant() -> None:
    """Fails if component_health is a literal: OIL_DEGRADATION must move lubrication."""
    nom, oil = _health("NOMINAL"), _health("OIL_DEGRADATION 1.0")
    assert nom["component_health"]["lubrication_health"] is not None
    assert oil["component_health"]["lubrication_health"] is not None
    assert oil["component_health"]["lubrication_health"] != nom["component_health"]["lubrication_health"]
    assert oil["component_health"]["lubrication_health"] < nom["component_health"]["lubrication_health"]
    assert oil["health_index"] < nom["health_index"]


@pytest.mark.parametrize("name", ["NOMINAL", "OIL_DEGRADATION 1.0"])
def test_api_matches_the_stateful_pipeline(name: str) -> None:
    api = _health(name)
    last = PipelineReplayAdapter().process_sequence(list(_records(name)))[-1]
    hs = last.health_state
    assert api["records_in_window"] == len(_records(name))
    assert api["health_index"] == pytest.approx(hs.health_index.value)
    assert api["trend"] == hs.trend
    assert api["health_rate_per_s"] == pytest.approx(hs.health_rate)
    assert api["coverage"] == pytest.approx(hs.evidence["coverage"]["data_coverage"])
    for c in HEALTH_COMPONENTS:
        assert api["component_health"][c] == (pytest.approx(hs.component_health[c]) if c in hs.component_health
                                              else None)


def test_component_without_data_is_null_with_reason() -> None:
    """CSV replay carries no electrical signals and no bursts: those components
    are null with a reason, never a default number."""
    api = _health("NOMINAL, CSV replay")
    assert set(api["component_health"]) == set(HEALTH_COMPONENTS)
    for c in ("electrical_health", "combustion_health"):
        assert api["component_health"][c] is None
        assert api["component_health_reasons"][c]
    for c, v in api["component_health"].items():
        assert (v is None) == (c in api["component_health_reasons"])
    assert api["coverage"] < 1.0


def test_running_pipeline_keeps_history_across_calls() -> None:
    """Records arriving between calls are stepped once, on the same state: the
    result equals one pass over the whole stream."""
    recs = list(_records("OIL_DEGRADATION 1.0"))
    api = _Api(recs[:60])
    first = api.health()
    assert first["records_in_window"] == 60
    api.engine.load_scenario("oi20", recs, [])  # stream grows; same source
    second = api.health()
    assert second["records_in_window"] == len(recs)
    assert second["health_index"] == pytest.approx(_health("OIL_DEGRADATION 1.0")["health_index"])
    assert second["trend"] == _health("OIL_DEGRADATION 1.0")["trend"]


def test_running_pipeline_restarts_on_a_different_stream() -> None:
    pipeline = RunningPipeline(PipelineReplayAdapter())
    pipeline.advance(list(_records("OIL_DEGRADATION 1.0")), source_key="a")
    pipeline.advance(list(_records("NOMINAL"))[:10], source_key="b")
    assert pipeline.records_processed == 10


def test_r_int_through_api_matches_pipeline() -> None:
    """Below cut-in the battery carries the load steps, so R_int becomes
    observable over the history window; a single-record API never could."""
    idle = [{"start_time_s": 0.0, "end_time_s": 1e6, "rpm": 1800.0, "map_pa": 60000.0}]
    recs, _, _ = ScenarioRunner(seed=7).run_scenario(duration_s=60.0, dt_s=0.1, operating_profile=idle)
    api = _Api(recs).health()
    elec = PipelineReplayAdapter().process_sequence(recs)[-1].electrical_state
    assert elec.battery_resistance_mohm.valid
    assert api["battery_resistance_mohm"] == pytest.approx(elec.battery_resistance_mohm.value)
