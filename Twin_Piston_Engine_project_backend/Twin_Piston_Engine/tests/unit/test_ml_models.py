"""
Prompt 16: feature hygiene, whole-trajectory splits, trained models present
vs absent, rule result kept beside ML, reduced confidence under dropout.

The model tests train small models on a few simulated runs in a temporary
model directory (no artifacts are committed); they check the plumbing, not
the metrics (those are in reports/ml_metrics.json).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

from src.core.config import get_settings  # noqa: E402
from src.core.provenance import FAULT_CLASS_COUNT, FaultClass, InferenceStatus  # noqa: E402
from src.l1_data.simulator.forward_simulator import (  # noqa: E402
    FaultScenarioConfig,
    ScenarioRunner,
    SimulationGroundTruth,
)
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter  # noqa: E402
from src.l3_ml import training as T  # noqa: E402
from src.l3_ml.anomaly_fault import AnomalyDetector, FaultClassifier  # noqa: E402
from src.l3_ml.ml_features import FEATURE_GROUPS, ML_FEATURES, feature_vector_from_row  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Feature hygiene
# ---------------------------------------------------------------------------

FORBIDDEN_PARTS = {"time", "timestamp", "sequence", "onset", "severity", "trajectory", "label", "ttf", "seed",
                   "injected", "ground", "gt", "truth", "split", "t", "s"}


def test_no_training_feature_comes_from_ground_truth_or_fault_config() -> None:
    gt_fields = {f.name for f in dataclasses.fields(SimulationGroundTruth)}
    cfg_fields = {f.name for f in dataclasses.fields(FaultScenarioConfig)}
    assert not set(ML_FEATURES) & (gt_fields | cfg_fields), set(ML_FEATURES) & (gt_fields | cfg_fields)
    for name in ML_FEATURES:  # whole name parts, e.g. "res_egt_cyl1" -> {res, egt, cyl1}
        assert not set(name.split("_")) & FORBIDDEN_PARTS, name
    assert set(FEATURE_GROUPS) == set(ML_FEATURES)
    # the dataset on disk (if generated) uses exactly this schema
    manifest = ROOT / "data" / "datasets" / "v2" / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["features"] == list(ML_FEATURES)


# ---------------------------------------------------------------------------
# Splits by whole trajectory
# ---------------------------------------------------------------------------

def test_splits_are_by_whole_trajectory_and_hold_out_the_operating_band() -> None:
    gen = _load_script("generate_dataset")
    plans = [gen.plan_trajectory(i, fc, 7, 240.0) for i, fc in
             enumerate([fc for fc in FaultClass for _ in range(8)])]
    split = gen.assign_splits(plans, 7)
    assert set(split) == {p["trajectory_id"] for p in plans}           # every trajectory, one split each
    for p in plans:
        in_band = gen.HOLDOUT_RPM_BAND[0] <= p["cruise_rpm"] <= gen.HOLDOUT_RPM_BAND[1]
        assert (split[p["trajectory_id"]] == "holdout_op") == in_band


def test_generated_dataset_has_no_trajectory_in_two_splits() -> None:
    path = ROOT / "data" / "datasets" / "v2" / "dataset.parquet"
    if not path.exists():
        pytest.skip("dataset v2 not generated (scripts/generate_dataset.py)")
    df = pd.read_parquet(path, columns=["trajectory_id", "split", "cruise_rpm"])
    per_traj = df.groupby("trajectory_id")["split"].nunique()
    assert int(per_traj.max()) == 1
    held = df[df["split"] == "holdout_op"]["cruise_rpm"]
    assert held.between(4400.0, 4600.0).all()
    assert not df[df["split"] != "holdout_op"]["cruise_rpm"].between(4400.0, 4600.0).any()


# ---------------------------------------------------------------------------
# Models present vs absent
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _rows() -> "pd.DataFrame":
    rows = []
    cases = [(0, None), (1, None), (2, FaultClass.MISFIRE), (3, FaultClass.OIL_DEGRADATION), (4, FaultClass.IMBALANCE)]
    for tid, fc in cases:
        faults = None if fc is None else [FaultScenarioConfig(fault_class=fc, severity=1.0, onset_time_s=0.0,
                                                              duration_s=1e4)]
        recs, _, _ = ScenarioRunner(seed=tid).run_scenario(duration_s=40.0, dt_s=1.0, fault_scenarios=faults,
                                                            operating_profile=[{"rpm": 4000.0, "map_pa": 110000.0}])
        for st in PipelineReplayAdapter().process_sequence(recs):
            rows.append({"trajectory_id": tid, "injected_class": int(fc or 0), "label": int(fc or 0),
                         **st.ml_features})
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _model_dir(tmp_root: str) -> Path:
    import joblib
    df = _rows()
    out = Path(tmp_root) / "vt"
    out.mkdir(parents=True, exist_ok=True)
    an = T.train_anomaly_detector(df, df, seed=1)
    joblib.dump(T.make_bundle("anomaly", "anomaly_detector", "vt", an["estimator"], ML_FEATURES,
                              {"threshold": an["threshold"], "group_missing_rate": an["group_missing_rate"]},
                              {"dataset_version": "unit"}, an["val_metrics"], "unit"),
                out / "anomaly_detector.joblib")
    fc = T.train_fault_classifier(df, df, seed=1)
    joblib.dump(T.make_bundle("classifier", "fault_classifier", "vt", fc["estimator"], ML_FEATURES,
                              {"importance": fc["importance"], "group_missing_rate": fc["group_missing_rate"]},
                              {"dataset_version": "unit"}, fc["val_metrics"], "unit"),
                out / "fault_classifier.joblib")
    return Path(tmp_root)


@pytest.fixture(scope="module")
def model_root(tmp_path_factory) -> Path:
    return _model_dir(str(tmp_path_factory.mktemp("models")))


def _settings(root: Path, version: str):
    s = get_settings().model_copy(deep=True)
    s.model_dir = str(root)
    s.ml.model_version = version
    return s


def _vector(k: int = -1):
    row = _rows().iloc[k].to_dict()
    return feature_vector_from_row({n: row[n] for n in ML_FEATURES})


def test_models_present_give_success_and_is_ml(model_root: Path) -> None:
    s = _settings(model_root, "vt")
    fv = _vector()
    an = AnomalyDetector(settings=s).detect_anomaly(fv)
    fr = FaultClassifier(settings=s).classify_fault(fv)
    assert an.status == InferenceStatus.SUCCESS and an.is_ml is True
    assert fr.status == InferenceStatus.SUCCESS and fr.is_ml is True
    assert len(fr.probabilities) == FAULT_CLASS_COUNT
    assert fr.evidence["top_features"] and fr.evidence["method"].startswith("LightGBM pred_contrib")
    assert fr.model_metadata.version == "vt" and fr.model_metadata.input_shape == [len(ML_FEATURES)]


def test_models_absent_keep_model_unavailable(model_root: Path) -> None:
    s = _settings(model_root, "missing_version")
    fv = _vector()
    assert AnomalyDetector(settings=s).detect_anomaly(fv).status == InferenceStatus.MODEL_UNAVAILABLE
    assert FaultClassifier(settings=s).classify_fault(fv).status == InferenceStatus.MODEL_UNAVAILABLE
    assert FaultClassifier().classify_fault(fv).status == InferenceStatus.MODEL_UNAVAILABLE  # default: none loaded


def test_bundle_metadata_carries_features_dataset_seeds_and_class_count(model_root: Path) -> None:
    import joblib
    b = joblib.load(model_root / "vt" / "fault_classifier.joblib")
    assert b["feature_names"] == list(ML_FEATURES) and b["provenance"]["class_count"] == FAULT_CLASS_COUNT == 14
    assert b["metadata"]["version"] == "vt" and b["feature_schema_version"] == "2.0.0"


def test_pipeline_keeps_the_rule_result_beside_ml(model_root: Path) -> None:
    s = _settings(model_root, "vt")
    recs, _, _ = ScenarioRunner(seed=9).run_scenario(duration_s=5.0, dt_s=1.0)
    last = PipelineReplayAdapter(s).process_sequence(recs)[-1]
    assert last.fault_result.is_ml is True and last.fault_result.status == InferenceStatus.SUCCESS
    assert last.rule_fault_result is not None and last.rule_fault_result.is_ml is False


def test_sensor_group_dropout_abstains_or_reduces_confidence(model_root: Path) -> None:
    """A removed sensor group never yields a confident answer: a group that was
    never fully missing in training -> MODEL_UNAVAILABLE (abstain); otherwise
    the confidence is scaled down by the importance-weighted coverage."""
    import joblib
    from src.l3_ml.ml_features import SENSOR_GROUPS
    from src.l3_ml.trained_models import OOD_GROUP_MISSING_RATE
    s = _settings(model_root, "vt")
    clf, det = FaultClassifier(settings=s), AnomalyDetector(settings=s)
    rates = joblib.load(model_root / "vt" / "fault_classifier.joblib")["group_missing_rate"]
    row = _rows().iloc[-1].to_dict()
    full = clf.classify_fault(feature_vector_from_row({n: row[n] for n in ML_FEATURES}))
    for group in SENSOR_GROUPS:
        dropped = {n: (math.nan if FEATURE_GROUPS[n] == group else row[n]) for n in ML_FEATURES}
        res = clf.classify_fault(feature_vector_from_row(dropped))
        if rates[group] < OOD_GROUP_MISSING_RATE:
            assert res.status == InferenceStatus.MODEL_UNAVAILABLE, group
            assert det.detect_anomaly(feature_vector_from_row(dropped)).status == InferenceStatus.MODEL_UNAVAILABLE
        else:
            assert res.status == InferenceStatus.SUCCESS and res.confidence <= full.confidence, group
