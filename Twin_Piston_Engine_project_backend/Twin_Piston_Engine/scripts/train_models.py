"""
Train the L3 models on a generated dataset (Prompt 16). Uses the train and
val splits only; test and holdout_op are never read here.

    uv run python scripts/train_models.py --dataset v1 --version v1

v3 (Prompt 17b) adds the lifetime fleet's TRAIN engines (per-record rows,
engine-to-engine variation and benign ageing) to TRAIN and its VAL engines to
VAL; lifetime test engines are not in that dataset at all:

    uv run python scripts/train_models.py --dataset v2 --extra-dataset life2 --version v3 --no-rul

and fits the fleet healthy-residual reference (src/l3_ml/fleet_reference.py)
on the healthy lifetime TRAIN records.

Writes models/<version>/anomaly_detector.joblib, fault_classifier.joblib
(bundles with their metadata: feature list, dataset hash, seeds, class count,
version, validation metrics) and model_card.json. Enable them in the pipeline
with ml.model_version = "<version>".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.provenance import FAULT_CLASS_COUNT  # noqa: E402
from src.l3_ml import training as T  # noqa: E402
from src.l3_ml.ml_features import ML_FEATURES  # noqa: E402


def lifetime_rows(life):
    """Lifetime per-record rows in the Prompt 16 row format. Label: the engine's
    fault class once its severity is above zero, NOMINAL before (the same rule
    as the Prompt 16 trajectories: labelled from onset). The whole engine is
    the grouping unit (trajectory_id offset so it cannot collide)."""
    life = life.copy()
    lab = life["fault_class"].where(life["sim_severity"] > 0.0, 0).astype(int)
    life["label"] = lab
    life["injected_class"] = lab
    life["trajectory_id"] = 1_000_000 + life["engine_id"].astype(int)
    life["time_s"] = life["t_h"] * 3600.0 + life["record_idx"]
    return life


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v1")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "datasets"))
    ap.add_argument("--model-root", default=str(ROOT / "models"))
    ap.add_argument("--extra-dataset", default=None, help="lifetime fleet dataset (per-record rows)")
    ap.add_argument("--no-rul", action="store_true", help="skip the short-trajectory RUL regressor")
    args = ap.parse_args()

    import joblib
    import pandas as pd

    ds_dir = Path(args.data_root) / args.dataset
    manifest = json.loads((ds_dir / "manifest.json").read_text())
    assert manifest["features"] == list(ML_FEATURES), "dataset feature schema differs from the code"
    df = pd.read_parquet(ds_dir / manifest["dataset_file"])
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    provenance = {"dataset_version": manifest["version"], "dataset_sha256": manifest["dataset_sha256"],
                  "dataset_seed": manifest["seed"], "training_seed": args.seed,
                  "code_git_sha_dataset": manifest["code_git_sha"], "config_hash": manifest["config_hash"],
                  "data_provenance": "SIMULATED"}
    life_train = None
    if args.extra_dataset:
        lx_dir = Path(args.data_root) / args.extra_dataset
        lm = json.loads((lx_dir / "manifest.json").read_text())
        assert lm["features"] == list(ML_FEATURES) and lm.get("per_record"), "lifetime dataset must be per record"
        life = pd.read_parquet(lx_dir / lm["dataset_file"])
        assert not (life["split"] == "test").any(), "lifetime test engines must not be in the training dataset"
        life = lifetime_rows(life)
        life_train = life[life["split"] == "train"]
        train = pd.concat([train, life_train], ignore_index=True)
        val = pd.concat([val, life[life["split"] == "val"]], ignore_index=True)
        provenance.update({"extra_dataset_version": lm["version"], "extra_dataset_sha256": lm["dataset_sha256"],
                           "extra_dataset_seed": lm["seed"], "extra_split_by": lm["split_by"]})
    print(f"train {len(train)} rows, val {len(val)} rows")

    out = Path(args.model_root) / args.version
    out.mkdir(parents=True, exist_ok=True)

    an = T.train_anomaly_detector(train, val, args.seed)
    joblib.dump(T.make_bundle("anomaly", "anomaly_detector", args.version, an["estimator"], ML_FEATURES,
                              {"threshold": an["threshold"], "group_missing_rate": an["group_missing_rate"]},
                              provenance, an["val_metrics"],
                              "IsolationForest on NOMINAL trajectories; threshold at 5 % FPR on validation"),
                out / "anomaly_detector.joblib")
    print("anomaly", an["val_metrics"], "threshold", round(an["threshold"], 4))

    fc = T.train_fault_classifier(train, val, args.seed)
    joblib.dump(T.make_bundle("classifier", "fault_classifier", args.version, fc["estimator"], ML_FEATURES,
                              {"importance": fc["importance"], "group_missing_rate": fc["group_missing_rate"]},
                              provenance, fc["val_metrics"],
                              f"LightGBM multi-class over {FAULT_CLASS_COUNT} classes, balanced class weights"),
                out / "fault_classifier.joblib")
    print("classifier", fc["val_metrics"])

    if args.no_rul:
        rul = {"decision": "not trained (lifetime RUL model is separate, scripts/train_rul.py)",
               "val_ml": None, "val_baseline": None}
    else:
        rul = T.train_rul_regressor(train, val, args.seed)
        if rul["adopted"]:
            joblib.dump({"estimator": rul["estimator"], "window_s": T.RUL_WINDOW_S}, out / "rul_regressor.joblib")
    print("rul", rul["decision"], "ml", rul["val_ml"], "baseline", rul["val_baseline"])

    fleet_ref_stats = None
    if life_train is not None:
        from src.l3_ml.drift_monitor import DRIFT_CHANNELS
        from src.l3_ml.fleet_reference import FLEET_REFERENCE_FILE, FleetResidualReference
        healthy = life_train[life_train["sim_severity"] <= 0.0]
        ops = healthy[["op_rpm", "op_map_pa", "op_altitude_m", "op_ambient_temp_k",
                       "op_ambient_pressure_pa"]].to_numpy(float)
        ref = FleetResidualReference.fit(ops, {ch: healthy[f"res_{ch}"].to_numpy(float) for ch in DRIFT_CHANNELS},
                                         meta={"version": args.version, "fitted_on": "healthy lifetime TRAIN records",
                                               **{k: provenance[k] for k in provenance if k.startswith("extra_")}})
        ref.save(out / FLEET_REFERENCE_FILE)
        fleet_ref_stats = ref.fit_stats
        print("fleet reference", {k: (round(v["residual_sd"], 4), round(v["sd_after_reference"], 4))
                                  for k, v in ref.fit_stats.items()})

    top = sorted(fc["importance"].items(), key=lambda kv: -kv[1])[:15]
    card = {"version": args.version, "provenance": provenance, "class_count": FAULT_CLASS_COUNT,
            "features": list(ML_FEATURES),
            "anomaly_detector": {"threshold": an["threshold"], "val": an["val_metrics"]},
            "fault_classifier": {"val": fc["val_metrics"], "top_importance": top},
            "rul": {"decision": rul["decision"], "val_ml": rul["val_ml"], "val_baseline": rul["val_baseline"]},
            "fleet_residual_reference": fleet_ref_stats}
    (out / "model_card.json").write_text(json.dumps(card, indent=1, default=str))
    print(f"wrote models to {out}")


if __name__ == "__main__":
    main()
