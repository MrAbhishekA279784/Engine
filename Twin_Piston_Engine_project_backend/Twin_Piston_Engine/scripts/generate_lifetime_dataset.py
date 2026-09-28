"""
Lifetime-scale fleet for RUL (method ported from aerotwin_ml/aerotwin/sim/fleet.py,
driven by THIS backend's simulator and L1 -> L2 -> L3 pipeline).

Each engine flies a life of MALE missions. It is either healthy (700 h) or has
ONE fault whose severity grows over tens to hundreds of engine hours
(src/l3_ml/lifetime_rul.FaultLife: onset 60-300 h, life 150-450 h, power law).
Every MONITOR_CADENCE_H engine hours one short monitoring window (20 s at 1 Hz)
is simulated at the flight condition of that moment (cruise / climb / descent,
altitude, ISA deviation) with the fault at its current severity, and run
through the real pipeline. One row per window: the ML feature row (schema
2.0.0, mean over the window's last 10 records), the pipeline health index
(for the baseline RUL estimator), and the ground truth (withheld from the
models): severity, failure time, RUL.

SENSOR_FAULT is not in the fleet: this simulator models it as a step dropout,
not a progressive drift, so it has no remaining life.

    uv run python scripts/generate_lifetime_dataset.py --version life1

life2 (Prompt 17b) adds healthy engine-to-engine variation and benign ageing
(`engine_variation`, physical simulator parameters, not output offsets) and
stores EVERY record of each window (the models score single records), with the
operating point as metadata columns (not model features). Only train and val
engines are simulated: the test engines stay held out for
scripts/evaluate_adaptation.py.

    uv run python scripts/generate_lifetime_dataset.py --version life2 --variation --per-record --skip-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.provenance import FaultClass  # noqa: E402
from src.l3_ml.lifetime_rul import MONITOR_CADENCE_H, FaultLife  # noqa: E402
from src.l3_ml.ml_features import ML_FEATURE_SCHEMA_VERSION, ML_FEATURES  # noqa: E402

WINDOW_S = 20.0
FEATURE_MEAN_RECORDS = 10
HEALTHY_HORIZON_H = 700.0
MAX_HORIZON_H = 1200.0
# simulator severity at functional failure (severity-1 of the lifetime process)
FAILURE_SEVERITY = {c: 1.0 for c in FaultClass}
FAILURE_SEVERITY.update({FaultClass.BEARING_WEAR: 0.10, FaultClass.INJECTOR_FAULT: 0.25})
# Healthy engine-to-engine variation and benign ageing (Prompt 17b). Project
# assumptions, VERIFY: no Rotax 915 iS fleet-spread or ageing data available.
VARIATION = {
    "ageing_egt_k_per_100h": (3.0, 6.0),          # uniform range
    "ageing_cht_k_per_100h": (1.0, 2.5),          # uniform range
    "egt_cylinder_offset_sd_k": 8.0,              # per cylinder, clipped at +/- 20 K
    "indicated_efficiency_rel": 0.02,             # multiplier 1 +/- 2 %
    "eta_v_peak_rel": 0.015,                      # multiplier 1 +/- 1.5 %
    "fmep_a_rel": 0.05,                           # multiplier 1 +/- 5 %
}
OP_COLUMNS = ("op_rpm", "op_map_pa", "op_altitude_m", "op_ambient_temp_k", "op_ambient_pressure_pa")
FLEET_CLASSES = [c for c in FaultClass if c not in (FaultClass.NOMINAL, FaultClass.SENSOR_FAULT)]
SUB_MODES = {FaultClass.COOLING_FAULT: ("radiator_blockage", "pump_degradation", "coolant_loss"),
             FaultClass.INJECTOR_FAULT: ("clog", "leak"),
             FaultClass.CHARGING_FAULT: ("setpoint_drift", "open_diode")}


def engine_variation(engine_idx: int, base_seed: int) -> dict:
    """Physical parameters of this engine (its own random stream, so the window
    plans are unchanged). The first two draws are the ageing rates."""
    rng = np.random.default_rng([base_seed, engine_idx, 0xA6E])
    v = VARIATION
    out = {"ageing_egt_k_per_100h": float(rng.uniform(*v["ageing_egt_k_per_100h"])),
           "ageing_cht_k_per_100h": float(rng.uniform(*v["ageing_cht_k_per_100h"]))}
    out["egt_cylinder_offsets_k"] = [float(x) for x in np.clip(rng.normal(0.0, v["egt_cylinder_offset_sd_k"], 4),
                                                               -20.0, 20.0)]
    out["indicated_efficiency_rel"] = float(rng.uniform(-1, 1) * v["indicated_efficiency_rel"])
    out["eta_v_peak_rel"] = float(rng.uniform(-1, 1) * v["eta_v_peak_rel"])
    out["fmep_a_rel"] = float(rng.uniform(-1, 1) * v["fmep_a_rel"])
    return out


def engine_settings(engine_idx: int, base_seed: int, base=None, variation: bool = True):
    """Settings for this engine (a deep copy); ageing_hours is set per window."""
    from src.core.config import get_settings
    s = (base or get_settings()).model_copy(deep=True)
    if variation:
        v, ph = engine_variation(engine_idx, base_seed), s.simulator.physics
        ph.ageing_egt_k_per_100h = v["ageing_egt_k_per_100h"]
        ph.ageing_cht_k_per_100h = v["ageing_cht_k_per_100h"]
        ph.egt_cylinder_offsets_k = v["egt_cylinder_offsets_k"]
        ph.indicated_efficiency *= 1.0 + v["indicated_efficiency_rel"]
        ph.eta_v_peak *= 1.0 + v["eta_v_peak_rel"]
        ph.fmep_barnes_moss_a_bar *= 1.0 + v["fmep_a_rel"]
    return s


def window_settings(engine_s, t_h: float):
    s = engine_s.model_copy(deep=True)
    s.simulator.physics.ageing_hours = float(t_h)
    return s


def operating_point_row(step) -> dict:
    er = getattr(step, "expectation_result", None)
    op = er.operating_point if er is not None else None
    if op is None:
        return dict.fromkeys(OP_COLUMNS, math.nan)
    return {"op_rpm": op.rpm, "op_map_pa": op.map_pressure_pa, "op_altitude_m": op.altitude_m,
            "op_ambient_temp_k": op.ambient_temp_k, "op_ambient_pressure_pa": op.ambient_pressure_pa}


def iter_engine_windows(engine_idx: int, cls_value: int, base_seed: int):
    """The engine's life as monitoring-window plans (all randomness here, in a
    fixed draw order): yields dicts with t_h, window_idx, phase, profile, ISA
    deviation, lifetime and simulator severity, fault list and window seed."""
    from src.l1_data.simulator.forward_simulator import FaultScenarioConfig

    fc = FaultClass(cls_value)
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, engine_idx]))
    life = FaultLife.sample(rng, fc != FaultClass.NOMINAL)
    horizon = HEALTHY_HORIZON_H if fc == FaultClass.NOMINAL else min(life.failure_h, MAX_HORIZON_H)
    sub_mode = str(rng.choice(SUB_MODES[fc])) if fc in SUB_MODES else None
    cylinder = int(rng.integers(1, 5))
    rpm_c, map_c = float(rng.uniform(4300.0, 5300.0)), float(rng.uniform(100000.0, 125000.0))
    t_h, k = MONITOR_CADENCE_H, 0
    while t_h < horizon:
        phase = str(rng.choice(["CRUISE", "CLIMB", "DESCENT"], p=[0.7, 0.15, 0.15]))
        isa = float(np.clip(rng.normal(5.0, 12.0), -20.0, 40.0))
        if phase == "CRUISE":
            prof = {"rpm": rpm_c + rng.normal(0.0, 60.0), "map_pa": map_c + rng.normal(0.0, 2000.0),
                    "altitude_m": float(rng.uniform(1500.0, 7600.0)), "airspeed_m_s": float(rng.uniform(45.0, 60.0))}
        elif phase == "CLIMB":
            prof = {"rpm": 5500.0, "map_pa": float(rng.uniform(130000.0, 140000.0)),
                    "altitude_m": float(rng.uniform(300.0, 4000.0)), "airspeed_m_s": 38.0}
        else:
            prof = {"rpm": 4000.0, "map_pa": float(rng.uniform(80000.0, 95000.0)),
                    "altitude_m": float(rng.uniform(1000.0, 5000.0)), "airspeed_m_s": 55.0}
        sev_life = life.severity(t_h)
        sim_sev = sev_life * FAILURE_SEVERITY[fc]
        faults = None
        if sim_sev > 0.0:
            # onset long before the window: the sub-models' own ramps are complete
            faults = [FaultScenarioConfig(fault_class=fc, severity=min(1.0, sim_sev), onset_time_s=-1e4,
                                          duration_s=1e7, sub_mode=sub_mode, affected_cylinder=cylinder)]
        yield {"t_h": t_h, "window_idx": k, "phase": phase, "profile": prof, "isa": isa, "sev_life": sev_life,
               "sim_severity": sim_sev, "faults": faults, "seed": int(rng.integers(0, 2**31 - 1)),
               "failure_h": life.failure_h, "onset_h": life.onset_h, "fault_class": int(fc)}
        t_h += MONITOR_CADENCE_H
        k += 1


def run_window(w: dict, settings=None, baseline=None):
    from src.l1_data.simulator.forward_simulator import ScenarioRunner
    from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter
    recs, _, _ = ScenarioRunner(settings=settings, seed=w["seed"]).run_scenario(
        duration_s=WINDOW_S, dt_s=1.0, operating_profile=[w["profile"]], fault_scenarios=w["faults"],
        isa_deviation_k=w["isa"])
    return PipelineReplayAdapter(settings, baseline=baseline).process_sequence(recs)


def simulate_engine(args) -> list[dict]:
    engine_idx, cls_value, base_seed = args[:3]
    variation, per_record = (args[3], args[4]) if len(args) > 3 else (False, False)
    eng_s = engine_settings(engine_idx, base_seed, variation=variation) if variation else None
    rows = []
    for w in iter_engine_windows(engine_idx, cls_value, base_seed):
        steps = run_window(w, window_settings(eng_s, w["t_h"]) if eng_s is not None else None)
        if per_record:
            f_h = w["failure_h"]
            for i, st in enumerate(steps):
                hi = st.health_state.health_index
                row = {"engine_id": engine_idx, "fault_class": w["fault_class"], "window_idx": w["window_idx"],
                       "record_idx": i, "t_h": w["t_h"], "phase": w["phase"], "isa_deviation_k": w["isa"],
                       "sev_life": w["sev_life"], "sim_severity": w["sim_severity"],
                       "failure_h": f_h if np.isfinite(f_h) else math.nan,
                       "gt_rul_h": (f_h - w["t_h"]) if np.isfinite(f_h) else math.nan,
                       "pipeline_hi": float(hi.value) if hi.valid else math.nan}
                row.update(operating_point_row(st))
                row.update({n: st.ml_features[n] for n in ML_FEATURES})
                rows.append(row)
            continue
        feats = np.array([[st.ml_features[n] for n in ML_FEATURES] for st in steps[-FEATURE_MEAN_RECORDS:]], float)
        with np.errstate(all="ignore"):
            fmean = np.nanmean(feats, axis=0) if feats.size else np.full(len(ML_FEATURES), np.nan)
        hi = steps[-1].health_state.health_index
        f_h = w["failure_h"]
        row = {"engine_id": engine_idx, "fault_class": w["fault_class"], "window_idx": w["window_idx"],
               "t_h": w["t_h"], "phase": w["phase"], "isa_deviation_k": w["isa"], "sev_life": w["sev_life"],
               "sim_severity": w["sim_severity"], "failure_h": f_h if np.isfinite(f_h) else math.nan,
               "gt_rul_h": (f_h - w["t_h"]) if np.isfinite(f_h) else math.nan,
               "pipeline_hi": float(hi.value) if hi.valid else math.nan}
        row.update(dict(zip(ML_FEATURES, fmean.tolist())))
        rows.append(row)
    return rows


def assign_splits(plan: list[tuple[int, int]], seed: int) -> dict[int, str]:
    """Whole-engine split, stratified by class (SRD-DAT-004)."""
    rng = np.random.default_rng(seed)
    by_cls: dict[int, list[int]] = {}
    for eid, c in plan:
        by_cls.setdefault(c, []).append(eid)
    out = {}
    for ids in by_cls.values():
        ids = list(rng.permutation(ids))
        n_tr, n_va = int(round(0.6 * len(ids))), int(round(0.2 * len(ids)))
        for i, e in enumerate(ids):
            out[int(e)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="life1")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--per-fault", type=int, default=20)
    ap.add_argument("--healthy", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets"))
    ap.add_argument("--variation", action="store_true", help="engine-to-engine variation and benign ageing")
    ap.add_argument("--per-record", action="store_true", help="one row per record instead of per window")
    ap.add_argument("--skip-test", action="store_true", help="do not simulate the test engines")
    args = ap.parse_args()
    import pandas as pd
    from scripts.generate_dataset import config_hash, git_sha

    plan = [(i, int(FaultClass.NOMINAL)) for i in range(args.healthy)]
    plan += [(len(plan) + j, int(c)) for j, c in enumerate(c for c in FLEET_CLASSES for _ in range(args.per_fault))]
    t0 = time.time()
    rows: list[dict] = []
    split = assign_splits(plan, args.seed)
    run_plan = [(e, c) for e, c in plan if not (args.skip_test and split[e] == "test")]
    jobs = [(e, c, args.seed, args.variation, args.per_record) for e, c in run_plan]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, part in enumerate(ex.map(simulate_engine, jobs, chunksize=1)):
            rows.extend(part)
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(run_plan)} engines, {len(rows)} rows, {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df["split"] = df["engine_id"].map(split)
    out = Path(args.out) / args.version
    out.mkdir(parents=True, exist_ok=True)
    path = out / "lifetime.parquet"
    df.to_parquet(path, index=False)
    manifest = {"version": args.version, "created_at": datetime.now(timezone.utc).isoformat(),
                "provenance": "SIMULATED", "seed": args.seed, "code_git_sha": git_sha(), "config_hash": config_hash(),
                "feature_schema_version": ML_FEATURE_SCHEMA_VERSION, "features": list(ML_FEATURES),
                "dataset_file": path.name, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "engines": len(plan), "engines_simulated": len(run_plan), "rows": int(len(df)),
                "windows": int(df.groupby(["engine_id", "window_idx"]).ngroups),
                "variation": VARIATION if args.variation else None, "per_record": args.per_record,
                "test_engines_simulated": not args.skip_test, "monitor_cadence_h": MONITOR_CADENCE_H,
                "window_s": WINDOW_S, "fleet_classes": [FaultClass(c).name for c in sorted({c for _, c in plan})],
                "split_by": "whole engine, stratified by class",
                "engine_split": {str(k): v for k, v in sorted(split.items())}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {path}: {len(df)} rows, {len(run_plan)} of {len(plan)} engines, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
