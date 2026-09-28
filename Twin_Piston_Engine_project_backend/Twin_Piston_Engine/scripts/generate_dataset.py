"""
Seeded trajectory dataset for the L3 models (Prompt 16; SRD-CON-008, SRD-DAT-004).

Each trajectory is one engine instance (its own seed) flying a random mission
(altitude, ISA deviation -10..+35 K, cruise, climbs, abrupt throttle steps,
gradual throttle ramps)
with either no fault or ONE fault class with a random onset time and
progression rate. Every record goes through the real L1 -> L2 -> L3 pipeline
(PipelineReplayAdapter) and one row is kept per record:

    features   ML feature schema 2.0.0 (src/l3_ml/ml_features.py): L2/L3
               outputs only, NaN where not derivable
    labels     label (FaultClass of the active fault, NOMINAL before onset),
               ttf_s (time until the fault reaches its full severity),
               gt_health_index (1 - effective severity)
    meta       trajectory_id, split, time_s, cruise_rpm, isa_deviation_k,
               rule_class (Prompt 15 rule fallback), rul_baseline_h
               (pipeline RULEstimator). Meta and labels are NOT features.

Split by WHOLE trajectory: trajectories whose cruise rpm lies in the held-out
operating band go to "holdout_op" (never trained on); the rest are split
60/20/20 per class into train/val/test.

Usage:
    uv run python scripts/generate_dataset.py --version v1 [--per-class 60] [--workers 16]
Output: data/datasets/<version>/dataset.parquet and manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import get_settings  # noqa: E402
from src.core.provenance import FaultClass  # noqa: E402
from src.l3_ml.ml_features import ML_FEATURE_SCHEMA_VERSION, ML_FEATURES  # noqa: E402

DT_S = 1.0
HOLDOUT_RPM_BAND = (4400.0, 4600.0)
SENSOR_CHANNELS = ("egt_cyl1_hot_uv", "map_counts", "oil_p_counts")

# target severity range per class (fraction of the simulator's severity-1 perturbation)
SEVERITY = {
    FaultClass.MISFIRE: (0.3, 1.0), FaultClass.DETONATION_KNOCK: (0.4, 1.0),
    FaultClass.EXHAUST_VALVE_LEAK: (0.4, 1.0), FaultClass.INTAKE_BOOST_LEAK: (0.4, 1.0),
    FaultClass.OIL_DEGRADATION: (0.4, 1.0), FaultClass.COOLING_FAULT: (0.5, 1.0),
    FaultClass.BEARING_WEAR: (0.02, 0.10), FaultClass.SENSOR_FAULT: (1.0, 1.0),
    FaultClass.INJECTOR_FAULT: (0.08, 0.25), FaultClass.FUEL_SYSTEM_FAULT: (0.5, 1.0),
    FaultClass.IMBALANCE: (0.3, 1.0), FaultClass.CHARGING_FAULT: (0.5, 1.0),
    FaultClass.BATTERY_DEGRADATION: (0.5, 1.0),
}
SUB_MODES = {
    FaultClass.COOLING_FAULT: ("radiator_blockage", "pump_degradation", "coolant_loss"),
    FaultClass.INJECTOR_FAULT: ("clog", "leak"),
    FaultClass.CHARGING_FAULT: ("setpoint_drift", "output_collapse", "open_diode"),
}


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def config_hash() -> str:
    return hashlib.sha256(get_settings().model_dump_json().encode()).hexdigest()


def plan_trajectory(traj_id: int, fault: FaultClass, base_seed: int, duration_s: float) -> dict:
    """Mission profile and fault for one trajectory (all randomness here, seeded)."""
    rng = np.random.default_rng([base_seed, traj_id])
    cruise_rpm = float(rng.uniform(3500.0, 5200.0))
    cruise_map = float(rng.uniform(90000.0, 125000.0))
    alt0 = float(rng.uniform(0.0, 2000.0))
    isa_dev = float(rng.uniform(-10.0, 35.0))
    wps = [{"t_s": 0.0, "rpm": cruise_rpm, "map_pa": cruise_map, "altitude_m": alt0, "airspeed_m_s": 50.0}]
    t = 0.0
    alt = alt0
    while t < duration_s:
        kind = rng.choice(["cruise", "climb", "step", "ramp"], p=[0.35, 0.2, 0.2, 0.25])
        if kind == "cruise":
            dur = float(rng.uniform(30.0, 90.0))
            wps.append({"t_s": t + dur, "rpm": cruise_rpm, "map_pa": cruise_map, "altitude_m": alt,
                        "airspeed_m_s": 50.0})
        elif kind == "climb":
            dur = float(rng.uniform(30.0, 90.0))
            rpm, mp = float(rng.uniform(5500.0, 5800.0)), float(rng.uniform(130000.0, 140000.0))
            wps.append({"t_s": t + 5.0, "rpm": rpm, "map_pa": mp, "altitude_m": alt, "airspeed_m_s": 38.0})
            alt = min(alt + float(rng.uniform(300.0, 1500.0)), 5000.0)
            wps.append({"t_s": t + dur, "rpm": rpm, "map_pa": mp, "altitude_m": alt, "airspeed_m_s": 38.0})
            wps.append({"t_s": t + dur + 5.0, "rpm": cruise_rpm, "map_pa": cruise_map, "altitude_m": alt,
                        "airspeed_m_s": 50.0})
            dur += 5.0
        elif kind == "ramp":
            # gradual power change (v2): rpm and MAP move linearly to a new
            # cruise setting over 20-90 s; covers slow MAP declines and rises
            # at constant rpm, which in v1 only boost-leak trajectories showed
            dur = float(rng.uniform(20.0, 90.0))
            cruise_rpm = float(np.clip(cruise_rpm + rng.uniform(-400.0, 400.0), 3000.0, 5500.0))
            cruise_map = float(np.clip(cruise_map + rng.uniform(-35000.0, 20000.0), 70000.0, 135000.0))
            wps.append({"t_s": t + dur, "rpm": cruise_rpm, "map_pa": cruise_map, "altitude_m": alt,
                        "airspeed_m_s": 50.0})
        else:  # rapid throttle transient: abrupt step and hold, then back
            dur = float(rng.uniform(10.0, 30.0))
            rpm = float(np.clip(cruise_rpm + rng.uniform(-800.0, 800.0), 2500.0, 5800.0))
            mp = float(np.clip(cruise_map + rng.uniform(-20000.0, 20000.0), 70000.0, 140000.0))
            wps.append({"t_s": t + 1.0, "rpm": rpm, "map_pa": mp, "altitude_m": alt, "airspeed_m_s": 50.0})
            wps.append({"t_s": t + dur, "rpm": rpm, "map_pa": mp, "altitude_m": alt, "airspeed_m_s": 50.0})
            wps.append({"t_s": t + dur + 1.0, "rpm": cruise_rpm, "map_pa": cruise_map, "altitude_m": alt,
                        "airspeed_m_s": 50.0})
            dur += 1.0
        t = wps[-1]["t_s"]

    plan = {"trajectory_id": traj_id, "seed": int(rng.integers(0, 2**31 - 1)), "fault": int(fault),
            "cruise_rpm": cruise_rpm, "cruise_map_pa": cruise_map, "isa_deviation_k": isa_dev,
            "waypoints": wps, "duration_s": duration_s}
    if fault != FaultClass.NOMINAL:
        lo, hi = SEVERITY[fault]
        target = float(rng.uniform(lo, hi))
        onset = float(rng.uniform(30.0, 150.0))
        ramp = 0.0 if fault == FaultClass.SENSOR_FAULT else float(rng.uniform(20.0, 180.0))
        plan.update(severity=target, onset_s=onset, progression_rate=(target / ramp if ramp > 0.0 else 0.0),
                    sub_mode=(str(rng.choice(SUB_MODES[fault])) if fault in SUB_MODES else None),
                    cylinder=int(rng.integers(1, 5)),
                    channel=(str(rng.choice(SENSOR_CHANNELS)) if fault == FaultClass.SENSOR_FAULT else None))
    return plan


def run_trajectory(plan: dict) -> list[dict]:
    from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
    from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, mission_profile_from_waypoints

    fault = FaultClass(plan["fault"])
    faults = None
    if fault != FaultClass.NOMINAL:
        faults = [FaultScenarioConfig(
            fault_class=fault, severity=plan["severity"], onset_time_s=plan["onset_s"], duration_s=1e6,
            progression_rate_per_s=plan["progression_rate"], sub_mode=plan["sub_mode"],
            affected_cylinder=plan["cylinder"], affected_channel=plan["channel"])]
    profile = mission_profile_from_waypoints(plan["waypoints"], step_s=1.0)
    recs, gts, _ = ScenarioRunner(seed=plan["seed"]).run_scenario(
        duration_s=plan["duration_s"], dt_s=DT_S, operating_profile=profile, fault_scenarios=faults,
        isa_deviation_k=plan["isa_deviation_k"])
    steps = PipelineReplayAdapter().process_sequence(recs)
    rows = []
    for k, (gt, st) in enumerate(zip(gts, steps)):
        sev = faults[0].severity_at(gt.time_s) if faults else 0.0
        label = int(fault) if sev > 0.0 else int(FaultClass.NOMINAL)
        if faults and plan["progression_rate"] > 0.0:
            t_fail = plan["onset_s"] + plan["severity"] / plan["progression_rate"]
            ttf = max(0.0, t_fail - gt.time_s)
        elif faults:
            ttf = 0.0 if gt.time_s >= plan["onset_s"] else plan["onset_s"] - gt.time_s
        else:
            ttf = math.nan
        rul = st.rul_state
        row = {
            "trajectory_id": plan["trajectory_id"], "trajectory_seed": plan["seed"], "injected_class": plan["fault"],
            "time_s": gt.time_s, "label": label, "severity_gt": sev, "ttf_s": ttf,
            "gt_health_index": 1.0 - (sev / max(plan.get("severity", 1.0), 1e-9) if faults else 0.0),
            "cruise_rpm": plan["cruise_rpm"], "isa_deviation_k": plan["isa_deviation_k"],
            "rule_class": int(st.rule_fault_result.class_id),
            "rul_baseline_h": (float(rul.hours_remaining) if rul is not None and rul.status.value == "SUCCESS"
                               else math.nan),
        }
        row.update(st.ml_features)
        rows.append(row)
    return rows


def assign_splits(plans: list[dict], base_seed: int) -> dict[int, str]:
    rng = np.random.default_rng([base_seed, 0x5917])
    split: dict[int, str] = {}
    by_class: dict[int, list[int]] = {}
    for p in plans:
        if HOLDOUT_RPM_BAND[0] <= p["cruise_rpm"] <= HOLDOUT_RPM_BAND[1]:
            split[p["trajectory_id"]] = "holdout_op"
        else:
            by_class.setdefault(p["fault"], []).append(p["trajectory_id"])
    for ids in by_class.values():
        ids = list(rng.permutation(ids))
        n = len(ids)
        n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
        for i, tid in enumerate(ids):
            split[int(tid)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--nominal-mult", type=int, default=3)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets"))
    args = ap.parse_args()

    import pandas as pd

    classes = [FaultClass.NOMINAL] * (args.per_class * args.nominal_mult) + [
        fc for fc in FaultClass if fc != FaultClass.NOMINAL for _ in range(args.per_class)]
    plans = [plan_trajectory(i, fc, args.seed, args.duration) for i, fc in enumerate(classes)]
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, part in enumerate(ex.map(run_trajectory, plans, chunksize=4)):
            rows.extend(part)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(plans)} trajectories, {time.time() - t0:.0f} s", flush=True)
    split = assign_splits(plans, args.seed)
    df = pd.DataFrame(rows)
    df["split"] = df["trajectory_id"].map(split)
    out = Path(args.out) / args.version
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dataset.parquet"
    df.to_parquet(path, index=False)
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": "SIMULATED",
        "seed": args.seed,
        "code_git_sha": git_sha(),
        "config_hash": config_hash(),
        "feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "features": list(ML_FEATURES),
        "dataset_file": path.name,
        "dataset_sha256": data_hash,
        "rows": int(len(df)),
        "trajectories": len(plans),
        "trajectories_per_class": {FaultClass(c).name: classes.count(FaultClass(c)) for c in set(classes)},
        "duration_s": args.duration,
        "dt_s": DT_S,
        "holdout_rpm_band": list(HOLDOUT_RPM_BAND),
        "split_by": "whole trajectory (SRD-DAT-004); cruise rpm band held out as holdout_op",
        "splits": {sp: int(sum(1 for v in split.values() if v == sp)) for sp in sorted(set(split.values()))},
        "trajectory_split": {str(k): v for k, v in sorted(split.items())},
        "plans": plans,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {path} ({len(df)} rows, {len(plans)} trajectories, {time.time() - t0:.0f} s); sha256 {data_hash[:12]}")


if __name__ == "__main__":
    main()
