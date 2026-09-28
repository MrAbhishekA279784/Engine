"""
Presentation headline numbers (Prompt 18). ALL NUMBERS ARE SIMULATED.

The 60 held-out TEST engines of the lifetime fleet (engine split of life2; never
in any training dataset), with life2 engine-to-engine variation and benign
ageing, are flown once through L1 -> L2 -> L3 with the v3 models loaded and no
adaptation. Per monitoring window (20 s at 1 Hz) three alarm sources are kept
separately, so the rule-based classifier and v3 are scored on identical data:

    rule      rule-based class not NOMINAL at the window's last record
    v3        v3 anomaly alarm on >= half of the window's records, or the v3
              ML class not NOMINAL with confidence > 0.7 at the last record
    combined  rule or v3

    healthy-window false-alarm rate  share of healthy-engine windows with an alarm
    detection                        two consecutive alarm windows after onset
    lead time                        failure time - detection time (median)

RUL: the adopted lifetime RUL bundle (models/life1/rul_lifetime.joblib, trained
on life1, i.e. WITHOUT engine variation) on the same windows (feature mean of
the last 10 records), errors per true-RUL band on ESTIMATED windows.

    uv run python scripts/headline_metrics.py
Writes reports/headline_metrics.json and .md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_lifetime_dataset import (  # noqa: E402
    FEATURE_MEAN_RECORDS, VARIATION, engine_settings, iter_engine_windows, run_window, window_settings)
from src.core.config import get_settings  # noqa: E402
from src.core.provenance import FaultClass, InferenceStatus  # noqa: E402
from src.l3_ml.ml_features import ML_FEATURES  # noqa: E402

CONF = 0.70
SOURCES = ("rule", "v3", "combined")


def run_engine(args) -> dict:
    engine_id, cls, seed, model_version, model_dir = args
    s = get_settings().model_copy(deep=True)
    s.model_dir, s.ml.model_version = model_dir, model_version
    s = engine_settings(engine_id, seed, s, variation=True)
    wins = []
    onset = failure = math.nan
    for w in iter_engine_windows(engine_id, cls, seed):
        steps = run_window(w, window_settings(s, w["t_h"]))
        an = np.mean([st.anomaly_result.status == InferenceStatus.SUCCESS and st.anomaly_result.is_anomaly
                      for st in steps]) >= 0.5
        last = steps[-1]
        rule = last.rule_fault_result.predicted_class != FaultClass.NOMINAL
        ml = last.fault_result
        ml_alarm = (ml.status == InferenceStatus.SUCCESS and ml.predicted_class != FaultClass.NOMINAL
                    and (ml.confidence or 0.0) > CONF)
        feats = np.array([[st.ml_features[n] for n in ML_FEATURES] for st in steps[-FEATURE_MEAN_RECORDS:]], float)
        with np.errstate(all="ignore"):
            fmean = np.nanmean(feats, axis=0)
        wins.append({"t_h": w["t_h"], "window_idx": w["window_idx"], "rule": bool(rule), "v3": bool(an or ml_alarm),
                     "combined": bool(rule or an or ml_alarm), "rule_class": last.rule_fault_result.class_name,
                     "features": fmean.tolist()})
        onset, failure = w["onset_h"], w["failure_h"]
    return {"engine_id": engine_id, "fault_class": cls, "onset_h": onset if math.isfinite(onset) else None,
            "failure_h": failure if math.isfinite(failure) else None, "windows": wins}


def detection(results: list[dict], src: str) -> dict:
    healthy = [w[src] for r in results if r["fault_class"] == 0 for w in r["windows"]]
    leads, missed, pre = [], [], []
    for r in results:
        if r["fault_class"] == 0:
            continue
        ws = r["windows"]
        pre += [w[src] for w in ws if w["t_h"] < r["onset_h"]]
        det = next((ws[i]["t_h"] for i in range(1, len(ws)) if ws[i - 1]["t_h"] >= r["onset_h"]
                    and ws[i][src] and ws[i - 1][src]), None)
        if det is None:
            missed.append(FaultClass(r["fault_class"]).name)
        else:
            leads.append(r["failure_h"] - det)
    n_f = sum(r["fault_class"] != 0 for r in results)
    return {"healthy_window_false_alarm_rate": float(np.mean(healthy)),
            "detection_rate": (n_f - len(missed)) / n_f, "detected": n_f - len(missed), "faulty_engines": n_f,
            "median_lead_h": float(np.median(leads)) if leads else None,
            "pre_onset_window_alarm_rate": float(np.mean(pre)) if pre else None,
            "missed": sorted(missed)}


def rul_errors(results: list[dict], bundle_path: Path) -> dict:
    import joblib
    import pandas as pd
    from src.l3_ml import lifetime_rul as L
    b = joblib.load(bundle_path)
    assert b["features"] == list(ML_FEATURES)
    rows = []
    for r in results:
        x = np.array([w["features"] for w in r["windows"]], float)
        sev = b["severity_model"].predict(x)
        h = L.history_features(sev)
        p = b["rul_model"].predict(h)
        for i, w in enumerate(r["windows"]):
            gt = (r["failure_h"] - w["t_h"]) if r["failure_h"] is not None else math.nan
            rows.append({"engine_id": r["engine_id"], "fault_class": r["fault_class"], "gt_rul_h": gt,
                         "status": p["status"][i], "rul_h": p["rul_h"][i], "lo": p["rul_lo_h"][i],
                         "hi": p["rul_hi_h"][i]})
    d = pd.DataFrame(rows)
    f = d[np.isfinite(d["gt_rul_h"])]
    est = f[f["status"] == "ESTIMATED"]
    t = np.minimum(est["gt_rul_h"].to_numpy(), L.RUL_CAP_H)
    healthy = d[d["fault_class"] == 0]
    return {"bundle": bundle_path.relative_to(ROOT).as_posix(), "trained_on": b["provenance"]["dataset"],
            "bands": L.band_metrics(est["gt_rul_h"].to_numpy(), est["rul_h"].to_numpy(), est["lo"].to_numpy(),
                                    est["hi"].to_numpy()),
            "coverage_90_overall": float(np.mean((t >= est["lo"]) & (t <= est["hi"]))) if len(est) else None,
            "faulty_window_status_share": f["status"].value_counts(normalize=True).round(4).to_dict(),
            "healthy_windows_estimated_share": float(np.mean(healthy["status"] == "ESTIMATED"))}


def markdown(r: dict) -> str:
    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"
    lines = ["# Headline numbers (ALL SIMULATED)", "",
             f"{r['healthy_engines']} healthy and {r['faulty_engines']} faulty held-out test engines of the lifetime "
             f"fleet (life2 engine split), engine-to-engine variation and benign ageing ({r['variation']}). "
             "Every number below is from the forward simulator: **simulated**.", "",
             "| (simulated) | rule-based alone | v3 alone | rules + v3 |", "|---|---|---|---|"]
    for key, label, nd in (("healthy_window_false_alarm_rate", "Healthy-window false-alarm rate", 3),
                           ("detection_rate", "Detection rate", 3), ("median_lead_h", "Median lead time before failure (h)", 1),
                           ("pre_onset_window_alarm_rate", "Pre-onset windows with an alarm", 3)):
        lines.append(f"| {label} | " + " | ".join(f(r["alarms"][s][key], nd) + " (simulated)" for s in SOURCES) + " |")
    lines.append("| Missed engines | " + " | ".join(", ".join(r["alarms"][s]["missed"]) or "none" for s in SOURCES) + " |")
    ru = r["rul"]
    lines += ["", f"RUL (lifetime model `{ru['bundle']}`, trained on {ru['trained_on']} without engine variation), "
              "ESTIMATED windows of faulty engines, true RUL capped at 400 h:", "",
              "| True-RUL band | n | MAE (h) | 90 % interval coverage |", "|---|---|---|---|"]
    for band, v in ru["bands"].items():
        lines.append(f"| {band} | {v['n']} | {f(v['mae_h'], 1)} (simulated) | {f(v.get('coverage_90'), 2)} (simulated) |")
    lines += ["", f"Overall 90 % coverage {f(ru['coverage_90_overall'], 3)} (simulated); healthy-engine windows with an "
              f"ESTIMATED RUL {f(ru['healthy_windows_estimated_share'], 3)} (simulated).", "",
              "v2 lead time is deliberately NOT a headline: v2 alarms on 93 % of pre-fault windows "
              "(reports/adaptation_eval.md), so its lead time measures constant alarming.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="life2")
    ap.add_argument("--plan-dataset", default="life1")
    ap.add_argument("--model-version", default="v3")
    ap.add_argument("--rul-bundle", default=str(ROOT / "models" / "life1" / "rul_lifetime.joblib"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()
    import pandas as pd
    m = json.loads((ROOT / "data" / "datasets" / args.dataset / "manifest.json").read_text())
    pm = json.loads((ROOT / "data" / "datasets" / args.plan_dataset / "manifest.json").read_text())
    assert pm["engine_split"] == m["engine_split"], "engine splits differ"
    df = pd.read_parquet(ROOT / "data" / "datasets" / args.plan_dataset / pm["dataset_file"],
                         columns=["engine_id", "fault_class"]).drop_duplicates("engine_id")
    test = df[df["engine_id"].astype(str).map(m["engine_split"]) == "test"]
    jobs = [(int(e), int(c), m["seed"], args.model_version, str(ROOT / "models"))
            for e, c in zip(test["engine_id"], test["fault_class"])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_engine, jobs, chunksize=1))
    rep = {"provenance": "SIMULATED", "note": "every number is simulated", "dataset_split": args.dataset,
           "model_version": args.model_version, "variation": VARIATION,
           "healthy_engines": int((test["fault_class"] == 0).sum()),
           "faulty_engines": int((test["fault_class"] != 0).sum()),
           "definitions": {"window": "20 s at 1 Hz every 5 engine hours", "confidence_threshold": CONF,
                           "detection": "two consecutive alarm windows after onset"},
           "alarms": {s: detection(results, s) for s in SOURCES},
           "rule_classes_on_healthy_windows": pd.Series([w["rule_class"] for r in results if r["fault_class"] == 0
                                                         for w in r["windows"] if w["rule"]]).value_counts().to_dict(),
           "rul": rul_errors(results, Path(args.rul_bundle)), "seconds": round(time.time() - t0)}
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "headline_metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "headline_metrics.md").write_text(markdown(rep))
    print(markdown(rep))


if __name__ == "__main__":
    main()
