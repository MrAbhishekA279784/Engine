"""
Adaptation and model-version evaluation on the lifetime fleet (Prompt 17, 17b).

Held-out TEST engines of the lifetime fleet (scripts/generate_lifetime_dataset.py,
same window plans and seeds; never in any training dataset) are flown with
each model version, without and with the gated per-engine adaptation
(src/l3_ml/adaptation.py). Every engine has the life2 engine-to-engine
variation and benign ageing (physical simulator parameters,
generate_lifetime_dataset.engine_variation). Adaptation targets the fleet
healthy-residual reference (--fleet-reference). Nothing here modifies a model
bundle or reads the Prompt 16 test split; the Prompt 16 test metrics come from
the scripts/evaluate_models.py reports.

    window alarm  anomaly alarm on >= half of the window's records, or the
                  rule-based class, or the ML class (confidence > 0.7), is not
                  NOMINAL at the window's last record
    (a) healthy engines: share of windows with an alarm (false alarms)
    (b) faulty engines: first detection = two consecutive alarm windows after
        onset; lead time = failure time - detection time; detection rate

Adaptation is accepted only if (a) drops and (b) is not worse. The model
version rule is pre-registered in compare_versions.

    uv run python scripts/evaluate_models.py --dataset v2 --version v3 --report-dir reports/v3
    uv run python scripts/evaluate_adaptation.py --model-versions v2,v3 --fleet-reference v3
Writes reports/adaptation_eval.json and .md. SIMULATED data only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_lifetime_dataset import (  # noqa: E402
    VARIATION, engine_settings, iter_engine_windows, run_window, window_settings)
from src.core.config import get_settings  # noqa: E402
from src.core.provenance import FaultClass, InferenceStatus  # noqa: E402

AGEING_EGT_K_PER_100H = VARIATION["ageing_egt_k_per_100h"]   # benign ageing rates sampled per engine (VERIFY)
AGEING_CHT_K_PER_100H = VARIATION["ageing_cht_k_per_100h"]


def window_alarm(steps, conf_threshold: float) -> dict:
    an = [s.anomaly_result.status == InferenceStatus.SUCCESS and s.anomaly_result.is_anomaly for s in steps]
    last = steps[-1]
    rule = last.rule_fault_result.predicted_class != FaultClass.NOMINAL
    ml = last.fault_result
    ml_alarm = (ml.status == InferenceStatus.SUCCESS and ml.predicted_class != FaultClass.NOMINAL
                and (ml.confidence or 0.0) > conf_threshold)
    anomaly = float(np.mean(an)) >= 0.5
    return {"anomaly": anomaly, "rule": rule, "ml": ml_alarm, "alarm": anomaly or rule or ml_alarm}


def run_engine(args) -> dict:
    from src.l3_ml import adaptation as A
    engine_id, cls, base_seed, adapt, model_version, model_dir, fleet_ref_version = args
    base_settings = get_settings().model_copy(deep=True)
    base_settings.model_dir, base_settings.ml.model_version = model_dir, model_version
    base_settings.adaptation.fleet_reference_version = fleet_ref_version
    base_settings = engine_settings(engine_id, base_seed, base_settings, variation=True)
    ph = base_settings.simulator.physics
    tmp = Path(tempfile.mkdtemp(prefix=f"adapt_{engine_id}_"))
    mgr = A.AdaptationManager(A.BaselineStore(tmp), A.AdaptationLog(tmp / "log.jsonl"), base_settings) if adapt else None
    windows, decisions = [], {}
    onset = failure = math.nan
    for w in iter_engine_windows(engine_id, cls, base_seed):
        baseline = mgr.store.active(str(engine_id)) if mgr else None
        adapted = baseline is not None and bool(baseline.corrections)
        steps = run_window(w, window_settings(base_settings, w["t_h"]), baseline if adapted else None)
        a = window_alarm(steps, base_settings.adaptation.fault_confidence_threshold)
        if mgr:
            d = mgr.process_flight(str(engine_id), steps, engine_hours=w["t_h"])
            decisions[d.decision] = decisions.get(d.decision, 0) + 1
        windows.append({"t_h": w["t_h"], **a, "adapted": adapted, "sev_life": w["sev_life"]})
        onset, failure = w["onset_h"], w["failure_h"]
    return {"engine_id": engine_id, "fault_class": cls, "adapt": adapt, "model_version": model_version,
            "ageing_egt": ph.ageing_egt_k_per_100h, "ageing_cht": ph.ageing_cht_k_per_100h,
            "onset_h": onset if math.isfinite(onset) else None,
            "failure_h": failure if math.isfinite(failure) else None, "windows": windows, "decisions": decisions}


def summarise(results: list[dict]) -> dict:
    healthy = [r for r in results if r["fault_class"] == 0]
    faulty = [r for r in results if r["fault_class"] != 0]
    fa = [w for r in healthy for w in r["windows"]]
    out = {"healthy_engines": len(healthy), "faulty_engines": len(faulty),
           "healthy_false_alarm_rate": float(np.mean([w["alarm"] for w in fa])) if fa else None,
           "healthy_false_alarm_by_source": {k: float(np.mean([w[k] for w in fa])) for k in ("anomaly", "rule", "ml")}
           if fa else {}}
    leads, detected, pre_onset = [], 0, []
    per_class: dict[str, list] = {}
    for r in faulty:
        ws = r["windows"]
        pre = [w["alarm"] for w in ws if r["onset_h"] is not None and w["t_h"] < r["onset_h"]]
        pre_onset.extend(pre)
        det = None
        for i in range(1, len(ws)):
            if ws[i]["t_h"] >= r["onset_h"] and ws[i]["alarm"] and ws[i - 1]["alarm"] and ws[i - 1]["t_h"] >= r["onset_h"]:
                det = ws[i]["t_h"]
                break
        name = FaultClass(r["fault_class"]).name
        per_class.setdefault(name, [])
        if det is not None:
            detected += 1
            leads.append(r["failure_h"] - det)
            per_class[name].append(r["failure_h"] - det)
    out.update({"detection_rate": detected / len(faulty) if faulty else None,
                "median_lead_h": float(np.median(leads)) if leads else None,
                "mean_lead_h": float(np.mean(leads)) if leads else None,
                "pre_onset_false_alarm_rate": float(np.mean(pre_onset)) if pre_onset else None,
                "per_class_detected_and_median_lead_h": {k: [len(v), float(np.median(v)) if v else None]
                                                         for k, v in sorted(per_class.items())}})
    dec: dict[str, int] = {}
    for r in results:
        for k, v in r["decisions"].items():
            dec[k] = dec.get(k, 0) + v
    out["adaptation_decisions"] = dec
    return out


def verdict(without: dict, with_: dict) -> dict:
    """Adaptation is accepted only if BOTH hold: (a) fewer false alarms on
    healthy engines and (b) detection rate and median lead time not worse."""
    worse = (with_["detection_rate"] < without["detection_rate"]) or (
        (with_["median_lead_h"] or 0.0) < (without["median_lead_h"] or 0.0))
    fewer_fa = (with_["healthy_false_alarm_rate"] or 0.0) < (without["healthy_false_alarm_rate"] or 0.0)
    if worse:
        v = "REJECTED: (b) fault detection is worse with adaptation"
    elif not fewer_fa:
        v = "REJECTED: (a) false alarms on healthy engines are not reduced"
    else:
        v = "ACCEPTED: (a) fewer false alarms and (b) detection not worse"
    return {"case_a_false_alarms_reduced": fewer_fa, "case_b_detection_worse": worse, "verdict": v}


# Pre-registered (Prompt 17b, written before any v3 result): the candidate
# model version is adopted only if, on the held-out lifetime test engines
# (without adaptation), healthy-window false alarms are LOWER and detection rate
# and median lead time are NOT worse, AND on the frozen Prompt 16 test split it
# does not regress beyond the Prompt 17 promotion margins (PR-AUC and macro-F1
# at most 0.02 lower, FPR at most 0.05).
def compare_versions(life: dict, p16: dict, champion: str, candidate: str, settings=None) -> dict:
    cfg = (settings or get_settings()).adaptation
    lc, lk = life[champion]["without_adaptation"], life[candidate]["without_adaptation"]
    pc, pk = p16[champion], p16[candidate]
    checks = {
        "lifetime_fewer_false_alarms": lk["healthy_false_alarm_rate"] < lc["healthy_false_alarm_rate"],
        "lifetime_detection_not_worse": lk["detection_rate"] >= lc["detection_rate"],
        "lifetime_lead_not_worse": (lk["median_lead_h"] or 0.0) >= (lc["median_lead_h"] or 0.0),
        "p16_pr_auc_within_margin": pk["pr_auc"] >= pc["pr_auc"] - cfg.promotion_max_pr_auc_drop,
        "p16_macro_f1_within_margin": pk["macro_f1"] >= pc["macro_f1"] - cfg.promotion_max_f1_drop,
        "p16_fpr_at_most_max": pk["fpr"] <= cfg.promotion_max_fpr,
    }
    strictly_better = {"p16_pr_auc": pk["pr_auc"] > pc["pr_auc"], "p16_macro_f1": pk["macro_f1"] > pc["macro_f1"],
                       "p16_fpr": pk["fpr"] < pc["fpr"],
                       "lifetime_false_alarms": checks["lifetime_fewer_false_alarms"]}
    adopt = all(checks.values())
    return {"champion": champion, "candidate": candidate, "checks": checks, "strictly_better": strictly_better,
            "adopt_candidate": adopt, "p16_test": {champion: pc, candidate: pk},
            "decision": (f"{candidate} adopted as the default model version" if adopt else
                         f"{champion} kept; failed: " + ", ".join(k for k, v in checks.items() if not v))}


def p16_test_metrics(report_path: Path) -> dict:
    t = json.loads(report_path.read_text())["test"]
    return {"pr_auc": t["anomaly"]["pr_auc"], "fpr": t["anomaly"]["fpr"], "macro_f1": t["classifier_ml"]["macro_f1"]}


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(report: dict, rep: Path) -> None:
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "adaptation_eval.json").write_text(json.dumps(report, indent=1, default=str))
    versions = report["versions"]
    lines = [f"# Adaptation and model-version evaluation on the lifetime fleet (SIMULATED)", "",
             f"{report['healthy_engines']} healthy and {report['faulty_engines']} faulty held-out TEST engines "
             f"(engine split of {report['dataset']}), engine-to-engine variation and benign ageing as in life2 "
             f"({report['variation']}); fleet healthy-residual reference {report['fleet_reference_version']}.",
             "Window alarm: anomaly on >= half the records, or rule / ML (confidence > 0.7) class not NOMINAL "
             "at the last record. Detection: two consecutive alarm windows after onset.", "",
             "| | " + " | ".join(f"{v} {m}" for v in versions for m in ("without", "with")) + " |",
             "|---|" + "---|" * (2 * len(versions))]

    def row(label, get, nd=3):
        cells = [_f(get(report["results"][v][m]), nd) for v in versions
                 for m in ("without_adaptation", "with_adaptation")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    row("(a) healthy windows with an alarm", lambda d: d["healthy_false_alarm_rate"])
    for k in ("anomaly", "ml", "rule"):
        row(f"- {k}", lambda d, k=k: d["healthy_false_alarm_by_source"].get(k))
    row("(b) detection rate", lambda d: d["detection_rate"])
    row("(b) median lead time (h)", lambda d: d["median_lead_h"], 1)
    row("pre-onset windows with an alarm", lambda d: d["pre_onset_false_alarm_rate"])
    row("healthy windows on an adapted baseline", lambda d: d.get("healthy_adapted_share"))
    lines.append("")
    for v in versions:
        r = report["results"][v]
        lines += [f"**{v} adaptation: {r['verdict']}**  ",
                  f"decisions (flights): {r['with_adaptation']['adaptation_decisions']}", ""]
    if report.get("comparison"):
        c = report["comparison"]
        lines += ["## Model version decision (pre-registered rule)", "",
                  "| Prompt 16 test | " + " | ".join(c["p16_test"]) + " |", "|---|---|---|"]
        for k in ("pr_auc", "fpr", "macro_f1"):
            lines.append(f"| {k} | " + " | ".join(_f(c["p16_test"][v][k]) for v in c["p16_test"]) + " |")
        lines += ["", "| check | pass |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c["checks"].items()]
        lines += ["", f"Strictly better: {c['strictly_better']}", "", f"**Decision: {c['decision']}**", ""]
    (rep / "adaptation_eval.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="life2", help="lifetime dataset whose manifest holds the engine split")
    ap.add_argument("--plan-dataset", default="life1", help="dataset with one row per engine and class (test engines)")
    ap.add_argument("--model-versions", default="v2,v3")
    ap.add_argument("--fleet-reference", default="v3")
    ap.add_argument("--p16-reports", default="v2=reports/ml_metrics.json,v3=reports/v3/ml_metrics.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()
    import pandas as pd
    manifest = json.loads((ROOT / "data" / "datasets" / args.dataset / "manifest.json").read_text())
    pm = json.loads((ROOT / "data" / "datasets" / args.plan_dataset / "manifest.json").read_text())
    assert pm["engine_split"] == manifest["engine_split"] and pm["seed"] == manifest["seed"], "engine splits differ"
    df = pd.read_parquet(ROOT / "data" / "datasets" / args.plan_dataset / pm["dataset_file"],
                         columns=["engine_id", "fault_class"]).drop_duplicates("engine_id")
    test = df[df["engine_id"].astype(str).map(manifest["engine_split"]) == "test"]
    versions = args.model_versions.split(",")
    jobs = [(int(e), int(c), manifest["seed"], adapt, v, str(ROOT / "models"), args.fleet_reference)
            for v in versions for e, c in zip(test["engine_id"], test["fault_class"]) for adapt in (False, True)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_engine, jobs, chunksize=1))
    out = {}
    for v in versions:
        rv = [r for r in results if r["model_version"] == v]
        wo, wi = summarise([r for r in rv if not r["adapt"]]), summarise([r for r in rv if r["adapt"]])
        hw = [w for r in rv if r["adapt"] and r["fault_class"] == 0 for w in r["windows"]]
        wi["healthy_adapted_share"] = float(np.mean([w["adapted"] for w in hw])) if hw else None
        wo["healthy_adapted_share"] = 0.0
        out[v] = {"without_adaptation": wo, "with_adaptation": wi, **verdict(wo, wi)}
    report = {"provenance": "SIMULATED", "dataset": args.dataset, "versions": versions,
              "fleet_reference_version": args.fleet_reference, "variation": VARIATION,
              "healthy_engines": int((test["fault_class"] == 0).sum()),
              "faulty_engines": int((test["fault_class"] != 0).sum()),
              "engines": sorted(int(e) for e in test["engine_id"]), "results": out,
              "seconds": round(time.time() - t0)}
    p16 = dict(kv.split("=") for kv in args.p16_reports.split(","))
    if len(versions) == 2 and all((ROOT / p16.get(v, "_missing")).exists() for v in versions):
        report["comparison"] = compare_versions(out, {v: p16_test_metrics(ROOT / p16[v]) for v in versions},
                                                versions[0], versions[1])
    write_report(report, Path(args.report_dir))
    print((Path(args.report_dir) / "adaptation_eval.md").read_text())


if __name__ == "__main__":
    main()
