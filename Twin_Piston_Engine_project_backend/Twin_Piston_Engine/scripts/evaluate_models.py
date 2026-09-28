"""
Evaluate trained models on the TEST split (and, separately, the held-out
operating point). Nothing here feeds back into training or thresholds.

    uv run python scripts/evaluate_models.py --dataset v1 --version v1

Writes reports/ml_metrics.json and reports/ml_metrics.md. Every metric is
measured on SIMULATOR data (provenance SIMULATED, PRD 4.4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.provenance import FAULT_CLASS_COUNT, FaultClass  # noqa: E402
from src.l3_ml import training as T  # noqa: E402
from src.l3_ml.ml_features import FEATURE_GROUPS, ML_FEATURES, SENSOR_GROUPS  # noqa: E402

TARGETS = {"anomaly_pr_auc": 0.93, "anomaly_fpr_max": 0.05, "classifier_macro_f1": 0.81}
UNOBSERVABLE = (FaultClass.INTAKE_BOOST_LEAK, FaultClass.BATTERY_DEGRADATION, FaultClass.INJECTOR_FAULT,
                FaultClass.COOLING_FAULT)
SUSTAIN = 5
CONF_THRESHOLD = 0.70  # ml.fault_confidence_threshold


class Models:
    def __init__(self, model_dir: Path) -> None:
        import joblib
        self.an = joblib.load(model_dir / "anomaly_detector.joblib")
        self.fc = joblib.load(model_dir / "fault_classifier.joblib")
        self.imp = np.asarray([self.fc["importance"][n] for n in ML_FEATURES])
        self.classes = [int(c) for c in self.fc["estimator"].classes_]

    @staticmethod
    def abstain(bundle, x):
        """Rows where a sensor group unseen-as-missing in training is fully
        missing (mirrors trained_models.unseen_missing_groups)."""
        from src.l3_ml.trained_models import OOD_GROUP_MISSING_RATE
        rates = bundle.get("group_missing_rate") or {}
        out = np.zeros(len(x), bool)
        for g in SENSOR_GROUPS:
            if rates.get(g, 0.0) >= OOD_GROUP_MISSING_RATE:
                continue
            cols = [j for j, n in enumerate(ML_FEATURES) if FEATURE_GROUPS[n] == g]
            out |= np.isnan(x[:, cols]).all(axis=1)
        return out

    def anomaly(self, x):
        s = -self.an["estimator"].score_samples(x)
        flag = s >= self.an["threshold"]
        flag[self.abstain(self.an, x)] = False  # abstained: no ML alarm (rules apply)
        return s, flag

    def classify(self, x):
        p = self.fc["estimator"].predict_proba(x)
        full = np.zeros((len(x), FAULT_CLASS_COUNT))
        for i, c in enumerate(self.classes):
            full[:, c] = p[:, i]
        cov = (~np.isnan(x)).astype(float) @ self.imp / self.imp.sum()
        pred, conf = full.argmax(1), full.max(1) * cov
        ab = self.abstain(self.fc, x)
        pred[ab] = -1   # MODEL_UNAVAILABLE: no ML class
        conf[ab] = 0.0
        return pred, conf, full


def cls_metrics(y, pred):
    from sklearn.metrics import confusion_matrix, f1_score
    labels = list(range(FAULT_CLASS_COUNT))
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "per_class_f1": {FaultClass(c).name: float(v) for c, v in
                             zip(labels, f1_score(y, pred, average=None, labels=labels, zero_division=0))},
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "accuracy": float(np.mean(y == pred))}


def anomaly_metrics(y, score, flag):
    from sklearn.metrics import average_precision_score
    pos = y != int(FaultClass.NOMINAL)
    return {"pr_auc": float(average_precision_score(pos, score)), "fpr": float(np.mean(flag[~pos])),
            "recall": float(np.mean(flag[pos])), "n_records": int(len(y)), "n_positive": int(pos.sum())}


def lead_times(df, flag, plans):
    leads, missed = [], 0
    df = df.assign(_flag=flag)
    for tid, g in df.groupby("trajectory_id"):
        plan = plans[int(tid)]
        if plan["fault"] == 0 or plan["progression_rate"] <= 0.0:
            continue
        t_fail = plan["onset_s"] + plan["severity"] / plan["progression_rate"]
        if t_fail > plan["duration_s"]:
            continue
        g = g.sort_values("time_s")
        f = g["_flag"].to_numpy()
        t = g["time_s"].to_numpy()
        run = np.convolve(f.astype(int), np.ones(SUSTAIN, int), "full")[:len(f)] >= SUSTAIN
        ok = np.where(run & (t >= plan["onset_s"]))[0]
        t_warn = t[ok[0]] - (SUSTAIN - 1) if len(ok) else math.inf
        if t_warn <= t_fail:
            leads.append(t_fail - t_warn)
        else:
            missed += 1
    return {"median_lead_s": float(np.median(leads)) if leads else None, "warned_before_failure": len(leads),
            "missed": missed, "sustain_records": SUSTAIN,
            "definition": "anomaly alarm sustained 5 records after onset, before the fault reaches full severity"}


def cruise_point_confusion(version: str, model_root: Path):
    from src.core.config import get_settings
    from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
    from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter
    from tests.scientific.test_fault_taxonomy import FAULTS

    s = get_settings().model_copy(deep=True)
    s.model_dir = str(model_root)
    s.ml.model_version = version
    out = {}
    for fc, kw in FAULTS.items():
        faults = None if kw is None else [FaultScenarioConfig(fault_class=fc, onset_time_s=30.0, duration_s=1e4, **kw)]
        recs, _, _ = ScenarioRunner(seed=42).run_scenario(duration_s=120.0, dt_s=1.0,
                                                           operating_profile=[{"rpm": 4000.0, "map_pa": 110000.0}],
                                                           fault_scenarios=faults)
        last = PipelineReplayAdapter(s).process_sequence(recs)[-1]
        ml = last.fault_result
        out[fc.name] = {"rule": last.rule_fault_result.class_name, "ml": ml.class_name, "ml_status": ml.status.value,
                        "ml_is_ml": ml.is_ml, "ml_confidence": ml.confidence,
                        "anomaly": bool(last.anomaly_result.is_anomaly), "anomaly_score": last.anomaly_result.anomaly_score,
                        "features": last.ml_features}
    return out


def permutation_audit(models, df, cls: FaultClass, seed: int, top: int = 6):
    rng = np.random.default_rng(seed)
    sub = df[(df["label"] == int(cls)) | (df["label"] == 0)]
    x = sub[list(ML_FEATURES)].to_numpy(float)
    y = sub["label"].to_numpy(int)
    pos = y == int(cls)
    base = float(np.mean(models.classify(x)[0][pos] == int(cls))) if pos.any() else math.nan
    drops = []
    for j, name in enumerate(ML_FEATURES):
        xp = x.copy()
        xp[:, j] = rng.permutation(xp[:, j])
        rec = float(np.mean(models.classify(xp)[0][pos] == int(cls)))
        drops.append((name, base - rec))
    drops.sort(key=lambda kv: -kv[1])
    return {"recall": base, "n_records": int(pos.sum()), "top_permutation_drops": drops[:top]}


def markdown(r: dict) -> str:
    def f(x, n=3):
        return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    t, h = r["test"], r["holdout_op"]
    lines = [
        "# ML metrics (Prompt 16)",
        "",
        "**Every metric below was measured on SIMULATOR data (provenance SIMULATED, PRD 4.4):**",
        "the models were trained, thresholded and evaluated on trajectories from this repository's",
        "forward simulator. None is a flight-data result.",
        "",
        f"Dataset {r['dataset']['version']} (sha256 {r['dataset']['dataset_sha256'][:12]}, seed {r['dataset']['seed']},",
        f"code {r['dataset']['code_git_sha']}), {r['dataset']['trajectories']} trajectories, splits {r['dataset']['splits']};",
        f"models {r['model_version']}. Test split only; the held-out operating point (cruise 4400-4600 rpm) separately.",
        "",
        "| Metric | Test | Held-out op | Target |",
        "|---|---|---|---|",
        f"| Anomaly PR-AUC (SRD-PER-016) | {f(t['anomaly']['pr_auc'])} | {f(h['anomaly']['pr_auc'])} | {TARGETS['anomaly_pr_auc']} |",
        f"| Anomaly FPR (SRD-PER-017) | {f(t['anomaly']['fpr'])} | {f(h['anomaly']['fpr'])} | <= {TARGETS['anomaly_fpr_max']} |",
        f"| Classifier macro-F1, ML (SRD-PER-018) | {f(t['classifier_ml']['macro_f1'])} | {f(h['classifier_ml']['macro_f1'])} | {TARGETS['classifier_macro_f1']} |",
        f"| Classifier macro-F1, rules (same records) | {f(t['classifier_rules']['macro_f1'])} | {f(h['classifier_rules']['macro_f1'])} | - |",
        f"| Median warning lead time [s] (SRD-PER-019) | {f(t['warning_lead_time']['median_lead_s'], 1)} | {f(h['warning_lead_time']['median_lead_s'], 1)} | - |",
        f"| RUL baseline MAE <=50 h [h] (SRD-PER-020) | {f(t['rul_baseline_error']['<=50h']['mae_h'], 1)} | {f(h['rul_baseline_error']['<=50h']['mae_h'], 1)} | - |",
        "",
        "RUL bands 50-150 h and >150 h have no samples: the simulated faults develop over minutes.",
        f"RUL regressor: {r['rul']['regressor']['decision']}.",
        "",
        "Per-class F1 (test): " + ", ".join(f"{k} {v:.2f}" for k, v in t["classifier_ml"]["per_class_f1"].items()),
        "",
        "Cruise point (4000 rpm / 110 kPa, 120 s): injected -> rule / ML",
        "",
    ] + [f"- {k}: {v['rule']} / {v['ml']} ({v['ml_status']})" for k, v in r["cruise_point"].items()] + [
        "",
        "Channel dropout (test): abstain rate / confident-wrong rate per sensor group removed",
        "",
    ] + [f"- {g}: {f(v['abstain_rate'])} / {f(v['confident_wrong_rate'])}" for g, v in r["channel_dropout"]["groups"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v1")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "datasets"))
    ap.add_argument("--model-root", default=str(ROOT / "models"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()

    import pandas as pd

    ds_dir = Path(args.data_root) / args.dataset
    manifest = json.loads((ds_dir / "manifest.json").read_text())
    df = pd.read_parquet(ds_dir / manifest["dataset_file"])
    plans = {p["trajectory_id"]: p for p in manifest["plans"]}
    models = Models(Path(args.model_root) / args.version)
    card = json.loads((Path(args.model_root) / args.version / "model_card.json").read_text())

    report: dict = {"provenance": "SIMULATED", "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": {k: manifest[k] for k in ("version", "dataset_sha256", "seed", "code_git_sha",
                                                         "feature_schema_version", "rows", "trajectories", "splits")},
                    "model_version": args.version, "targets": TARGETS}
    for split in ("test", "holdout_op"):
        d = df[df["split"] == split]
        x, y = d[list(ML_FEATURES)].to_numpy(float), d["label"].to_numpy(int)
        score, flag = models.anomaly(x)
        pred, conf, _ = models.classify(x)
        rule = d["rule_class"].to_numpy(int)
        true_h = d["ttf_s"].to_numpy(float) / 3600.0
        fault_rows = (d["injected_class"].to_numpy() != 0) & np.isfinite(true_h)
        report[split] = {
            "trajectories": int(d["trajectory_id"].nunique()),
            "anomaly": anomaly_metrics(y, score, flag),
            "classifier_ml": cls_metrics(y, pred),
            "classifier_rules": cls_metrics(y, rule),
            "warning_lead_time": lead_times(d, flag, plans),
            "rul_baseline_error": T.rul_band_errors(true_h[fault_rows], d["rul_baseline_h"].to_numpy(float)[fault_rows]),
        }

    # Channel dropout (test split): each sensor group removed
    d = df[df["split"] == "test"]
    x0, y = d[list(ML_FEATURES)].to_numpy(float), d["label"].to_numpy(int)
    nominal = y == 0
    p0, c0, _ = models.classify(x0)
    def dropout_stats(pg, cg, flag, cols=0):
        answered = pg >= 0
        return {"features_removed": cols, "abstain_rate": float(np.mean(~answered)),
                "accuracy_when_answered": float(np.mean(pg[answered] == y[answered])) if answered.any() else None,
                "mean_confidence_when_answered": float(np.mean(cg[answered])) if answered.any() else None,
                "confident_wrong_rate": float(np.mean(answered & (pg != y) & (cg >= CONF_THRESHOLD))),
                "anomaly_alarm_rate_nominal": float(np.mean(flag[nominal]))}

    drop = {"none": dropout_stats(p0, c0, models.anomaly(x0)[1])}
    for g in SENSOR_GROUPS:
        cols = [j for j, n in enumerate(ML_FEATURES) if FEATURE_GROUPS[n] == g]
        xg = x0.copy()
        xg[:, cols] = np.nan
        pg, cg, _ = models.classify(xg)
        drop[g] = dropout_stats(pg, cg, models.anomaly(xg)[1], len(cols))
    report["channel_dropout"] = {"confidence_threshold": CONF_THRESHOLD, "groups": drop}
    report["rul"] = {"baseline": "RULEstimator (existing trend estimator) kept", "regressor": card["rul"]}

    # Cruise point: ML vs rules (Prompt 15 scenarios) and the leakage audit
    cruise = cruise_point_confusion(args.version, Path(args.model_root))
    report["cruise_point"] = {k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in cruise.items()}
    audit = {}
    val = df[df["split"] == "val"]
    for fc in UNOBSERVABLE:
        c = cruise[fc.name]
        audit[fc.name] = {"rule_at_cruise": c["rule"], "ml_at_cruise": c["ml"],
                          "ml_beats_rule_at_cruise": c["ml"] == fc.name and c["rule"] != fc.name,
                          "val_split_decision": permutation_audit(models, val, fc, manifest["seed"]),
                          "test_split_report": permutation_audit(models, d, fc, manifest["seed"])}
    report["leakage_audit"] = audit
    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "ml_metrics.json").write_text(json.dumps(report, indent=1, default=str))
    (rep / "ml_metrics.md").write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ("test", "holdout_op")}, default=str)[:3000])


if __name__ == "__main__":
    main()
