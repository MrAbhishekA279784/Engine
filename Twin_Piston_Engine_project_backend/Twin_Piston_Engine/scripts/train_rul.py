"""
Train and evaluate the lifetime RUL model (method ported from aerotwin_ml,
src/l3_ml/lifetime_rul.py) on the lifetime fleet, and compare it with the
existing baseline RUL estimator (RULEstimator trend extrapolation) on the same
engines.

    uv run python scripts/train_rul.py --dataset life1 --version life1

Severity model: train engines (early stopping on val engines); out-of-fold
severity estimates for the train engines (3 folds by engine) feed the RUL
model, as in aerotwin. Intervals: calibrated on VALIDATION engines. Metrics:
held-out TEST engines, per true-RUL band 0-50 / 50-150 / 150-400 h, with 90 %
interval coverage. Adoption is decided on validation (beats the baseline in
every band that has data) and reported on test.

Writes models/<version>/rul_lifetime.joblib (git-ignored) and
reports/rul_metrics.json / .md. SIMULATED data only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.provenance import FaultClass, Provenance  # noqa: E402
from src.l3_ml import lifetime_rul as L  # noqa: E402
from src.l3_ml.ml_features import ML_FEATURES  # noqa: E402

SEED = 20260929


def baseline_rul(g) -> np.ndarray:
    """The existing baseline method (RULEstimator._estimate_rul_baseline: least-
    squares HI trend over the history, projected to the target HI, capped at
    the max horizon) applied to this engine's per-window health index on the
    engine-hour axis. The live RULEstimator itself cannot do this: its history
    tracker resets on any gap > 300 s, so across 5-hour monitoring windows it
    only ever returns INSUFFICIENT_HISTORY."""
    from src.core.config import get_settings
    cfg = get_settings().rul
    t = g["t_h"].to_numpy(float)
    hi = g["pipeline_hi"].to_numpy(float)
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        m = np.isfinite(hi[: i + 1])
        if m.sum() < cfg.min_history_samples or not np.isfinite(hi[i]):
            continue
        tt, yy = t[: i + 1][m], hi[: i + 1][m]
        sxx = float(np.sum((tt - tt.mean()) ** 2))
        slope = float(np.sum((tt - tt.mean()) * (yy - yy.mean())) / sxx) if sxx > 0 else 0.0  # per hour
        if slope >= 0.0:
            out[i] = cfg.max_prediction_horizon_hours
        else:
            out[i] = min((hi[i] - cfg.baseline_degradation_target_hi) / abs(slope), cfg.max_prediction_horizon_hours)
    return out


def per_engine(df, sev_hat, rul_model=None):
    import pandas as pd
    parts = []
    for eid, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("window_idx")
        h = L.history_features(sev_hat[g.index])
        h.index = g.index
        h["engine_id"], h["fault_class"], h["t_h"] = eid, g["fault_class"], g["t_h"]
        h["gt_rul_h"] = g["gt_rul_h"]
        if rul_model is not None:
            p = rul_model.predict(h)
            for k, v in p.items():
                h[k] = v
            h["baseline_h"] = baseline_rul(g)
        parts.append(h)
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="life1")
    ap.add_argument("--version", default="life1")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "datasets"))
    ap.add_argument("--model-root", default=str(ROOT / "models"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()
    import joblib
    import pandas as pd

    ds = Path(args.data_root) / args.dataset
    manifest = json.loads((ds / "manifest.json").read_text())
    assert manifest["features"] == list(ML_FEATURES)
    df = pd.read_parquet(ds / manifest["dataset_file"]).reset_index(drop=True)
    x = df[list(ML_FEATURES)].to_numpy(float)
    sev = df["sev_life"].to_numpy(float)
    tr, va, te = (df["split"] == s for s in ("train", "val", "test"))
    tr, va, te = tr.to_numpy(), va.to_numpy(), te.to_numpy()

    # severity model and out-of-fold estimates for the train engines
    sev_model = L.SeverityModel(SEED).fit(x[tr], sev[tr], x[va], sev[va])
    sev_hat = np.full(len(df), np.nan)
    tr_eng = np.array(sorted(df.loc[tr, "engine_id"].unique()))
    fold = dict(zip(tr_eng, np.random.default_rng(SEED).permutation(np.arange(tr_eng.size) % 3)))
    fold_of = df["engine_id"].map(fold).to_numpy()
    for k in range(3):
        fit_k, pred_k = tr & (fold_of != k), tr & (fold_of == k)
        mk = L.SeverityModel(SEED + k).fit(x[fit_k], sev[fit_k], x[va], sev[va])
        sev_hat[pred_k] = mk.predict(x[pred_k])
    sev_hat[va | te] = sev_model.predict(x[va | te])
    sev_mae = {s: float(np.mean(np.abs(sev_hat[m] - sev[m]))) for s, m in (("val", va), ("test", te))}

    faulty = np.isfinite(df["gt_rul_h"].to_numpy())
    h_tr = per_engine(df[tr & faulty], sev_hat)
    h_tr = h_tr[h_tr["windows_since_degradation"] >= 0]
    rul = L.RULModel(SEED).fit(h_tr, h_tr["gt_rul_h"].to_numpy())
    h_va0 = per_engine(df[va & faulty], sev_hat)
    h_va0 = h_va0[h_va0["windows_since_degradation"] >= 0]
    rul.calibrate(h_va0, h_va0["gt_rul_h"].to_numpy())

    def evaluate(mask):
        h = per_engine(df[mask], sev_hat, rul)
        f = h[np.isfinite(h["gt_rul_h"])]
        est = f[f["status"] == "ESTIMATED"]
        base_same = np.minimum(est["baseline_h"].to_numpy(), L.RUL_CAP_H)
        healthy = h[~np.isfinite(h["gt_rul_h"])]
        return {
            "engines": int(h["engine_id"].nunique()),
            "ml": L.band_metrics(est["gt_rul_h"].to_numpy(), est["rul_h"].to_numpy(), est["rul_lo_h"].to_numpy(),
                                 est["rul_hi_h"].to_numpy()),
            "baseline_same_rows": L.band_metrics(est["gt_rul_h"].to_numpy(), base_same),
            "baseline_all_faulty_rows": L.band_metrics(f["gt_rul_h"].to_numpy(),
                                                       np.minimum(f["baseline_h"].to_numpy(), L.RUL_CAP_H)),
            "ml_interval_coverage_90": float(np.mean((np.minimum(est["gt_rul_h"], L.RUL_CAP_H) >= est["rul_lo_h"])
                                                     & (np.minimum(est["gt_rul_h"], L.RUL_CAP_H) <= est["rul_hi_h"])))
            if len(est) else None,
            "status_share_faulty": f["status"].value_counts(normalize=True).round(4).to_dict(),
            "healthy_engines_estimated_share": float(np.mean(healthy["status"] == "ESTIMATED")) if len(healthy) else None,
            "baseline_median_h_healthy": float(np.nanmedian(healthy["baseline_h"])) if len(healthy) else None,
        }, h

    val_res, _ = evaluate(va)
    test_res, h_test = evaluate(te)

    def beats(res):
        bands = [b for b, v in res["ml"].items() if v["n"] and res["baseline_same_rows"][b]["mae_h"] is not None]
        return bool(bands) and all(res["ml"][b]["mae_h"] < res["baseline_same_rows"][b]["mae_h"] for b in bands), bands

    adopt_val, bands_val = beats(val_res)
    adopt_test, bands_test = beats(test_res)
    decision = {"adopted": adopt_val, "decided_on": "validation engines",
                "bands_with_data_val": bands_val, "beats_baseline_val": adopt_val,
                "bands_with_data_test": bands_test, "beats_baseline_test": adopt_test}

    out = Path(args.model_root) / args.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": "lifetime_rul", "severity_model": sev_model, "rul_model": rul, "features": list(ML_FEATURES),
                 "history_cols": L.HISTORY_COLS, "monitor_cadence_h": L.MONITOR_CADENCE_H, "rul_cap_h": L.RUL_CAP_H,
                 "band_q": rul.band_q, "decision": decision,
                 "provenance": {"dataset": manifest["version"], "dataset_sha256": manifest["dataset_sha256"],
                                "seed": SEED, "data_provenance": "SIMULATED"},
                 "trained_at": datetime.now(timezone.utc).isoformat()}, out / "rul_lifetime.joblib")
    report = {"provenance": "SIMULATED", "dataset": {k: manifest[k] for k in ("version", "dataset_sha256", "seed",
                                                                             "engines", "windows", "code_git_sha",
                                                                             "monitor_cadence_h", "fleet_classes")},
              "method": "ported from aerotwin_ml (fleet/faults/rul), driven by this backend's simulator and pipeline",
              "severity_mae": sev_mae, "interval_bands_q05_q95_val": rul.band_q,
              "validation": val_res, "test": test_res, "decision": decision}
    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "rul_metrics.json").write_text(json.dumps(report, indent=1, default=str))
    (rep / "rul_metrics.md").write_text(markdown(report))
    print(json.dumps({"severity_mae": sev_mae, "test": test_res, "decision": decision}, indent=1, default=str))


def markdown(r) -> str:
    def f(v, n=1):
        return "n/a" if v is None else f"{v:.{n}f}"
    t = r["test"]
    lines = ["# Lifetime RUL metrics", "",
             "**Measured on SIMULATOR data only (provenance SIMULATED).** Held-out TEST engines.", "",
             f"Fleet {r['dataset']['version']}: {r['dataset']['engines']} engines, {r['dataset']['windows']} monitoring "
             f"windows (one per {r['dataset']['monitor_cadence_h']} engine hours), sha256 {r['dataset']['dataset_sha256'][:12]}.",
             f"Severity model MAE: val {f(r['severity_mae']['val'], 3)}, test {f(r['severity_mae']['test'], 3)}.", "",
             "| True-RUL band | n | ML MAE [h] | ML 90 % coverage | Baseline MAE [h], same rows |",
             "|---|---|---|---|---|"]
    for b, v in t["ml"].items():
        lines.append(f"| {b} | {v['n']} | {f(v['mae_h'])} | {f(v.get('coverage_90'), 2)} | "
                     f"{f(t['baseline_same_rows'][b]['mae_h'])} |")
    lines += ["", f"Overall 90 % interval coverage: {f(t['ml_interval_coverage_90'], 3)}.",
              f"Status share on faulty test windows: {t['status_share_faulty']}.",
              f"Healthy test engines: share of windows with an ESTIMATED RUL {f(t['healthy_engines_estimated_share'], 3)}; "
              f"baseline median {f(t['baseline_median_h_healthy'])} h.", "",
              f"Decision: {'ADOPTED' if r['decision']['adopted'] else 'NOT ADOPTED'} "
              f"(decided on validation: beats baseline in every band with data = {r['decision']['beats_baseline_val']}; "
              f"test: {r['decision']['beats_baseline_test']})."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
