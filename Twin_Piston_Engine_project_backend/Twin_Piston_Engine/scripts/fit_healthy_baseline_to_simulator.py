"""
DIAGNOSTIC ONLY -- reports, never writes configuration.

Compares the healthy-expectation models in HealthyBaselineConfig (the values
the residual engine actually uses) with the nominal forward simulator, and
prints what a least-squares fit of the same model forms WOULD give.

The fitted numbers are for information. They must NOT be copied into config:
the healthy expectations are deliberately not calibrated to the simulator,
so the residual layer stays an independent check of it (docs/OPEN_ITEMS.md,
OI-5). They need healthy-flight data instead.

Run from the repo root:
    uv run python scripts/fit_healthy_baseline_to_simulator.py
"""

from __future__ import annotations

import numpy as np

from src.core.config import get_settings
from src.l1_data.simulator.forward_simulator import ForwardPhysicsModel

STD_P = 101325.0
STD_T = 288.15
RATED_KW = 105.0
RATED_RPM = 5800.0
CHT_LOAD_MAP_REF = 140000.0   # M-11 RATED_MAP_PA (load definition, not fitted)

RPMS = np.arange(2000.0, 5801.0, 200.0)
MAPS_FIT = np.arange(70000.0, 135001.0, 5000.0)   # normal envelope (lambda = 1)
MAPS_REPORT = np.arange(70000.0, 160001.0, 5000.0)


def grid(maps):
    m = ForwardPhysicsModel(seed=7)
    rows = []
    for r in RPMS:
        for p in maps:
            gt = m.compute_ground_truth(time_s=1.0, rpm=float(r), map_pa=float(p), ambient_temp_k=STD_T)
            rows.append((r, p, gt.egt_k[0], gt.cht_k, gt.brake_power_kw, gt.fuel_flow_kg_s))
    return np.array(rows)


def fit(data):
    r, p, egt, cht, pw, fuel = data.T
    # EGT: base + k_map (MAP - 101325) + k_rpm max(0, rpm - 1000)   (ambient term 0 at ISA)
    a = np.column_stack([np.ones_like(r), p - STD_P, np.maximum(0.0, r - 1000.0)])
    egt_c = np.linalg.lstsq(a, egt, rcond=None)[0]
    # CHT: base + k_load * load, load = (MAP/140k)(rpm/5800) clipped at 1.2
    load = np.minimum((p / CHT_LOAD_MAP_REF) * (r / RATED_RPM), 1.2)
    cht_c = np.linalg.lstsq(np.column_stack([np.ones_like(r), load]), cht, rcond=None)[0]
    # Power: 105 kW * (MAP / map_ref) * (rpm / 5800)  ->  fit 1/map_ref
    x = RATED_KW * p * r / RATED_RPM
    inv_ref = float(np.dot(x, pw) / np.dot(x, x))
    p_exp = x * inv_ref
    # Fuel: P_expected * SFC / 3600  ->  fit SFC
    sfc = float(np.dot(p_exp, fuel) / np.dot(p_exp, p_exp) * 3600.0)
    return {
        "base_egt_k": float(egt_c[0]), "k_egt_map_pa": float(egt_c[1]), "k_egt_rpm": float(egt_c[2]),
        "base_cht_k": float(cht_c[0]), "k_cht_load_k": float(cht_c[1]),
        "expected_power_map_ref_pa": 1.0 / inv_ref, "baseline_sfc_kg_kwh": sfc,
    }


def errors(data, c):
    r, p, egt, cht, pw, fuel = data.T
    e_egt = c["base_egt_k"] + c["k_egt_map_pa"] * (p - STD_P) + c["k_egt_rpm"] * np.maximum(0.0, r - 1000.0)
    load = np.minimum((p / CHT_LOAD_MAP_REF) * (r / RATED_RPM), 1.2)
    e_cht = c["base_cht_k"] + c["k_cht_load_k"] * load
    e_pw = RATED_KW * (p / c["expected_power_map_ref_pa"]) * (r / RATED_RPM)
    e_fuel = np.maximum(0.00035, e_pw * c["baseline_sfc_kg_kwh"] / 3600.0)
    return {
        "egt_K": np.max(np.abs(egt - e_egt)),
        "cht_pct": 100 * np.max(np.abs(cht - e_cht) / e_cht),
        "power_pct": 100 * np.max(np.abs(pw - e_pw) / e_pw),
        "fuel_pct": 100 * np.max(np.abs(fuel - e_fuel) / e_fuel),
    }


def configured() -> dict[str, float]:
    """The coefficients the residual engine currently uses."""
    b = get_settings().healthy_baseline
    return {
        "base_egt_k": b.base_egt_k, "k_egt_map_pa": b.k_egt_map_pa, "k_egt_rpm": b.k_egt_rpm,
        "base_cht_k": b.base_cht_k, "k_cht_load_k": b.k_cht_load_k,
        "expected_power_map_ref_pa": b.expected_power_map_ref_pa, "baseline_sfc_kg_kwh": b.baseline_sfc_kg_kwh,
    }


def _fmt(errs: dict) -> dict:
    return {k: round(float(v), 2) for k, v in errs.items()}


if __name__ == "__main__":
    print("DIAGNOSTIC ONLY: nothing is written to configuration.")
    print()
    current = configured()
    print("Configured HealthyBaselineConfig vs nominal simulator (worst case):")
    print("  70-135 kPa :", _fmt(errors(grid(MAPS_FIT), current)))
    print("  70-160 kPa :", _fmt(errors(grid(MAPS_REPORT), current)))

    coeffs = fit(grid(MAPS_FIT))
    print()
    print("For information only -- least-squares fit of the same forms to the")
    print("simulator (2000-5800 rpm, 70-135 kPa). Do NOT copy into config (OI-5):")
    for k, v in coeffs.items():
        print(f"  {k}: {v:.6g}")
    print("  residual error of that fit, 70-135 kPa:", _fmt(errors(grid(MAPS_FIT), coeffs)))
