"""
D-01 — Thermocouple conversion: NIST ITS-90 Type K inverse.

DEFECT
------
File   : src/l2_digital_twin/sensor_inverse.py
Lines  : 97 (EGT, inside _convert_tc_egt) and 116 (CHT)
Config : src/core/config.py:227   type_k_uv_per_c: float = 41.27

    egt_c = cold_c + (hot_uv / cal.type_k_uv_per_c)

A single sensitivity constant replaces the NIST inverse polynomial. Type K
sensitivity is not constant — it runs from roughly 39 to 42 uV/degC across the
band, and 41.27 is its value near 1000 degC.

WRONG VALUES (shipped formula vs NIST ITS-90)
---------------------------------------------
    true 100 degC ->   99.25   error  -0.75
    true 200 degC ->  197.19   error  -2.81
    true 300 degC ->  295.83   error  -4.17
    true 600 degC ->  603.46   error  +3.46
    true 800 degC ->  806.28   error  +6.28
    true 1200 degC -> 1183.38  error -16.62

Worst error 16.62 degC against a +/-0.5 degC requirement — 33x tolerance.
Within the normal 750-900 degC EGT band the error is 6.28 degC, still 13x.

SECOND-ORDER CONSEQUENCE, WHICH IS WORSE
----------------------------------------
The error is non-monotonic: -4.17 degC at 300 degC, +6.28 degC at 800 degC.
Two cylinders at genuinely different temperatures therefore acquire DIFFERENT
conversion errors, which corrupts EGT SPREAD — the per-cylinder diagnostic
this system exists to provide. A healthy engine can show false spread, and a
real imbalance can be masked.

IMPLEMENTATION NOTE
-------------------
The NIST inverse polynomial seeds the two lower ranges, then EVERY inversion
is Newton-refined against the forward reference function, which is monotonic
across the Type K range. This makes the result exact by construction and
independent of any inverse-coefficient transcription error.

That is not a theoretical concern. The first draft of this module used
high-range inverse coefficients recalled from memory and returned -148 degC
for a true 800 degC — a larger error than the defect it was written to fix.
An inverse polynomial quoted from a secondary source is exactly the class of
defect this file corrects, so the implementation does not depend on one.

Requirements: SRD-FUN-001, SRD-FUN-002, SRD-CON-006, SRD-PER-010
Verified    : worst error 0.012 degC across 0-1200 degC
"""

from __future__ import annotations

import math

# --- NIST ITS-90 Type K forward reference function (E in mV, T in degC) -----

_K_FWD_NEG = [
    0.0, 3.9450128025e-2, 2.3622373598e-5, -3.2858906784e-7,
    -4.9904828777e-9, -6.7509059173e-11, -5.7410327428e-13,
    -3.1088872894e-15, -1.0451609365e-17, -1.9889266878e-20,
    -1.6322697486e-23,
]
_K_FWD_POS = [
    -1.7600413686e-2, 3.8921204975e-2, 1.8558770032e-5, -9.9457592874e-8,
    3.1840945719e-10, -5.6072844889e-13, 5.6075059059e-16,
    -3.2020720003e-19, 9.7151147152e-23, -1.2104721275e-26,
]
_K_A0, _K_A1, _K_A2 = 0.1185976, -1.183432e-4, 126.9686

# --- Inverse polynomial seeds (E in uV, T in degC) --------------------------

_K_INV_SEEDS = (
    (-5891.0, 0.0, [
        0.0, 2.5173462e-2, -1.1662878e-6, -1.0833638e-9, -8.9773540e-13,
        -3.7342377e-16, -8.6632643e-20, -1.0450598e-23, -5.1920577e-28,
    ]),
    (0.0, 20644.0, [
        0.0, 2.508355e-2, 7.860106e-8, -2.503131e-10, 8.315270e-14,
        -1.228034e-17, 9.804036e-22, -4.413030e-26, 1.057734e-30,
        -1.052755e-35,
    ]),
)

TYPE_K_TEMP_MIN_C = -270.0
TYPE_K_TEMP_MAX_C = 1372.0


def type_k_emf_uv(temp_c: float) -> float:
    """Type K Seebeck EMF in microvolts, referenced to 0 degC."""
    if temp_c < 0.0:
        e_mv = sum(c * temp_c ** i for i, c in enumerate(_K_FWD_NEG))
    else:
        e_mv = sum(c * temp_c ** i for i, c in enumerate(_K_FWD_POS))
        e_mv += _K_A0 * math.exp(_K_A1 * (temp_c - _K_A2) ** 2)
    return e_mv * 1000.0


# Guard bounds derived from the forward function so they cannot disagree.
TYPE_K_EMF_MIN_UV = type_k_emf_uv(TYPE_K_TEMP_MIN_C)
TYPE_K_EMF_MAX_UV = type_k_emf_uv(TYPE_K_TEMP_MAX_C)


def _seed_c(emf_uv: float) -> float:
    for e_lo, e_hi, coeffs in _K_INV_SEEDS:
        if e_lo <= emf_uv <= e_hi:
            return sum(c * emf_uv ** i for i, c in enumerate(coeffs))
    return 500.0 + (emf_uv - 20644.0) / 39.0


def type_k_temp_c(emf_uv: float, tol_c: float = 1e-9) -> float:
    """Invert Type K EMF (uV, referenced to 0 degC) to temperature in degC.

    Raises ValueError outside the tabulated range rather than extrapolating.
    """
    if not math.isfinite(emf_uv):
        raise ValueError("Type K EMF must be finite")
    if emf_uv < TYPE_K_EMF_MIN_UV or emf_uv > TYPE_K_EMF_MAX_UV:
        raise ValueError(
            f"Type K EMF {emf_uv:.1f} uV outside range "
            f"[{TYPE_K_EMF_MIN_UV:.1f}, {TYPE_K_EMF_MAX_UV:.1f}] uV"
        )

    t = min(max(_seed_c(emf_uv), TYPE_K_TEMP_MIN_C), TYPE_K_TEMP_MAX_C)
    for _ in range(60):
        f = type_k_emf_uv(t) - emf_uv
        h = 1e-4
        df = (type_k_emf_uv(t + h) - type_k_emf_uv(t - h)) / (2.0 * h)
        if abs(df) < 1e-9:
            break
        step = f / df
        t -= step
        if t < TYPE_K_TEMP_MIN_C or t > TYPE_K_TEMP_MAX_C:
            break
        if abs(step) < tol_c:
            return t
    else:
        return t

    lo, hi = TYPE_K_TEMP_MIN_C, TYPE_K_TEMP_MAX_C
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if type_k_emf_uv(mid) < emf_uv:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol_c:
            break
    return 0.5 * (lo + hi)


def thermocouple_k_to_celsius(hot_uv: float, cold_junction_c: float) -> float:
    """Cold-junction-compensated Type K conversion.

    The measured EMF is referenced to the cold junction, not to 0 degC, so the
    cold junction's own EMF is added before inversion:

        E_total = E_measured + E(T_cold)
        T_hot   = invert(E_total)
    """
    return type_k_temp_c(hot_uv + type_k_emf_uv(cold_junction_c))


def thermocouple_k_to_kelvin(hot_uv: float, cold_junction_c: float) -> float:
    """As thermocouple_k_to_celsius, returned in kelvin."""
    return thermocouple_k_to_celsius(hot_uv, cold_junction_c) + 273.15


def convert_tc_channel_kelvin(hot_uv: float, cold_c: float) -> tuple[float, bool, str]:
    """DROP-IN for _convert_tc_egt (line 95) and the CHT block (line 115).

    Returns (kelvin, valid, reason). Never raises: an out-of-range EMF is
    reported invalid rather than substituted, per SRD-INT-004.
    """
    if not math.isfinite(hot_uv) or not math.isfinite(cold_c):
        return 0.0, False, "Non-finite thermocouple input"
    try:
        return thermocouple_k_to_kelvin(hot_uv, cold_c), True, ""
    except ValueError as exc:
        return 0.0, False, str(exc)


if __name__ == "__main__":
    NIST = {0: 0, 100: 4096, 200: 8138, 300: 12209, 400: 16397, 500: 20644,
            600: 24905, 700: 29129, 800: 33275, 900: 37326, 1000: 41276,
            1200: 48838}
    print("D-01  Type K inverse vs NIST ITS-90       tolerance +/-0.5 degC")
    print(f"{'true':>6}{'NIST uV':>10}{'shipped':>10}{'err':>9}"
          f"{'fixed':>11}{'err':>9}")
    ws = wf = 0.0
    for t, uv in NIST.items():
        shipped = uv / 41.27
        fixed = type_k_temp_c(uv)
        ws = max(ws, abs(shipped - t))
        wf = max(wf, abs(fixed - t))
        print(f"{t:6d}{uv:10d}{shipped:10.2f}{shipped - t:+9.2f}"
              f"{fixed:11.3f}{fixed - t:+9.3f}")
    print(f"\n  shipped worst {ws:6.2f} degC  -> {ws / 0.5:.0f}x tolerance  FAIL")
    print(f"  fixed   worst {wf:6.3f} degC  -> {'PASS' if wf <= 0.5 else 'FAIL'}")

    rt = max(abs(type_k_temp_c(type_k_emf_uv(t)) - t) for t in range(-270, 1373, 10))
    print(f"  round-trip worst {rt:.2e} degC")

    meas = type_k_emf_uv(800.0) - type_k_emf_uv(25.0)
    print(f"\n  cold-junction check (hot 800, cold 25):")
    print(f"    fixed   {thermocouple_k_to_celsius(meas, 25.0):8.3f} degC")
    print(f"    shipped {25.0 + meas / 41.27:8.3f} degC")
    print(f"\n  guards: {convert_tc_channel_kelvin(9e9, 25.0)[1]}, "
          f"{convert_tc_channel_kelvin(float('nan'), 25.0)[1]}")
