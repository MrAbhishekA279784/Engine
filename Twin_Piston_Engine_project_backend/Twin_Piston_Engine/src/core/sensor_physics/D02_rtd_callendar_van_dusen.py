"""
D-02 — RTD conversion: IEC 60751 Callendar-Van Dusen.

DEFECT
------
File   : src/l2_digital_twin/sensor_inverse.py
Lines  : 129-132
Config : src/core/config.py:228-229
             pt100_r0_ohms:    float = 100.0
             pt100_alpha_per_c: float = 0.00385

    (raw_record.oil_rtd_ohms - cal.pt100_r0_ohms)
        / (cal.pt100_r0_ohms * cal.pt100_alpha_per_c)

This is the linear alpha-form approximation. The true relation is quadratic
above 0 degC and quartic below it.

WRONG VALUES (shipped formula vs IEC 60751)
-------------------------------------------
    true  -40 degC ->  -40.86   error  -0.86
    true    0 degC ->    0.00   error  +0.00
    true   50 degC ->   50.38   error  +0.38
    true  100 degC ->  100.01   error  +0.01
    true  125 degC ->  124.55   error  -0.45
    true  150 degC ->  148.90   error  -1.10

Worst error 1.10 degC against a +/-0.2 degC requirement — 6x tolerance.

The error is near zero at 0 degC and 100 degC and grows between and beyond
them. That is the signature of a linear fit pinned at two points of a
quadratic — it looks convincing wherever someone is most likely to spot-check
it, and is worst at the top of the operating range.

WHY IT MATTERS DOWNSTREAM
-------------------------
Oil temperature drives the viscosity model, which drives expected oil
pressure, which drives the bearing-wear indicator. A 1.1 degC bias at the top
of the range propagates into all three.

Requirements: SRD-FUN-003, SRD-PER-011
Verified    : worst error ~1e-13 degC across -40 to +150 degC
"""

from __future__ import annotations

import math

# IEC 60751 coefficients for a standard Pt100 (alpha = 0.00385)
PT100_R0 = 100.0
PT100_A = 3.9083e-3
PT100_B = -5.775e-7
PT100_C = -4.183e-12      # below 0 degC only


def pt100_resistance_ohms(temp_c: float, r0: float = PT100_R0) -> float:
    """Callendar-Van Dusen forward: temperature -> resistance."""
    if temp_c >= 0.0:
        return r0 * (1.0 + PT100_A * temp_c + PT100_B * temp_c * temp_c)
    return r0 * (
        1.0
        + PT100_A * temp_c
        + PT100_B * temp_c * temp_c
        + PT100_C * (temp_c - 100.0) * temp_c ** 3
    )


def pt100_temp_c(resistance_ohms: float, r0: float = PT100_R0) -> float:
    """Invert Pt100 resistance to temperature in degC.

    For T >= 0 the relation is quadratic and solved exactly:

        R = R0 (1 + A T + B T^2)
        =>  B T^2 + A T + (1 - R/R0) = 0
        =>  T = (-A + sqrt(A^2 - 4 B (1 - R/R0))) / (2 B)

    B is negative, so the physical root is the one written above.

    Below 0 degC the C term makes the relation quartic; Newton iteration on
    the forward function converges in a few steps and is used there.
    """
    if not math.isfinite(resistance_ohms) or resistance_ohms <= 0.0:
        raise ValueError("Pt100 resistance must be finite and positive")

    if resistance_ohms >= r0:
        disc = PT100_A * PT100_A - 4.0 * PT100_B * (1.0 - resistance_ohms / r0)
        if disc < 0.0:
            raise ValueError(
                f"Pt100 resistance {resistance_ohms:.3f} ohm has no real solution"
            )
        return (-PT100_A + math.sqrt(disc)) / (2.0 * PT100_B)

    t = (resistance_ohms - r0) / (r0 * PT100_A)      # linear seed
    for _ in range(50):
        f = pt100_resistance_ohms(t, r0) - resistance_ohms
        df = r0 * (
            PT100_A + 2.0 * PT100_B * t
            + PT100_C * (4.0 * t ** 3 - 300.0 * t * t)
        )
        if abs(df) < 1e-12:
            break
        step = f / df
        t -= step
        if abs(step) < 1e-10:
            break
    return t


def pt100_temp_kelvin(resistance_ohms: float, r0: float = PT100_R0) -> float:
    """As pt100_temp_c, returned in kelvin."""
    return pt100_temp_c(resistance_ohms, r0) + 273.15


def convert_rtd_channel_kelvin(resistance_ohms: float) -> tuple[float, bool, str]:
    """DROP-IN for the RTD block at sensor_inverse.py:129-132.

    Returns (kelvin, valid, reason). Never raises.
    """
    if not math.isfinite(resistance_ohms) or resistance_ohms <= 0.0:
        return 0.0, False, "Non-positive RTD resistance"
    try:
        return pt100_temp_kelvin(resistance_ohms), True, ""
    except ValueError as exc:
        return 0.0, False, str(exc)


if __name__ == "__main__":
    print("D-02  Pt100 inverse vs IEC 60751          tolerance +/-0.2 degC")
    print(f"{'true':>6}{'IEC ohm':>11}{'shipped':>10}{'err':>9}"
          f"{'fixed':>13}{'err':>12}")
    ws = wf = 0.0
    for t in (-40, -20, 0, 25, 50, 80, 100, 125, 150):
        r = pt100_resistance_ohms(t)
        shipped = (r - 100.0) / (100.0 * 0.00385)
        fixed = pt100_temp_c(r)
        ws = max(ws, abs(shipped - t))
        wf = max(wf, abs(fixed - t))
        print(f"{t:6d}{r:11.3f}{shipped:10.2f}{shipped - t:+9.2f}"
              f"{fixed:13.5f}{fixed - t:+12.2e}")
    print(f"\n  shipped worst {ws:6.2f} degC  -> {ws / 0.2:.0f}x tolerance  FAIL")
    print(f"  fixed   worst {wf:.2e} degC  -> {'PASS' if wf <= 0.2 else 'FAIL'}")
    print(f"\n  guards: {convert_rtd_channel_kelvin(-5.0)[1]}, "
          f"{convert_rtd_channel_kelvin(pt100_resistance_ohms(90.0))[1]}")
