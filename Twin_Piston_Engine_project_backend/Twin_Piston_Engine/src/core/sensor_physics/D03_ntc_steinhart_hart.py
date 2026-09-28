"""
D-03 — NTC thermistor: Steinhart-Hart.

    1 / T = A + B ln R + C (ln R)^3          (T in K, R in ohm)

Inverse (L2): direct evaluation. Forward (simulator): the cubic in x = ln R,
C x^3 + B x + (A - 1/T) = 0, has one real root for B, C > 0 (Cardano).

Coefficients come from config (sensor_calibration.ntc_sh_*, and the
simulator's own copy); they must be the fitted sensor's datasheet values.
"""

from __future__ import annotations

import math

KELVIN_OFFSET = 273.15


def ntc_temp_k(r_ohm: float, a: float, b: float, c: float) -> float:
    """Temperature [K] of an NTC with resistance r_ohm (> 0)."""
    if not (r_ohm > 0.0 and math.isfinite(r_ohm)):
        raise ValueError(f"NTC resistance must be positive and finite, got {r_ohm}")
    x = math.log(r_ohm)
    return 1.0 / (a + b * x + c * x ** 3)


def ntc_temp_c(r_ohm: float, a: float, b: float, c: float) -> float:
    return ntc_temp_k(r_ohm, a, b, c) - KELVIN_OFFSET


def ntc_resistance_ohm(temp_c: float, a: float, b: float, c: float) -> float:
    """Resistance [ohm] at temp_c, solving the Steinhart-Hart cubic in ln R."""
    t_k = temp_c + KELVIN_OFFSET
    if t_k <= 0.0:
        raise ValueError(f"temperature below absolute zero: {temp_c} C")
    y = (a - 1.0 / t_k) / c
    p = b / c
    half_y = y / 2.0
    root = math.sqrt((p / 3.0) ** 3 + half_y ** 2)
    x = math.copysign(abs(-half_y + root) ** (1.0 / 3.0), -half_y + root) \
        + math.copysign(abs(-half_y - root) ** (1.0 / 3.0), -half_y - root)
    return math.exp(x)
