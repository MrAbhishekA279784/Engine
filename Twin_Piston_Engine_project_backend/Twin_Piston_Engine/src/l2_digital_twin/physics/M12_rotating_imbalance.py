"""
M-12 — Rotating imbalance: 1X velocity dominance and lateral/vertical split
(SRD-FUN-070; Prompt 15).

A rotating imbalance (e.g. the propeller) produces a force m e omega^2 that
rotates with the shaft, so the vibration is at 1X of THAT shaft (the
propeller shaft turns at crank / gear ratio) and, with laterally softer
mounts, is lateral-dominant. At the propeller frequency (tens of Hz) it is
small in acceleration next to the firing orders but dominant in VELOCITY,
which is how imbalance is judged (ISO 10816 / 20816 use velocity):

    velocity power spectrum  P_v(f) = P_a(f) / (2 pi f)^2,  f >= f_min
    one_x_velocity_fraction = P_v in [0.9, 1.1] x f_1X  /  total P_v
    lateral_vertical_ratio  = sqrt(P_a,lateral(1X band) / P_a,vertical(1X band))

f_min excludes the near-DC bins, where the 1/f^2 weighting would amplify
sensor noise and detrending residue.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.l2_digital_twin.physics.M02_half_order_fraction import _power_spectrum

BAND_REL_HALF_WIDTH = 0.1   # 1X band = [0.9, 1.1] x f_1X
VELOCITY_F_MIN_HZ = 5.0


def _band_mask(freqs: np.ndarray, f_center: float, rel: float) -> np.ndarray:
    return (freqs >= (1.0 - rel) * f_center) & (freqs <= (1.0 + rel) * f_center)


def one_x_velocity_fraction(samples: Sequence[float], fs_hz: float, f_1x_hz: float,
                            f_min_hz: float = VELOCITY_F_MIN_HZ) -> float | None:
    """Share of the velocity-weighted energy in the 1X band; None if not computable."""
    freqs, power = _power_spectrum(samples, fs_hz)
    if freqs.size == 0 or not (f_1x_hz > f_min_hz and f_1x_hz * (1.0 + BAND_REL_HALF_WIDTH) < fs_hz / 2.0):
        return None
    keep = freqs >= f_min_hz
    pv = power[keep] / (2.0 * math.pi * freqs[keep]) ** 2
    total = float(np.sum(pv))
    if total <= 0.0:
        return None
    return float(np.sum(pv[_band_mask(freqs[keep], f_1x_hz, BAND_REL_HALF_WIDTH)]) / total)


def one_x_band_power(samples: Sequence[float], fs_hz: float, f_1x_hz: float) -> float | None:
    freqs, power = _power_spectrum(samples, fs_hz)
    if freqs.size == 0 or f_1x_hz * (1.0 + BAND_REL_HALF_WIDTH) >= fs_hz / 2.0:
        return None
    return float(np.sum(power[_band_mask(freqs, f_1x_hz, BAND_REL_HALF_WIDTH)]))


def lateral_vertical_ratio(lateral: Sequence[float], vertical: Sequence[float], fs_hz: float,
                           f_1x_hz: float) -> float | None:
    p_lat = one_x_band_power(lateral, fs_hz, f_1x_hz)
    p_vert = one_x_band_power(vertical, fs_hz, f_1x_hz)
    if p_lat is None or p_vert is None or p_vert <= 0.0:
        return None
    return math.sqrt(p_lat / p_vert)
