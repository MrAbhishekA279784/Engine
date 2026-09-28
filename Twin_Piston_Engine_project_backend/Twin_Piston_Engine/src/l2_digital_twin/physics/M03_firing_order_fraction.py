"""
M-03 — Firing-order band energy fraction.

GAP
---
File  : src/l2_digital_twin/vibration_processor.py lines 176-186
Config: src/core/config.py:257-259

    freq_band_low_max_hz:  float = 100.0
    freq_band_mid_max_hz:  float = 500.0
    freq_band_high_max_hz: float = 1000.0

    low_mask  = (ac_freqs >= 0.0) & (ac_freqs < freq_band_low_max_hz)
    mid_mask  = (ac_freqs >= freq_band_low_max_hz) & (ac_freqs < ...)
    high_mask = ...

The three band energies are anchored to ABSOLUTE FREQUENCY. Engine orders are
anchored to SHAFT SPEED, so the two do not correspond.

THE PROBLEM, CONCRETELY
-----------------------
The firing order of a four-cylinder four-stroke is 2X (two firings per
crankshaft revolution).

    rpm 2000  ->  firing at  66.7 Hz   -> falls in the "low" band
    rpm 3500  ->  firing at 116.7 Hz   -> falls in the "mid" band
    rpm 5800  ->  firing at 193.3 Hz   -> falls in the "mid" band

The same physical phenomenon migrates between bands as the engine
accelerates. A trend line on `band_energy_low` therefore mixes "combustion
got weaker" with "the engine sped up", and no threshold on it can mean one
thing.

`dominant_order` at line 53 is closer in spirit but is a single scalar — the
order of the largest peak — not the energy in a band, so it cannot measure
how much of the signal is combustion.

MEASURED: order bands hold across the speed range
-------------------------------------------------
Firing-order fraction of a pure firing-order signal:

    rpm 2000 -> 0.981
    rpm 3500 -> 0.981
    rpm 5800 -> 0.984

A fixed-Hz band cannot produce that.

WHAT IT DETECTS
---------------
Falling firing-order energy at constant load means combustion is contributing
less of the total vibration — weak combustion, degraded ignition, or fuelling
loss. Paired with M-02 it separates "one cylinder stopped firing" (half order
rises) from "all cylinders are burning weakly" (firing order falls, half
order stays low).

Requirements: SRD-FUN-064, SRD-FUN-069
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

FIRING_BAND_TOLERANCE = 0.15     # +/-15 % of the firing order


def _power_spectrum(samples: Sequence[float], fs_hz: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum, Hann windowed."""
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n < 8 or fs_hz <= 0.0 or not np.all(np.isfinite(arr)):
        return np.empty(0), np.empty(0)
    spec = np.fft.rfft((arr - float(np.mean(arr))) * np.hanning(n))
    return np.fft.rfftfreq(n, d=1.0 / fs_hz), np.abs(spec) ** 2


def firing_order(n_cylinders: int = 4, strokes: int = 4) -> float:
    """Firing order in multiples of shaft rotation.

    Four-stroke: each cylinder fires once per two revolutions, so
    n_cylinders / 2. A four-cylinder four-stroke fires at 2X.
    """
    if strokes == 4:
        return n_cylinders / 2.0
    return float(n_cylinders)


def order_band_fraction(samples: Sequence[float], fs_hz: float, rpm: float,
                        order_lo: float, order_hi: float) -> float:
    """Fraction of spectral power inside an engine-order band."""
    if rpm <= 0.0 or fs_hz <= 0.0:
        return 0.0
    f_1x = rpm / 60.0
    f_lo, f_hi = order_lo * f_1x, order_hi * f_1x
    if f_hi > fs_hz / 2.0:          # Nyquist — see M-05
        return 0.0
    freqs, power = _power_spectrum(samples, fs_hz)
    if freqs.size == 0:
        return 0.0
    total = float(np.sum(power[1:]))
    if total <= 0.0:
        return 0.0
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(power[mask]) / total)


def firing_order_fraction(samples: Sequence[float], fs_hz: float, rpm: float,
                          n_cylinders: int = 4) -> float:
    """Energy fraction in the firing-order band.

    ADD TO AxisVibrationFeatures as: firing_order_fraction: float

    The +/-15 % width tolerates speed drift within the analysis window: at
    2048 samples and 2048 Hz the window is one second, over which a climbing
    engine can move several hundred rpm.
    """
    fo = firing_order(n_cylinders)
    return order_band_fraction(samples, fs_hz, rpm,
                               fo * (1.0 - FIRING_BAND_TOLERANCE),
                               fo * (1.0 + FIRING_BAND_TOLERANCE))


def order_spectrum(samples: Sequence[float], fs_hz: float, rpm: float,
                   orders: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 4.0)
                   ) -> dict[str, float]:
    """Energy fraction at several named orders at once.

    Useful for the diagnostics endpoint: 1X is imbalance, 2X is firing on a
    four-cylinder, 0.5X is misfire, and the ratio between them says more than
    any one alone.
    """
    out: dict[str, float] = {}
    for o in orders:
        out[f"order_{o:g}X"] = order_band_fraction(
            samples, fs_hz, rpm, o * 0.85, o * 1.15)
    return out


if __name__ == "__main__":
    fs, n = 2048.0, 2048
    rng = np.random.default_rng(5)
    t = np.arange(n) / fs

    print("M-03  Firing-order band vs the shipped fixed-Hz bands")
    print(f"{'rpm':>7}{'firing Hz':>11}{'shipped band':>14}{'order frac':>12}")
    for rpm in (2000.0, 3500.0, 5800.0):
        f_fire = 2.0 * rpm / 60.0
        sig = np.sin(2 * np.pi * f_fire * t) + 0.1 * rng.standard_normal(n)
        which = "low (0-100)" if f_fire < 100 else "mid (100-500)"
        print(f"{rpm:7.0f}{f_fire:11.1f}{which:>14}"
              f"{firing_order_fraction(sig, fs, rpm):12.3f}")

    print("\n  the shipped band changes between 2000 and 3500 rpm;")
    print("  the order band holds ~0.98 throughout.")

    print("\n  full order spectrum at 3000 rpm, healthy vs misfire:")
    f1 = 50.0
    healthy = np.sin(2 * np.pi * 2 * f1 * t) + 0.2 * np.sin(2 * np.pi * f1 * t) \
        + 0.15 * rng.standard_normal(n)
    misfire = healthy + 0.9 * np.sin(2 * np.pi * 0.5 * f1 * t)
    hs = order_spectrum(healthy, fs, 3000.0)
    ms = order_spectrum(misfire, fs, 3000.0)
    print(f"{'order':>10}{'healthy':>10}{'misfire':>10}")
    for k in hs:
        print(f"{k:>10}{hs[k]:10.4f}{ms[k]:10.4f}")
