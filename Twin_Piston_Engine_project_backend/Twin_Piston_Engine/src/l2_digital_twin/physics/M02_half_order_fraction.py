"""
M-02 — Half-order band energy fraction. THE MISFIRE CHANNEL.

GAP
---
Zero matches in src/ for `half_order`, `half-order` or any order-referenced
band. This is the most consequential single omission in the audit.

THE PHYSICS
-----------
A four-stroke fires each cylinder once every TWO crankshaft revolutions. A
cylinder that skips its firing event therefore imposes a disturbance that
repeats once per two revolutions — at half the shaft rotation frequency.

    f_1X   = rpm / 60          shaft rotation
    f_0.5X = rpm / 120         half order  <- misfire signature

Nothing else in a healthy reciprocating engine produces energy there, which
is what makes it specific.

WHY IT MATTERS MORE THAN ITS SIZE SUGGESTS
------------------------------------------
SRD-FUN-082 requires a misfire to be declared only when crank-speed
variability AND half-order energy both fire. That requirement exists because
no public dataset contains a reciprocating-engine misfire, so neither channel
can be validated on its own. Requiring two PHYSICALLY INDEPENDENT mechanisms
to agree is the strongest evidence obtainable in that absence — and it is the
argument a reviewer will be shown.

Without this band there is no second channel, so the shipped
`MisfireDetector` substitutes OVERALL VIBRATION RMS
(misfire_classifier.py:139-153). Overall RMS is not misfire-specific: it
rises for rotating imbalance, bearing wear and load changes alike, so a
bearing fault can push the misfire score up. See M-06.

MEASURED SEPARATION
-------------------
    healthy  0.0002
    misfire  0.4207        factor of 2354

WHY A FIXED-Hz BAND CANNOT DO THIS
----------------------------------
The shipped bands are anchored to absolute frequency (config.py:257-259,
100 / 500 / 1000 Hz). Half order sits at 16.7 Hz at 2000 rpm and 48.3 Hz at
5800 rpm — both inside the single "low" band, mixed in with 1X imbalance,
structural modes and everything else below 100 Hz. Anchoring to ORDERS is
what isolates it.

DEPENDENCY
----------
Requires a real sample window. See B01_vibration_ring_buffer.py — on the
current live path this function would receive one sample and return 0.0.

Requirements: SRD-FUN-065, SRD-FUN-069
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

HALF_ORDER_LO = 0.4
HALF_ORDER_HI = 0.6


def _power_spectrum(samples: Sequence[float], fs_hz: float,
                    window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum with a window applied."""
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n < 8 or fs_hz <= 0.0 or not np.all(np.isfinite(arr)):
        return np.empty(0), np.empty(0)
    detrended = arr - float(np.mean(arr))
    w = np.hanning(n) if window == "hann" else (
        np.hamming(n) if window == "hamming" else np.ones(n))
    spec = np.fft.rfft(detrended * w)
    return np.fft.rfftfreq(n, d=1.0 / fs_hz), np.abs(spec) ** 2


def order_band_fraction(
    samples: Sequence[float],
    fs_hz: float,
    rpm: float,
    order_lo: float,
    order_hi: float,
) -> float:
    """Fraction of spectral power inside an engine-order band.

    Band edges are given in ORDERS of shaft rotation and converted to Hz with
    the instantaneous engine speed, so the band tracks the engine instead of
    sitting at a fixed frequency.

    Returns 0.0 when the band is not representable at this speed and sample
    rate, rather than reporting a truncated band as if it were complete.
    """
    if rpm <= 0.0 or fs_hz <= 0.0:
        return 0.0
    f_1x = rpm / 60.0
    f_lo, f_hi = order_lo * f_1x, order_hi * f_1x
    if f_hi > fs_hz / 2.0:        # Nyquist — see M-05
        return 0.0

    freqs, power = _power_spectrum(samples, fs_hz)
    if freqs.size == 0:
        return 0.0
    total = float(np.sum(power[1:]))          # exclude DC
    if total <= 0.0:
        return 0.0
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(power[mask]) / total)


def half_order_fraction(samples: Sequence[float], fs_hz: float, rpm: float) -> float:
    """Energy fraction in the 0.4-0.6 x shaft-speed band.

    ADD TO AxisVibrationFeatures as: half_order_fraction: float
    ADD TO VibrationState        as: half_order_fraction

    Computed on the VERTICAL axis, which is firing-dominant on a boxer
    engine — that is where a missed firing event shows most strongly.
    """
    return order_band_fraction(samples, fs_hz, rpm, HALF_ORDER_LO, HALF_ORDER_HI)


def half_order_band(value: float) -> str:
    """Classify against the SRD-003 Appendix B band.

        below 0.03   NORMAL
        0.03 to 0.08 WARNING
        above 0.08   ALARM
    """
    if value < 0.03:
        return "NORMAL"
    if value <= 0.08:
        return "WARNING"
    return "ALARM"


if __name__ == "__main__":
    fs, n = 2048.0, 2048
    rng = np.random.default_rng(3)
    t = np.arange(n) / fs

    print("M-02  Half-order band energy fraction")
    print(f"{'rpm':>7}{'f_1X':>8}{'f_0.5X':>9}{'healthy':>10}{'misfire':>10}"
          f"{'band':>10}{'sep':>9}")
    for rpm in (2000.0, 3000.0, 4500.0, 5800.0):
        f1 = rpm / 60.0
        healthy = (np.sin(2 * np.pi * 2 * f1 * t)
                   + 0.2 * np.sin(2 * np.pi * f1 * t)
                   + 0.15 * rng.standard_normal(n))
        misfire = healthy + 0.9 * np.sin(2 * np.pi * 0.5 * f1 * t)
        h = half_order_fraction(healthy, fs, rpm)
        m = half_order_fraction(misfire, fs, rpm)
        print(f"{rpm:7.0f}{f1:8.1f}{0.5 * f1:9.1f}{h:10.4f}{m:10.4f}"
              f"{half_order_band(m):>10}{m / max(h, 1e-9):9.0f}x")

    print("\n  the band tracks rpm; a fixed 0-100 Hz band would contain")
    print("  half order, 1X imbalance and structural modes together.")

    print(f"\n  single sample (current live path): "
          f"{half_order_fraction([1.0], fs, 3000.0):.4f}  <- see B-01")
