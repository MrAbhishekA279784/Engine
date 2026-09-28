"""
M-01 — Excess kurtosis per axis.

GAP
---
File  : src/l2_digital_twin/vibration_processor.py
Class : AxisVibrationFeatures (lines 34-54)

The feature set contains mean, rms, peak, peak_to_peak, std_dev, variance,
crest_factor, dominant_frequency_hz, dominant_amplitude, spectral_centroid_hz,
spectral_bandwidth_hz, spectral_energy and three band energies.

It does NOT contain kurtosis. A repository-wide search for `kurt` returns zero
matches in src/.

WHY CREST FACTOR IS NOT A SUBSTITUTE
------------------------------------
The source framework described crest factor as a "kurtosis indicator for
impulsive events". They are different statistics and behave differently.

Crest factor is peak / RMS — two order statistics. One large sample moves it.
Kurtosis is a fourth central moment describing the whole distribution's tail
weight, so it responds to a POPULATION of small impacts.

A bearing entering early spalling produces many small impacts before any
single one is large. Kurtosis rises steadily through that phase; crest factor
stays near its healthy value until one impact finally dominates. Kurtosis is
therefore the earlier indicator, which is the point of monitoring at all.

Measured on 2048 samples (reproduce with `python3 M01_excess_kurtosis.py`,
seed 11 — the numbers below are this module's own printed output):

    signal              kurtosis   band      crest
    gaussian noise         0.033   NORMAL     3.62
    sparse impacts         2.716   NORMAL     6.83
    moderate impacts       4.195   WARNING    6.57
    dense impacts          7.380   ALARM      6.29

Read the two right-hand columns together. As impact DENSITY rises from sparse
to dense, kurtosis climbs 2.716 -> 7.380 and walks the signal through all
three bands. Over the same progression crest factor FALLS, 6.83 -> 6.29,
because it only ever tracked the single largest impact and the RMS underneath
it is rising. Crest factor is not a weaker version of kurtosis here — across
the phase that matters it moves the wrong way.

Requirements: SRD-FUN-062
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def excess_kurtosis(samples: Sequence[float]) -> float | None:
    """Excess kurtosis, Fisher definition (Gaussian -> 0).

        k = m4 / m2^2 - 3

    Returns None — NOT 0.0 — for a window shorter than 4 samples, non-finite
    input, or a constant signal (zero variance). It never raises, because this
    runs on every tick and a sensor dropout must not stop the pipeline.

    The None matters. 0.0 is the GAUSSIAN value, which `kurtosis_band`
    classifies as NORMAL. Returning it for a flatlined or dropped-out channel
    would report a dead sensor as a healthy one — the same silent-wrong-value
    pattern documented in M-07, where 0.0 SFC would read as perfect
    efficiency. "Unknown" and "normal" must stay distinguishable downstream.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return None
    mu = float(np.mean(arr))
    m2 = float(np.mean((arr - mu) ** 2))
    if m2 < 1e-18:
        return None                      # constant signal: a dead channel
    m4 = float(np.mean((arr - mu) ** 4))
    return m4 / (m2 * m2) - 3.0


def kurtosis_per_axis(
    x: Sequence[float], y: Sequence[float], z: Sequence[float]
) -> dict[str, float | None]:
    """Excess kurtosis for all three axes.

    ADD TO AxisVibrationFeatures as:  kurtosis: float
    ADD TO VibrationState        as:  kurtosis_x / _y / _z

    Per-axis matters: a bearing defect is usually strongest on the axis
    carrying the load, so a combined-magnitude kurtosis dilutes the signal.
    """
    return {
        "kurtosis_x": excess_kurtosis(x),
        "kurtosis_y": excess_kurtosis(y),
        "kurtosis_z": excess_kurtosis(z),
    }


def kurtosis_band(value: float | None) -> str:
    """Classify against the SRD-003 Appendix B band.

        below 3   NORMAL
        3 to 6    WARNING
        above 6   ALARM

    An unavailable value returns "UNKNOWN", which the dashboard must show as
    "no data" rather than as a green light.
    """
    if value is None or not math.isfinite(value):
        return "UNKNOWN"
    if value < 3.0:
        return "NORMAL"
    if value <= 6.0:
        return "WARNING"
    return "ALARM"


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 2048
    gaussian = rng.standard_normal(n)

    def crest(a: np.ndarray) -> float:
        return float(np.max(np.abs(a)) / np.sqrt(np.mean(a ** 2)))

    print("M-01  Excess kurtosis vs crest factor")
    print(f"{'signal':>28}{'kurtosis':>11}{'band':>10}{'crest':>9}")
    cases = [("gaussian noise", gaussian)]
    for rate, label in ((360, "sparse impacts"), (180, "moderate impacts"),
                        (60, "dense impacts")):
        sig = gaussian.copy()
        sig[::rate] += 6.0
        cases.append((label, sig))

    for label, sig in cases:
        k = excess_kurtosis(sig)
        ks = f"{k:.3f}" if k is not None else "None"
        print(f"{label:>28}{ks:>11}{kurtosis_band(k):>10}{crest(sig):>9.2f}")

    print("\n  kurtosis rises with impact DENSITY; crest factor tracks only")
    print("  the single largest impact, so it saturates early.")

    print("\n  guards return None, not 0.0 — 0.0 is the GAUSSIAN value and")
    print("  would classify a dead channel as NORMAL:")
    for label, sig in (("short window", [1.0, 2.0]),
                       ("constant (dead sensor)", [5.0] * 100),
                       ("contains nan", [1.0, float("nan"), 3.0, 4.0])):
        print(f"    {label:<24}{str(excess_kurtosis(sig)):>6}  -> "
              f"{kurtosis_band(excess_kurtosis(sig))}")
