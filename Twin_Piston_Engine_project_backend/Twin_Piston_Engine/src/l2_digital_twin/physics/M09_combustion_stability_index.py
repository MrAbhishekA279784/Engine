"""
M-09 — Combustion Stability Index.

GAP
---
Zero matches in src/ for `csi` or `combustion_stability`. The index is
defined in SRD-003 Appendix A.4 and given threshold bands in Appendix B, but
is never computed.

    CSI = EGT_std / EGT_mean + RPM_CoV% / 100 + |CHT_slope| / reference_slope

    below 0.15   NORMAL
    0.15 - 0.35  WARNING
    above 0.35   ALARM

CORRECTION TO THE APPENDIX A.4 FORM — CHT SLOPE MUST BE A RESIDUAL
------------------------------------------------------------------
The literal Appendix A.4 formula uses the RAW CHT slope. Implementing it
literally and running it produced an ALARM during a normal full-power climb:

    climb, healthy   CHT slope 0.55 K/s   ->  thermal term 1.10  ->  CSI 1.108

A climb is not unstable combustion. A rising CHT is the CORRECT behaviour of
a healthy engine whose power has just been increased, and the raw slope
cannot tell that apart from a thermal runaway.

The term therefore uses the slope RESIDUAL against the expected slope for the
current operating point:

    thermal term = |CHT_slope - expected_CHT_slope| / reference_slope

With `expected_cht_slope_k_per_s` left at 0.0 this reduces exactly to the
Appendix A.4 form, so steady-state behaviour is unchanged and only the
transient false alarm is removed. The expected slope comes from the thermal
model or, more simply, from the commanded power change.

This is a deliberate, documented departure from the written formula. It is
recorded here rather than silently applied because the SRD text will need the
same correction.

WHAT IT ADDS OVER THE INDIVIDUAL CHANNELS
-----------------------------------------
Each term on its own has an innocent explanation:

    EGT dispersion alone    -> could be one hot cylinder, or sensor noise
    speed variability alone -> could be turbulence or throttle movement
    CHT slope alone         -> could be a climb, or a cooling-flow change

Combustion that is genuinely unstable moves all three together. Summing them
makes the joint case visible while keeping each contribution inspectable,
which is what an operator needs when asked to justify a divert decision.

DIMENSIONAL DISCIPLINE
----------------------
Every term is made dimensionless BEFORE summation, per SRD-FUN-078:

    EGT_std / EGT_mean          coefficient of variation, dimensionless
    RPM_CoV% / 100              percentage to fraction
    |CHT_slope| / ref_slope     K/s divided by K/s

This matters. The audit found composite indices elsewhere in this codebase
summing terms of mixed units, which makes the total uninterpretable and its
threshold arbitrary — you cannot say what 0.35 means if the terms do not
share a scale.

Requirements: SRD-FUN-085, SRD-FUN-078, SRD-FUN-118 (Appendix B band)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

CSI_NORMAL_MAX = 0.15
CSI_WARNING_MAX = 0.35
DEFAULT_REFERENCE_CHT_SLOPE_K_S = 0.5


@dataclass(frozen=True)
class CSIResult:
    """CSI with its terms exposed for explainability (SRD-FUN-115)."""

    value: float | None
    egt_dispersion_term: float
    speed_variability_term: float
    thermal_slope_term: float
    band: str
    dominant_term: str
    valid: bool
    reason: str = ""


def combustion_stability_index(
    egt_std_k: float | None,
    egt_mean_k: float | None,
    crank_cov_pct: float | None,
    cht_slope_k_per_s: float | None,
    reference_cht_slope_k_per_s: float = DEFAULT_REFERENCE_CHT_SLOPE_K_S,
    expected_cht_slope_k_per_s: float = 0.0,
) -> CSIResult:
    """Composite combustion instability metric.

    ADD TO the diagnostics output as: combustion_stability_index

    Returns the index together with its three terms and the dominant one, so
    an alert can say WHICH aspect of combustion is unstable rather than only
    that the total crossed a line.

    `expected_cht_slope_k_per_s` defaults to 0.0, which reproduces the literal
    Appendix A.4 formula. Supply the expected slope for the current phase to
    suppress the climb false alarm documented in the module header.
    """
    for name, v in (("egt_std", egt_std_k), ("egt_mean", egt_mean_k),
                    ("crank_cov", crank_cov_pct), ("cht_slope", cht_slope_k_per_s)):
        if v is None or not math.isfinite(v):
            return CSIResult(None, 0.0, 0.0, 0.0, "UNKNOWN", "none", False,
                             f"{name} unavailable")
    if egt_mean_k <= 0.0:
        return CSIResult(None, 0.0, 0.0, 0.0, "UNKNOWN", "none", False,
                         "Non-positive mean EGT")
    if reference_cht_slope_k_per_s <= 0.0:
        return CSIResult(None, 0.0, 0.0, 0.0, "UNKNOWN", "none", False,
                         "Non-positive reference CHT slope")

    if not math.isfinite(expected_cht_slope_k_per_s):
        expected_cht_slope_k_per_s = 0.0

    dispersion = egt_std_k / egt_mean_k
    speed = crank_cov_pct / 100.0
    thermal = (abs(cht_slope_k_per_s - expected_cht_slope_k_per_s)
               / reference_cht_slope_k_per_s)
    total = dispersion + speed + thermal

    terms = {"egt_dispersion": dispersion,
             "speed_variability": speed,
             "thermal_slope": thermal}
    dominant = max(terms, key=terms.get)

    return CSIResult(total, dispersion, speed, thermal,
                     csi_band(total), dominant, True)


def csi_band(value: float | None) -> str:
    """Classify against SRD-003 Appendix B."""
    if value is None or not math.isfinite(value):
        return "UNKNOWN"
    if value < CSI_NORMAL_MAX:
        return "NORMAL"
    if value <= CSI_WARNING_MAX:
        return "WARNING"
    return "ALARM"


@dataclass
class EGTDispersionWindow:
    """Rolling per-cylinder EGT statistics for the dispersion term.

    SRD-003 Appendix A.4 specifies the standard deviation over a 10-second
    window. At the 10 Hz EGT rate that is 100 samples.
    """

    window: int = 100
    _samples: deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.window)

    def push_cylinder_set(self, egt_values_k: list[float]) -> None:
        """Push one tick's worth of per-cylinder EGT readings."""
        valid = [v for v in egt_values_k if v is not None and math.isfinite(v)]
        if valid:
            self._samples.append(sum(valid) / len(valid))

    def ready(self) -> bool:
        return len(self._samples) >= max(4, self.window // 4)

    def mean_and_std(self) -> tuple[float, float]:
        n = len(self._samples)
        if n < 2:
            return 0.0, 0.0
        mean = sum(self._samples) / n
        var = sum((s - mean) ** 2 for s in self._samples) / (n - 1)
        return mean, math.sqrt(var)

    def clear(self) -> None:
        self._samples.clear()


if __name__ == "__main__":
    print("M-09  Combustion Stability Index")
    print(f"{'scenario':>28}{'CSI':>8}{'band':>10}{'dominant term':>20}")
    #  label                      egt_std  egt_mean  cov%  slope  expected slope
    scenarios = [
        ("steady cruise",             4.0, 1100.0, 0.3, 0.01, 0.00),
        ("climb, healthy",            5.0, 1120.0, 0.4, 0.55, 0.55),
        ("climb, thermal runaway",    6.0, 1130.0, 0.5, 1.10, 0.55),
        ("turbulence (speed only)",   5.0, 1100.0, 8.0, 0.01, 0.00),
        ("one weak cylinder",        45.0, 1090.0, 1.2, 0.06, 0.00),
        ("unstable combustion",      60.0, 1100.0, 6.0, 0.30, 0.00),
    ]
    for label, std, mean, cov, slope, exp_slope in scenarios:
        r = combustion_stability_index(std, mean, cov, slope,
                                       expected_cht_slope_k_per_s=exp_slope)
        print(f"{label:>28}{r.value:8.3f}{r.band:>10}{r.dominant_term:>20}")

    print("\n  rows 2 and 3 have the same raw CHT slope behaviour class but")
    print("  different residuals. The literal Appendix A.4 formula alarms on")
    print("  BOTH; the residual form separates the healthy climb from the")
    print("  runaway, which is the whole point of the term.")

    literal = combustion_stability_index(5.0, 1120.0, 0.4, 0.55)
    print(f"\n  healthy climb under the literal formula: CSI {literal.value:.3f}"
          f" -> {literal.band}   (false alarm)")

    print("\n  the dominant term tells the operator WHICH aspect is unstable,")
    print("  so an alert can name a cause rather than only a number.")

    r = combustion_stability_index(60.0, 1100.0, 6.0, 0.30)
    print(f"\n  terms of the unstable case: dispersion {r.egt_dispersion_term:.3f}"
          f"  speed {r.speed_variability_term:.3f}"
          f"  thermal {r.thermal_slope_term:.3f}")
    print("  all three dimensionless before summation (SRD-FUN-078)")

    print(f"\n  missing input: {combustion_stability_index(None, 1100.0, 1.0, 0.1).reason}")
