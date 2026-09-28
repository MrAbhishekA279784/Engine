"""
M-06 — Dual-channel misfire confirmation gate.

GAP
---
File  : src/l2_digital_twin/misfire_classifier.py
Lines : 104-228, `MisfireDetector.evaluate`

The shipped detector fuses three weighted evidence terms:

    line 130-134  crank evidence     from rate_rpm_s (RPM RATE OF CHANGE)
    line 139-153  vibration evidence from OVERALL VIBRATION RMS
    line 182-206  EGT evidence       from per-cylinder deviation and drop rate

    line 222:
        if total_score >= self._cfg.misfire_confidence_threshold
           or (egt_evidence_val >= 0.8 and crank_evidence_val >= 0.5):

TWO PROBLEMS
------------
1. OVERALL RMS IS NOT A MISFIRE-SPECIFIC CHANNEL. It rises for rotating
   imbalance, bearing wear and load changes alike. A bearing fault therefore
   pushes the misfire score upward. The channel that IS specific — half-order
   band energy — does not exist in the repo (see M-02).

2. A WEIGHTED OR IS NOT THE SPECIFIED GATE. SRD-FUN-082 requires both
   channels to fire. That requirement is not bureaucratic: no public dataset
   contains a reciprocating-engine misfire, so neither channel can be
   validated alone. Requiring two PHYSICALLY INDEPENDENT mechanisms to agree
   is the strongest evidence obtainable in that absence, and it is the
   argument a reviewer will be shown. A weighted score over a non-specific
   channel gives that argument away.

Also: `rate_rpm_s` is an instantaneous RATE OF CHANGE. It responds to ordinary
throttle movement as strongly as to a missed firing event. SRD-FUN-080
specifies variability over a window spanning at least one complete firing
cycle — a dispersion statistic, not a derivative.

THE TWO CHANNELS ARE INDEPENDENT BY CONSTRUCTION
------------------------------------------------
    Channel A  crank-speed variability   torsional, from the crank pickup
    Channel B  half-order band energy    structural, from the accelerometer

Different transducers, different physical quantities, different signal paths.
A fault in one sensor cannot manufacture agreement in both.

VERDICTS
--------
    CONFIRMED    both channels exceed threshold
    UNCONFIRMED  exactly one does — reported WITH the channel that fired
    NORMAL       neither
    INVALID      a channel is unavailable; no verdict is issued

DEPENDENCY
----------
Channel B comes from M02_half_order_fraction.py, which in turn needs a real
sample window from B01_vibration_ring_buffer.py.

Requirements: SRD-FUN-080, 081, 082, 083
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class MisfireVerdict(str, Enum):
    NORMAL = "NORMAL"
    UNCONFIRMED = "UNCONFIRMED"
    CONFIRMED = "CONFIRMED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class MisfireAssessment:
    verdict: MisfireVerdict
    crank_cov_pct: float
    crank_fired: bool
    half_order_fraction: float
    half_order_fired: bool
    triggering_channel: str | None
    evidence: str
    crank_limit_pct: float = 0.0
    half_order_limit: float = 0.0


@dataclass
class CrankSpeedVariability:
    """Rolling crank-speed variability over a firing-cycle window.

    SRD-FUN-080 requires a window spanning at least one complete firing
    cycle — two crankshaft revolutions on a four-stroke. The default window
    holds several cycles so the statistic is stable.

    Replaces `rate_rpm_s` at misfire_classifier.py:130.
    """

    window: int = 32
    _samples: deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 4:
            raise ValueError("Window must hold at least 4 samples")
        self._samples = deque(maxlen=self.window)

    def push(self, rpm: float) -> None:
        if math.isfinite(rpm) and rpm > 0.0:
            self._samples.append(float(rpm))

    def ready(self) -> bool:
        return len(self._samples) >= max(4, self.window // 2)

    def coefficient_of_variation_pct(self) -> float:
        """100 * sigma / mean over the window."""
        n = len(self._samples)
        if n < 4:
            return 0.0
        mean = sum(self._samples) / n
        if mean <= 0.0:
            return 0.0
        var = sum((s - mean) ** 2 for s in self._samples) / (n - 1)
        return 100.0 * math.sqrt(var) / mean

    def clear(self) -> None:
        self._samples.clear()


class DualChannelMisfireGate:
    """Declare a misfire only when two independent channels agree.

    Thresholds reference the healthy baseline for the current operating point
    when one is supplied (SRD-FUN-081) rather than a fixed constant. A
    threshold tuned at one operating point is a liability at another.
    """

    def __init__(
        self,
        crank_cov_threshold_pct: float = 2.0,
        half_order_threshold: float = 0.03,
        baseline_multiplier: float = 3.0,
    ) -> None:
        self.crank_cov_threshold_pct = crank_cov_threshold_pct
        self.half_order_threshold = half_order_threshold
        self.baseline_multiplier = baseline_multiplier

    def evaluate(
        self,
        crank_cov_pct: float,
        half_order_fraction: float,
        baseline_cov_pct: float | None = None,
        baseline_half_order: float | None = None,
        inputs_valid: bool = True,
    ) -> MisfireAssessment:
        if not inputs_valid or not math.isfinite(crank_cov_pct) \
                or not math.isfinite(half_order_fraction):
            return MisfireAssessment(
                MisfireVerdict.INVALID, 0.0, False, 0.0, False, None,
                "One or both channels unavailable; no verdict issued.")

        crank_limit = (
            max(self.crank_cov_threshold_pct,
                baseline_cov_pct * self.baseline_multiplier)
            if baseline_cov_pct and baseline_cov_pct > 0.0
            else self.crank_cov_threshold_pct)
        half_limit = (
            max(self.half_order_threshold,
                baseline_half_order * self.baseline_multiplier)
            if baseline_half_order and baseline_half_order > 0.0
            else self.half_order_threshold)

        crank_fired = crank_cov_pct >= crank_limit
        half_fired = half_order_fraction >= half_limit

        if crank_fired and half_fired:
            verdict, chan = MisfireVerdict.CONFIRMED, "both"
            ev = (f"Crank-speed CoV {crank_cov_pct:.2f}% >= {crank_limit:.2f}% "
                  f"and half-order energy {half_order_fraction:.4f} >= "
                  f"{half_limit:.4f}. Two independent mechanisms agree.")
        elif crank_fired or half_fired:
            verdict = MisfireVerdict.UNCONFIRMED
            chan = "crank_speed_variability" if crank_fired else "half_order_energy"
            other = "half-order energy" if crank_fired else "crank-speed variability"
            ev = (f"Only {chan} exceeded its threshold; {other} did not. "
                  f"Reported as unconfirmed — one channel is not sufficient "
                  f"to declare a misfire (SRD-FUN-083).")
        else:
            verdict, chan = MisfireVerdict.NORMAL, None
            ev = (f"Crank-speed CoV {crank_cov_pct:.2f}% and half-order energy "
                  f"{half_order_fraction:.4f} both within limits.")

        return MisfireAssessment(
            verdict, crank_cov_pct, crank_fired, half_order_fraction,
            half_fired, chan, ev, crank_limit, half_limit)


if __name__ == "__main__":
    gate = DualChannelMisfireGate()
    print("M-06  Dual-channel misfire gate")
    print(f"{'scenario':>34}{'CoV %':>8}{'half':>8}{'verdict':>14}{'channel':>26}")
    scenarios = [
        ("healthy cruise", 0.3, 0.001),
        ("bearing fault (high RMS, no 0.5X)", 0.4, 0.002),
        ("throttle transient", 1.2, 0.004),
        ("rough air / turbulence", 2.5, 0.002),
        ("sensor noise on crank only", 5.0, 0.001),
        ("real misfire", 5.0, 0.20),
    ]
    for label, cov, half in scenarios:
        r = gate.evaluate(cov, half)
        print(f"{label:>34}{cov:8.1f}{half:8.3f}{r.verdict.value:>14}"
              f"{str(r.triggering_channel):>26}")

    print("\n  the shipped detector uses OVERALL RMS as its vibration term,")
    print("  so row 2 (a bearing fault) would raise its misfire score.")

    print("\n  crank variability: window statistic, not a rate")
    cov = CrankSpeedVariability(window=16)
    for _ in range(16):
        cov.push(4000.0)
    print(f"    steady 4000 rpm         CoV {cov.coefficient_of_variation_pct():.3f}%")
    cov.clear()
    for i in range(16):
        cov.push(4000.0 + (400.0 if i % 2 else -400.0))
    print(f"    oscillating +/-400 rpm  CoV {cov.coefficient_of_variation_pct():.3f}%")
    cov.clear()
    for i in range(16):
        cov.push(3000.0 + i * 60.0)       # steady acceleration
    print(f"    smooth acceleration     CoV {cov.coefficient_of_variation_pct():.3f}%")

    print(f"\n  invalid input: {gate.evaluate(float('nan'), 0.2).verdict.value}")
