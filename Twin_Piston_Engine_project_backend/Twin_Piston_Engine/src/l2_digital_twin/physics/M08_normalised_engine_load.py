"""
M-08 — Normalised engine load.

GAP
---
Zero matches in src/ for `engine_load`, `load_frac`, `load_pct`,
`load_ratio` or `normalized_load`.

WHY A SEPARATE PARAMETER IS NEEDED
----------------------------------
Almost every health threshold in the system is load-dependent. CHT, EGT, oil
temperature, vibration RMS and SFC all rise with load on a healthy engine.
Without a normalised load figure, a threshold has to be either:

    - set high enough for full power, and therefore blind at cruise, or
    - set for cruise, and therefore alarming continuously in the climb

The healthy-expectation models in `residual_engine.py` already need this:
`expected_oil_temp_k` takes load implicitly through the operating point, but
there is no first-class normalised value to key those expectations on.

Load is also what separates "this engine is working hard" from "this engine
is working badly" — the same 200 degC CHT is normal at takeoff power and a
fault at idle.

DEFINITION
----------
    load = P_brake / P_rated

Rated power for the Rotax 915 iS is 105 kW (takeoff, 5800 rpm). The value is
clipped at 1.2 rather than 1.0 because the manufacturer states delivered
power may exceed the published figure by up to 6 %, and a derivation with its
own tolerance can legitimately land slightly above unity. Clipping at exactly
1.0 would hide a derivation error; clipping at 1.2 lets it show while keeping
the value bounded.

Requirements: SRD-FUN-031
"""

from __future__ import annotations

import math

ROTAX_915IS_RATED_POWER_KW = 104.0  # peak, flyrotax.com/products/915-is-a-isc-a (was 105.0)
LOAD_CLIP_MAX = 1.2


def normalised_engine_load(
    brake_power_kw: float | None,
    rated_power_kw: float = ROTAX_915IS_RATED_POWER_KW,
) -> float | None:
    """Engine load as a fraction of rated output.

    ADD TO MechanicalResults (thermo_mechanical_twin.py:89) as:
        engine_load: ProvenanceTaggedValue[float]

    Returns None when power is unavailable, so downstream code can tell
    "no load information" from "zero load".
    """
    if rated_power_kw is None or rated_power_kw <= 0.0:
        return None
    if brake_power_kw is None or not math.isfinite(brake_power_kw):
        return None
    return max(0.0, min(brake_power_kw / rated_power_kw, LOAD_CLIP_MAX))


def load_band(load: float | None) -> str:
    """Coarse operating regime, useful for keying baselines and advisories."""
    if load is None or not math.isfinite(load):
        return "UNKNOWN"
    if load < 0.15:
        return "IDLE"
    if load < 0.45:
        return "LIGHT"
    if load < 0.75:
        return "CRUISE"
    if load <= 1.02:
        return "HIGH"
    return "ABOVE_RATED"


def load_adjusted_threshold(
    base_threshold: float,
    load: float | None,
    load_sensitivity: float = 0.0,
) -> float:
    """Scale a threshold with load.

        threshold = base * (1 + sensitivity * load)

    Use this instead of a single fixed limit for any load-dependent channel.
    A `load_sensitivity` of 0.0 reproduces the current fixed-threshold
    behaviour, so it can be introduced channel by channel without changing
    anything until each one is tuned.
    """
    if load is None or not math.isfinite(load):
        return base_threshold
    return base_threshold * (1.0 + load_sensitivity * load)


if __name__ == "__main__":
    print("M-08  Normalised engine load")
    print(f"{'power kW':>10}{'load':>8}{'band':>14}"
          f"{'CHT limit (sens 0.25)':>24}")
    for power in (8.0, 30.0, 55.0, 80.0, 99.0, 105.0, 111.0):
        load = normalised_engine_load(power)
        limit = load_adjusted_threshold(200.0, load, load_sensitivity=0.25)
        print(f"{power:10.1f}{load:8.3f}{load_band(load):>14}{limit:>24.1f}")

    print("\n  a single fixed CHT limit of 200 degC is either blind at")
    print("  takeoff power or alarming continuously at cruise.")

    print(f"\n  above-rated is visible, not hidden: "
          f"{normalised_engine_load(111.0):.3f} -> "
          f"{load_band(normalised_engine_load(111.0))}")
    print(f"  no power available: {normalised_engine_load(None)}")
