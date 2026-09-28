"""
M-07 — Specific fuel consumption.

GAP
---
Zero matches in src/ for `sfc`, `bsfc`, `specific_fuel`, `fuel_consumption`,
`kg_kwh` or `kg_per_kwh`. The quantity is never computed anywhere.

WHY THIS ONE HAS A VISIBLE CONSEQUENCE
--------------------------------------
SRD-003 Appendix B defines threshold bands for SFC:

    normal   0.28 - 0.33 kg/kW.h
    warning  0.33 - 0.38
    alarm    above 0.38

`SRD-FUN-118` requires every monitored parameter to be classified against
those bands. A band whose parameter is never computed is DEAD
CONFIGURATION — it looks like coverage in the config file and classifies
nothing at runtime.

WHY IT MATTERS DIAGNOSTICALLY
-----------------------------
SFC is the single best integrated measure of engine health, because it is
the ratio of what you spend to what you get. Almost every degradation mode
raises it:

    ring wear         less trapped air -> rich -> efficiency falls
    injector fouling  poor atomisation -> incomplete burn
    valve leakage     lost compression
    ignition degradation   late or weak burn

A parameter that responds to most fault modes is exactly what a trend-based
health index wants, which is why its absence is worth more than one line in
a gap list.

NOTE ON THE EXPECTED VALUE
--------------------------
The Rotax 915 iS published figure is near 0.312 kg/kW.h at continuous power.
The project's own forward model currently returns about 0.228 — a roughly
27 % disagreement that is tracked as OI-1 and must be resolved before any
performance claim is presented. Wiring this parameter in will make that
disagreement visible on the dashboard, which is a good reason to close OI-1
first.

Requirements: SRD-FUN-031, SRD-FUN-118 (Appendix B band)
"""

from __future__ import annotations

import math

# SRD-003 Appendix B
SFC_NORMAL_LO = 0.28
SFC_NORMAL_HI = 0.33
SFC_WARNING_HI = 0.38

MIN_POWER_KW_FOR_SFC = 1.0


def specific_fuel_consumption_kg_kwh(
    fuel_mass_flow_kg_s: float | None,
    brake_power_kw: float | None,
    min_power_kw: float = MIN_POWER_KW_FOR_SFC,
) -> float | None:
    """Brake specific fuel consumption in kg/kW.h.

        SFC = m_dot_fuel * 3600 / P_brake

    ADD TO MechanicalResults (thermo_mechanical_twin.py:89) as:
        sfc_kg_kwh: ProvenanceTaggedValue[float]

    Returns None rather than 0.0 when power is below `min_power_kw`. At idle,
    during shutdown, or whenever power is not derivable, SFC is genuinely
    undefined — and a zero would be read downstream as perfect efficiency,
    which is the opposite of the truth.
    """
    if fuel_mass_flow_kg_s is None or brake_power_kw is None:
        return None
    if not math.isfinite(fuel_mass_flow_kg_s) or not math.isfinite(brake_power_kw):
        return None
    if brake_power_kw <= min_power_kw or fuel_mass_flow_kg_s < 0.0:
        return None
    return (fuel_mass_flow_kg_s * 3600.0) / brake_power_kw


def sfc_band(sfc_kg_kwh: float | None) -> str:
    """Classify against SRD-003 Appendix B.

    Returns "UNKNOWN" for an undefined SFC — distinct from "NORMAL", so the
    dashboard can show "not available at idle" rather than a green light.
    """
    if sfc_kg_kwh is None or not math.isfinite(sfc_kg_kwh):
        return "UNKNOWN"
    if sfc_kg_kwh < SFC_NORMAL_LO:
        return "BELOW_EXPECTED"       # suspiciously good — check the derivation
    if sfc_kg_kwh <= SFC_NORMAL_HI:
        return "NORMAL"
    if sfc_kg_kwh <= SFC_WARNING_HI:
        return "WARNING"
    return "ALARM"


def sfc_degradation_pct(sfc_kg_kwh: float | None,
                        baseline_sfc_kg_kwh: float | None) -> float | None:
    """Percentage rise above the healthy baseline for this operating point.

    More useful than the absolute value for trend work: SFC varies with load
    and altitude, so a fixed band alone cannot tell a heavily loaded healthy
    engine from a lightly loaded degraded one.
    """
    if sfc_kg_kwh is None or baseline_sfc_kg_kwh is None:
        return None
    if baseline_sfc_kg_kwh <= 0.0:
        return None
    return 100.0 * (sfc_kg_kwh - baseline_sfc_kg_kwh) / baseline_sfc_kg_kwh


if __name__ == "__main__":
    print("M-07  Specific fuel consumption")
    print(f"{'fuel kg/s':>11}{'power kW':>10}{'SFC':>9}{'band':>16}"
          f"{'vs 0.312 base':>15}")
    cases = [(0.0065, 75.0), (0.0070, 75.0), (0.0078, 75.0),
             (0.0085, 75.0), (0.0042, 50.0), (0.0010, 0.5)]
    for fuel, power in cases:
        sfc = specific_fuel_consumption_kg_kwh(fuel, power)
        deg = sfc_degradation_pct(sfc, 0.312)
        sfc_s = f"{sfc:.3f}" if sfc is not None else "None"
        deg_s = f"{deg:+.1f}%" if deg is not None else "n/a"
        print(f"{fuel:11.4f}{power:10.1f}{sfc_s:>9}{sfc_band(sfc):>16}{deg_s:>15}")

    print("\n  idle returns None, not 0.0 — a zero would read as perfect")
    print("  efficiency. 'UNKNOWN' is distinct from 'NORMAL' downstream.")

    print("\n  Appendix B bands are currently dead configuration:")
    print("  nothing in src/ computes the parameter they classify.")
