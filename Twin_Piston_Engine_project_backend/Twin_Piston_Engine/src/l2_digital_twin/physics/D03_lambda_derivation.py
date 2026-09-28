"""
D-03 — Lambda derivation: replace the hardcoded placeholder.

DEFECT
------
File : src/l2_digital_twin/sensor_inverse.py
Line : 209

    lambda_ch = _def_ch(1.0)

Lambda is modelled as a SENSOR INPUT — `lambda_sensor` in
`NormalizedSignalRecord` (signal_record.py:145) — and populated with a
constant placeholder. It is consumed downstream at
thermo_mechanical_twin.py:133-136:

    lam = record.lambda_sensor
    afr_val = lam.value * self.geom.stoichiometric_afr

WRONG VALUE
-----------
    lambda = 1.000   always, at every operating point
    AFR    = 14.700  always

CASCADE
-------
    - Mixture state can never change. LEAN and RICH are unreachable states.
    - Combustion efficiency is inert.
    - THE RING-WEAR SIGNATURE CANNOT APPEAR. Worn rings trap less air at
      unchanged fuelling, so lambda falls, combustion efficiency drops and
      specific fuel consumption rises. With lambda pinned, that entire causal
      chain is severed at its first link — and ring wear is one of the nine
      fault classes the system is specified to detect.
    - Any fault that changes trapped air is invisible for the same reason.

WHY NOT JUST READ A LAMBDA SENSOR
---------------------------------
The raw signal record (SRD-003 clause 3.1.1) has no lambda channel, and the
Rotax 915 iS does not publish one on the bus. Lambda is not measured on this
airframe — it must be derived, which is what SRD-FUN-028 requires.

DEPENDENCY
----------
Needs air mass flow from D04_air_mass_flow_speed_density.py. Applying this
file alone, while air is still `fuel x AFR`, produces lambda = 1.0 by
construction — the two fixes must go in together.

Requirements: SRD-FUN-028, SRD-FUN-029, SRD-FUN-034
"""

from __future__ import annotations

import math

STOICH_AFR_GASOLINE = 14.7


def air_fuel_ratio(air_mass_flow_kg_s: float, fuel_mass_flow_kg_s: float) -> float | None:
    """Measured AFR. Returns None when fuel flow is unusable."""
    if fuel_mass_flow_kg_s is None or air_mass_flow_kg_s is None:
        return None
    if not math.isfinite(fuel_mass_flow_kg_s) or not math.isfinite(air_mass_flow_kg_s):
        return None
    if fuel_mass_flow_kg_s <= 0.0:
        return None
    return air_mass_flow_kg_s / fuel_mass_flow_kg_s


def lambda_from_mass_flows(
    air_mass_flow_kg_s: float,
    fuel_mass_flow_kg_s: float,
    stoich_afr: float = STOICH_AFR_GASOLINE,
) -> float | None:
    """Derive lambda from the DERIVED air mass flow and measured fuel flow.

        lambda = (air / fuel) / AFR_stoich

    DROP-IN REPLACEMENT for sensor_inverse.py line 209 — except that lambda
    is no longer a sensor channel at all. See the integration map: the field
    should be re-tagged DERIVED, not REAL.

    Returns None rather than 1.0 when it cannot be derived. Returning 1.0
    would recreate the defect: a caller cannot distinguish "stoichiometric"
    from "unknown" if both are reported as 1.0.
    """
    afr = air_fuel_ratio(air_mass_flow_kg_s, fuel_mass_flow_kg_s)
    if afr is None or stoich_afr <= 0.0:
        return None
    return afr / stoich_afr


def combustion_efficiency(lambda_value: float | None) -> float | None:
    """Oxygen-limited combustion efficiency.

    Below stoichiometric there is not enough oxygen to burn the fuel, so the
    released fraction tracks lambda almost linearly. Slightly lean burns best;
    very lean approaches the misfire limit.

    This is the mechanism that makes ring blow-by visible in fuel
    consumption.
    """
    if lambda_value is None or not math.isfinite(lambda_value) or lambda_value <= 0.0:
        return None
    if lambda_value < 1.0:
        return min(0.97, 0.97 * lambda_value)
    if lambda_value < 1.35:
        return 0.97 - 0.05 * (lambda_value - 1.0)
    return max(0.55, 0.955 - 0.9 * (lambda_value - 1.35))


def classify_mixture(lambda_value: float | None,
                     lean_threshold: float = 1.03,
                     rich_threshold: float = 0.97) -> str:
    """Classify mixture from the DERIVED equivalence ratio.

    SRD-FUN-034 forbids inferring mixture from exhaust temperature alone.
    EGT peaks slightly LEAN of stoichiometric and falls on both sides, so one
    temperature maps to two mixture states. Acting on the wrong branch moves
    the engine in exactly the wrong direction — leaning an already-lean
    cylinder that was misread as rich.
    """
    if lambda_value is None or not math.isfinite(lambda_value) or lambda_value <= 0.0:
        return "UNKNOWN"
    if lambda_value > lean_threshold:
        return "LEAN"
    if lambda_value < rich_threshold:
        return "RICH"
    return "NOMINAL"


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from D04_air_mass_flow_speed_density import solve_air_flow

    DISP = 1.352e-3
    print("D-03  Lambda derived from air and fuel (shipped: pinned at 1.000)")
    print(f"{'fuel kg/s':>11}{'air kg/s':>10}{'AFR':>8}{'lambda':>9}"
          f"{'state':>9}{'eta_c':>8}{'shipped':>9}")
    for fuel in (0.0020, 0.0030, 0.0042, 0.0050, 0.0060):
        air, _, _, _, _ = solve_air_flow(5000.0, 120000.0, 101325.0, 288.15, DISP)
        lam = lambda_from_mass_flows(air, fuel)
        print(f"{fuel:11.4f}{air:10.5f}{air / fuel:8.2f}{lam:9.3f}"
              f"{classify_mixture(lam):>9}{combustion_efficiency(lam):8.3f}"
              f"{1.000:9.3f}")

    print("\n  ring-wear chain (20% of charge lost past worn rings):")
    for label, disp in (("healthy", DISP), ("worn", DISP * 0.80)):
        air, _, _, _, _ = solve_air_flow(5000.0, 120000.0, 101325.0, 288.15, disp)
        lam = lambda_from_mass_flows(air, 0.0042)
        print(f"    {label:8s} lambda {lam:.3f}  {classify_mixture(lam):7s} "
              f"eta_c {combustion_efficiency(lam):.3f}")
    print("    ^ severed by the shipped placeholder; restored here")

    print(f"\n  unknown is reported as None, not 1.0: "
          f"{lambda_from_mass_flows(0.05, 0.0)}")
