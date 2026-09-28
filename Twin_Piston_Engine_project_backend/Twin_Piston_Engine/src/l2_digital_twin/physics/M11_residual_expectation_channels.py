"""
M-11 — Missing healthy-expectation channels in the residual engine.

GAP
---
File  : src/l2_digital_twin/residual_engine.py
Lines : 72-113, `HealthyExpectationModel`

The shipped model exposes five expectation methods:

    line  79   expected_egt_k
    line  86   expected_oil_pressure_pa
    line  99   expected_oil_temp_k
    line 104   expected_vibration_rms_m_s2
    line 109   expected_brake_power_kw

`grep -ni "cht|fuel" residual_engine.py` returns ZERO matches. Two monitored
channels therefore have no healthy expectation at all, and a third has one
that is structurally wrong.

A.  CYLINDER HEAD TEMPERATURE — no expectation, no residual
    CHT is the primary thermal-distress channel on an air-cooled aero piston
    engine and the one an operator acts on first. It is acquired (the Type-K
    CHT channel exists at sensor_inverse.py:115-116) and then never compared
    against anything. A channel with no expectation cannot produce a
    residual, so detonation, cooling-baffle damage and a lean cruise are all
    invisible to the residual layer.

B.  FUEL MASS FLOW — no expectation, no residual
    Without an expected fuel flow there is no way to say a given flow is
    high FOR THIS OPERATING POINT. This is also what makes SFC (M-07)
    trendable rather than merely bandable.

C.  PER-CYLINDER EGT — one expectation shared by four cylinders
    Lines 204-224 loop over `egt_cyl1..egt_cyl4` and pass the SAME `exp_egt`
    to all four (line 217). On a boxer four the rear cylinders sit in the
    downstream cooling flow and run measurably hotter when perfectly
    healthy. Feeding one expectation to four cylinders therefore produces a
    STANDING RESIDUAL on a healthy engine, which has two costs:

        - the false-alarm floor rises for the hot cylinders
        - the alarm threshold has to be widened to suppress it, which blinds
          the cold cylinders by exactly the amount of the offset

    Note what is NOT wrong here: the repo does not average the four channels
    or mask one cylinder with its siblings. Each is compared individually,
    which is the correct structure (SRD-FUN-091). Only the reference is
    shared. That is fixable with a per-cylinder offset, not a rewrite.

WHY EXPECTATIONS MUST NOT BE FITTED TO THE MEASURED CHANNEL
-----------------------------------------------------------
Every method below is a function of the OPERATING POINT only — speed, load,
ambient. None reads the channel it predicts. If an expectation is derived
from the measurement it is supposed to judge, the residual collapses toward
zero exactly when the fault grows, which is the failure mode this whole
layer exists to avoid.

Requirements: SRD-FUN-089, 090, 091, 092
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STD_PRESSURE_PA = 101325.0
STD_TEMP_K = 288.15

# Rotax 915 iS reference point
RATED_RPM = 5800.0
RATED_MAP_PA = 140000.0
RATED_POWER_KW = 105.0
GASOLINE_LHV_J_KG = 43.5e6


@dataclass(frozen=True)
class CHTExpectationConfig:
    """ADD TO HealthyBaselineConfig (core/config.py:278).

    Values are the steady-state cruise baseline for an air-cooled boxer four
    with a nominal cooling-air supply. They are a STARTING POINT to be
    re-fitted against the operator's own healthy flights, not published
    constants — which is why they live in config rather than in the code.
    """

    base_cht_k: float = 398.15          # 125 degC at light load, cooled
    k_cht_load_k: float = 65.0          # rise from idle to rated load
    k_cht_ambient: float = 0.85         # fraction of ambient rise passed through
    k_cht_airspeed: float = -18.0       # cooling-mass-flow credit at cruise
    scale_cht_k: float = 12.0           # residual normalisation scale


@dataclass(frozen=True)
class FuelExpectationConfig:
    """ADD TO HealthyBaselineConfig (core/config.py:278)."""

    baseline_sfc_kg_kwh: float = 0.312  # Rotax 915 iS published continuous
    idle_fuel_kg_s: float = 0.00035     # floor: an idling engine still burns
    scale_fuel_kg_s: float = 0.0004     # residual normalisation scale


# Per-cylinder EGT offsets in kelvin, cylinder 1..4.
# Boxer four, rear pair downstream in the cooling flow.
# MUST be re-fitted per airframe from healthy-flight data — the installation
# and cowling matter more than the engine here.
DEFAULT_EGT_CYLINDER_OFFSETS_K: tuple[float, float, float, float] = (
    -8.0, -3.0, 4.0, 7.0,
)


def expected_cht_k(
    rpm: float,
    map_pressure_pa: float,
    ambient_temp_k: float,
    airspeed_fraction: float = 0.0,
    cfg: CHTExpectationConfig | None = None,
) -> float | None:
    """Expected healthy cylinder head temperature.

    ADD TO HealthyExpectationModel (residual_engine.py:72) alongside
    `expected_egt_k` at line 79.

    Load is the product of the pressure ratio and the speed ratio, matching
    the form already used by `expected_oil_temp_k` at line 99 so the two
    thermal channels stay consistent with one another.

    `airspeed_fraction` is the cooling-air credit: 0.0 on the ground with no
    ram air, 1.0 at cruise. Without it the model reads a descent — low power,
    high airspeed — as a cooling fault.
    """
    cfg = cfg or CHTExpectationConfig()
    for v in (rpm, map_pressure_pa, ambient_temp_k):
        if v is None or not math.isfinite(v):
            return None
    if rpm <= 0.0:
        return None

    load = (max(0.0, map_pressure_pa) / RATED_MAP_PA) * (max(0.0, rpm) / RATED_RPM)
    load = min(load, 1.2)
    ambient_rise = (ambient_temp_k - STD_TEMP_K) * cfg.k_cht_ambient
    cooling = cfg.k_cht_airspeed * max(0.0, min(airspeed_fraction, 1.0))
    return cfg.base_cht_k + cfg.k_cht_load_k * load + ambient_rise + cooling


def expected_fuel_flow_kg_s(
    expected_brake_power_kw: float,
    cfg: FuelExpectationConfig | None = None,
) -> float | None:
    """Expected healthy fuel mass flow at this operating point.

    ADD TO HealthyExpectationModel (residual_engine.py:72) alongside
    `expected_brake_power_kw` at line 109.

        m_dot_fuel = P_expected * SFC_baseline / 3600

    Keyed on EXPECTED power, not measured power. Keying it on measured power
    would make the residual blind to the most important case: an engine
    burning the right fuel for the power it is producing but producing less
    power than it should. That is the signature of every efficiency loss, and
    it must appear as a POSITIVE fuel residual, not cancel out.
    """
    cfg = cfg or FuelExpectationConfig()
    if expected_brake_power_kw is None or not math.isfinite(expected_brake_power_kw):
        return None
    flow = (max(0.0, expected_brake_power_kw) * cfg.baseline_sfc_kg_kwh) / 3600.0
    return max(cfg.idle_fuel_kg_s, flow)


def expected_egt_per_cylinder_k(
    common_expected_egt_k: float,
    offsets_k: tuple[float, float, float, float] = DEFAULT_EGT_CYLINDER_OFFSETS_K,
) -> tuple[float, float, float, float] | None:
    """Per-cylinder expectations from the existing common expectation.

    REPLACES the shared `exp_egt` passed at residual_engine.py:217.

    Deliberately built on top of `expected_egt_k` rather than replacing it:
    the operating-point physics is unchanged and only the installation offset
    is added, so the existing model stays the single source of truth and this
    is a small, reviewable change.

    The offsets sum to zero by design, so the FLEET-MEAN expectation is
    untouched and only the distribution across cylinders changes.
    """
    if common_expected_egt_k is None or not math.isfinite(common_expected_egt_k):
        return None
    if len(offsets_k) != 4:
        return None
    return tuple(common_expected_egt_k + o for o in offsets_k)  # type: ignore[return-value]


def normalised_residual(observed: float | None,
                        expected: float | None,
                        scale: float) -> float | None:
    """Same contract as `ResidualEngine.compute_residual` (line 123).

    Reproduced here only so this module can self-test without importing the
    repo. When integrating, call the repo's own `compute_residual` — do not
    introduce a second residual definition.
    """
    if observed is None or expected is None or scale <= 0.0:
        return None
    if not math.isfinite(observed) or not math.isfinite(expected):
        return None
    return (observed - expected) / scale


if __name__ == "__main__":
    print("M-11  Missing healthy-expectation channels")

    print("\n  A. CHT expectation — currently absent, so CHT has no residual")
    print(f"{'condition':>32}{'expected CHT':>14}{'measured':>10}{'resid':>8}{'verdict':>10}")
    cases = [
        ("ground idle, hot day",      1800.0,  60000.0, 308.15, 0.0, 405.0),
        ("full-power climb",          5800.0, 140000.0, 288.15, 0.7, 462.0),
        ("cruise, healthy",           4800.0, 115000.0, 278.15, 1.0, 428.0),
        ("cruise, baffle damage",     4800.0, 115000.0, 278.15, 1.0, 461.0),
        ("cruise, lean detonation",   4800.0, 115000.0, 278.15, 1.0, 473.0),
    ]
    cfg = CHTExpectationConfig()
    for label, rpm, mapp, amb, air, measured in cases:
        exp = expected_cht_k(rpm, mapp, amb, air)
        res = normalised_residual(measured, exp, cfg.scale_cht_k)
        verdict = "OK" if abs(res) < 1.5 else ("WARN" if abs(res) < 3.0 else "ALARM")
        print(f"{label:>32}{exp:14.1f}{measured:10.1f}{res:8.2f}{verdict:>10}")

    print("\n  B. fuel-flow expectation — keyed on EXPECTED power, not measured")
    print(f"{'condition':>32}{'exp power':>11}{'exp fuel':>10}{'meas fuel':>11}{'resid':>8}")
    fcfg = FuelExpectationConfig()
    for label, exp_p, meas_fuel in (
            ("healthy cruise",            75.0, 0.00650),
            ("worn rings, same throttle", 75.0, 0.00745),
            ("idle",                       0.4, 0.00040)):
        ef = expected_fuel_flow_kg_s(exp_p)
        res = normalised_residual(meas_fuel, ef, fcfg.scale_fuel_kg_s)
        print(f"{label:>32}{exp_p:11.1f}{ef:10.5f}{meas_fuel:11.5f}{res:8.2f}")

    print("\n  row 2 is the case that matters: the engine is burning more fuel")
    print("  than the power it SHOULD make requires. Keying the expectation on")
    print("  measured power would cancel that residual to zero.")

    print("\n  C. per-cylinder EGT — one expectation for four cylinders")
    common = 1065.0
    per_cyl = expected_egt_per_cylinder_k(common)
    measured = (1057.0, 1062.0, 1069.0, 1072.0)   # a HEALTHY engine
    print(f"{'cyl':>5}{'measured':>10}{'shared exp':>12}{'resid':>8}"
          f"{'per-cyl exp':>13}{'resid':>8}")
    for i in range(4):
        r_shared = normalised_residual(measured[i], common, 50.0)
        r_percyl = normalised_residual(measured[i], per_cyl[i], 50.0)
        print(f"{i+1:>5}{measured[i]:10.1f}{common:12.1f}{r_shared:8.3f}"
              f"{per_cyl[i]:13.1f}{r_percyl:8.3f}")
    print(f"\n  offsets sum to {sum(DEFAULT_EGT_CYLINDER_OFFSETS_K):.1f} K — the fleet-mean")
    print("  expectation is unchanged, only its distribution across cylinders.")
    print("  the standing residual on a healthy engine goes to zero, which is")
    print("  what lets the alarm threshold be tightened rather than widened.")
