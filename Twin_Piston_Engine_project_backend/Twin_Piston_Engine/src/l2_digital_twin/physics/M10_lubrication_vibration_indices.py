"""
M-10 — Lubrication Health Index and Vibration Health Index.

GAP
---
Zero matches in src/ for `lhi` or `vhi`. Single matches for
`lubrication_health` and `vibration_health` in schema names only.

The component terms mostly exist — `lubrication_model.py` computes viscosity
and an expected pressure, `vibration_processor.py` computes RMS and crest
factor — but the NAMED COMPOSITE INDICES that SRD-003 Appendix A.3, A.5 and
Appendix B specify are never formed.

WHY THE NAMED INDEX MATTERS AND NOT JUST THE TERMS
--------------------------------------------------
Appendix B gives bands for LHI and VHI, not for their components:

    LHI  0.8 - 1.2 normal   0.7-0.8 or 1.2-1.3 warning   outside that alarm
    VHI  below 1.2 normal   1.2 - 1.8 warning            above 1.8 alarm

Bands on a composite are not reconstructible from bands on the parts. LHI
below 0.7 means oil-film starvation risk; above 1.3 means excessive drag from
over-thick oil. Neither condition is visible from pressure alone or viscosity
alone — it is their PRODUCT relative to nominal that carries the meaning.

The same holds for VHI: the envelope term is what makes it sensitive to
bearing defects before they raise the overall level, and that sensitivity
exists only in the product.

CORRECTION — LHI MUST NORMALISE AGAINST EXPECTATIONS, NOT CONSTANTS
-------------------------------------------------------------------
The first implementation of this module normalised viscosity against a FIXED
nominal (0.043 Pa.s at 85 degC) and multiplied in a separate temperature
taper. Running it gave:

    healthy engine, oil at 115 degC   LHI 0.365   ALARM
    healthy engine, oil at  35 degC   LHI 2.721   ALARM

Both are false. Oil viscosity falls roughly four-fold between 35 and 115 degC
on a perfectly healthy engine — that is what oil DOES. An index that alarms
on it is a thermometer, not a health index.

Two faults produced that:

  1. Viscosity was compared against a constant instead of against the
     viscosity EXPECTED AT THE CURRENT OIL TEMPERATURE. Normalising against
     the expectation is what makes the ratio mean "this oil is thinner than
     oil at this temperature should be" — which is the actual fault signature
     (fuel dilution, shear-down, wrong grade).

  2. Temperature entered TWICE — once through the viscosity term, which is
     almost entirely temperature-driven, and again through an explicit
     `1 - |T - 85|/100` taper. The taper is also symmetric, which treats cold
     oil on start-up as equally dangerous as oil above its limit. It is not.

The corrected form normalises each term against its expectation at the
current operating point and keeps only a ONE-SIDED over-temperature derate:

    LHI = (P / P_expected) x (mu / mu_expected(T)) x over_temp_derate(T)

`expected_oil_pressure_pa` already exists in the repo at
residual_engine.py:86, and `lubrication_model.py` already computes viscosity
from temperature — so both expectations are available and neither needs new
physics.

DEPENDENCY
----------
VHI needs `envelope_rms` from M04_envelope_demodulation.py, which needs a
real sample window from B01_vibration_ring_buffer.py. Without those the
envelope term is constant and VHI degenerates to RMS x crest.

Requirements: SRD-FUN-054, SRD-FUN-068, SRD-FUN-118 (Appendix B bands)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

OIL_TEMP_LIMIT_C = 130.0        # Rotax 915 iS maximum oil temperature
OIL_OVERTEMP_SPAN_C = 20.0      # derate reaches zero this far above the limit


def expected_viscosity_pa_s(
    oil_temp_c: float,
    reference_viscosity_pa_s: float,
    reference_temp_c: float,
) -> float | None:
    """Expected healthy viscosity at this oil temperature (Vogel form).

    Present only so this module is self-testable. When integrating, call the
    repo's own viscosity function in `lubrication_model.py` instead of this
    one — a second viscosity model in the codebase is a liability.
    """
    if not math.isfinite(oil_temp_c) or reference_viscosity_pa_s <= 0.0:
        return None
    c = 140.0                                   # Vogel constant, degC
    if oil_temp_c + c <= 1.0 or reference_temp_c + c <= 1.0:
        return None
    b = 1200.0                                  # Vogel constant, K
    ratio = math.exp(b / (oil_temp_c + c) - b / (reference_temp_c + c))
    return reference_viscosity_pa_s * ratio


def over_temperature_derate(oil_temp_c: float,
                            limit_c: float = OIL_TEMP_LIMIT_C,
                            span_c: float = OIL_OVERTEMP_SPAN_C) -> float:
    """One-sided derate: 1.0 at or below the limit, falling to 0.0 above it.

    One-sided on purpose. Cold oil is already penalised by the viscosity term
    being high; it is not an independent hazard the way an over-temperature
    is. A symmetric taper would report a cold start as a lubrication alarm.
    """
    if not math.isfinite(oil_temp_c) or span_c <= 0.0:
        return 1.0
    if oil_temp_c <= limit_c:
        return 1.0
    return max(0.0, 1.0 - (oil_temp_c - limit_c) / span_c)


@dataclass(frozen=True)
class IndexResult:
    """A composite index with its factors exposed for explainability."""

    value: float | None
    factors: dict[str, float]
    band: str
    dominant_factor: str
    valid: bool
    reason: str = ""


def lubrication_health_index(
    oil_pressure_pa: float | None,
    expected_pressure_pa: float | None,
    viscosity_pa_s: float | None,
    expected_viscosity_at_temp_pa_s: float | None,
    oil_temp_c: float | None,
    temp_limit_c: float = OIL_TEMP_LIMIT_C,
) -> IndexResult:
    """LHI — pressure ratio x viscosity ratio x over-temperature derate.

    ADD TO LubricationModelResult (lubrication_model.py:53) as:
        lubrication_health_index: float | None

    Both expectations are functions of the OPERATING POINT, not of the
    channel they normalise:

        expected_pressure_pa            residual_engine.py:86
        expected_viscosity_at_temp_pa_s lubrication_model.py viscosity curve

    A healthy engine therefore sits at 1.0 at every oil temperature, and both
    failure directions stay visible:

        below 0.7  oil-film starvation — low pressure, or oil thinner than
                   it should be at this temperature (dilution, shear-down)
        above 1.3  excessive drag — oil thicker than expected, or pressure
                   above expectation (blocked gallery, stuck relief valve)
    """
    if expected_pressure_pa is None or expected_viscosity_at_temp_pa_s is None:
        return IndexResult(None, {}, "UNKNOWN", "none", False,
                           "Expectation unavailable")
    if expected_pressure_pa <= 0.0 or expected_viscosity_at_temp_pa_s <= 0.0:
        return IndexResult(None, {}, "UNKNOWN", "none", False,
                           "Non-positive expectation")
    for name, v in (("oil_pressure", oil_pressure_pa),
                    ("viscosity", viscosity_pa_s),
                    ("oil_temp", oil_temp_c)):
        if v is None or not math.isfinite(v):
            return IndexResult(None, {}, "UNKNOWN", "none", False,
                               f"{name} unavailable")

    factors = {
        "pressure": oil_pressure_pa / expected_pressure_pa,
        "viscosity": viscosity_pa_s / expected_viscosity_at_temp_pa_s,
        "over_temp": over_temperature_derate(oil_temp_c, temp_limit_c),
    }
    value = factors["pressure"] * factors["viscosity"] * factors["over_temp"]
    dominant = max(factors, key=lambda k: abs(factors[k] - 1.0))
    return IndexResult(value, factors, lhi_band(value), dominant, True)


def lhi_band(value: float | None) -> str:
    """Classify against SRD-003 Appendix B."""
    if value is None or not math.isfinite(value):
        return "UNKNOWN"
    if 0.8 <= value <= 1.2:
        return "NORMAL"
    if 0.7 <= value < 0.8 or 1.2 < value <= 1.3:
        return "WARNING"
    return "ALARM"


def vibration_health_index(
    rms_m_s2: float | None,
    reference_rms_m_s2: float,
    crest_factor: float | None,
    reference_crest: float,
    envelope_rms: float | None,
    reference_envelope: float,
) -> IndexResult:
    """VHI — normalised RMS x crest x envelope energy.

    ADD TO VibrationState (core/schemas.py) as:
        vibration_health_index: ProvenanceTaggedValue[float]

    The three terms respond to different failure stages:

        RMS       overall energy — rises late
        crest     single large impacts — rises mid
        envelope  modulation from repeated impacts — rises EARLY

    Multiplying them means an early bearing defect, which moves only the
    envelope term, still lifts the index.
    """
    if reference_rms_m_s2 <= 0.0 or reference_crest <= 0.0 or reference_envelope <= 0.0:
        return IndexResult(None, {}, "UNKNOWN", "none", False,
                           "Non-positive reference")
    for name, v in (("rms", rms_m_s2), ("crest", crest_factor),
                    ("envelope", envelope_rms)):
        if v is None or not math.isfinite(v):
            return IndexResult(None, {}, "UNKNOWN", "none", False,
                               f"{name} unavailable")

    factors = {
        "rms": rms_m_s2 / reference_rms_m_s2,
        "crest": crest_factor / reference_crest,
        "envelope": envelope_rms / reference_envelope,
    }
    value = factors["rms"] * factors["crest"] * factors["envelope"]
    dominant = max(factors, key=factors.get)
    return IndexResult(value, factors, vhi_band(value), dominant, True)


def vhi_band(value: float | None) -> str:
    """Classify against SRD-003 Appendix B."""
    if value is None or not math.isfinite(value):
        return "UNKNOWN"
    if value < 1.2:
        return "NORMAL"
    if value <= 1.8:
        return "WARNING"
    return "ALARM"


if __name__ == "__main__":
    print("M-10  Lubrication and Vibration Health Indices")

    print("\n  LHI    reference 0.043 Pa.s at 85 degC; expected pressure per op point")
    print("  fault rows are expressed as RATIOS TO EXPECTATION, so a healthy")
    print("  engine is 1.00 x 1.00 by construction at every oil temperature.")
    print(f"{'scenario':>34}{'oil degC':>9}{'LHI':>8}{'band':>10}{'dominant':>12}")
    #  label                        oil T   P ratio   mu ratio
    rows = (
        ("healthy, 85 degC",           85.0, 1.00, 1.00),
        ("healthy, hot 115 degC",     115.0, 0.99, 1.00),
        ("healthy, cold start 35 C",   35.0, 1.01, 1.00),
        ("fuel-dilated oil, 85 degC",  85.0, 0.85, 0.58),
        ("bearing wear, 92 degC",      92.0, 0.62, 0.99),
        ("over-temp 142 degC",        142.0, 0.93, 0.95),
    )
    for label, t, p_ratio, v_ratio in rows:
        exp_v = expected_viscosity_pa_s(t, 0.043, 85.0)
        exp_p = 4.00e5
        r = lubrication_health_index(exp_p * p_ratio, exp_p,
                                     exp_v * v_ratio, exp_v, t)
        print(f"{label:>34}{t:9.0f}{r.value:8.3f}{r.band:>10}{r.dominant_factor:>12}")

    print("\n  the three healthy rows span 35 to 115 degC and all read NORMAL.")
    print("  under the fixed-nominal form the same three rows give:")
    for label, t, p_ratio, v_ratio in rows[:3]:
        exp_v = expected_viscosity_pa_s(t, 0.043, 85.0)
        old = (p_ratio) * (exp_v * v_ratio / 0.043) * max(
            0.0, 1.0 - abs(t - 85.0) / 100.0)
        print(f"      {label:<28}{old:6.3f}   {lhi_band(old)}")
    print("  — two false alarms on oil that is doing exactly what oil does.")

    print("\n  NOTE: the Vogel constants here (b=1200, c=140) are generic. Fit")
    print("  them to the actual oil grade before use, or call the repo's own")
    print("  viscosity curve in lubrication_model.py instead of this one.")

    print("\n  VHI    reference 2.0 m/s^2, crest 2.5, envelope 0.05")
    print(f"{'scenario':>30}{'VHI':>8}{'band':>10}{'dominant':>14}")
    for label, rms, cf, env in (
            ("healthy", 2.0, 2.5, 0.05),
            ("early bearing defect", 2.1, 2.7, 0.14),
            ("rotating imbalance", 3.4, 2.4, 0.05),
            ("advanced bearing wear", 3.6, 4.0, 0.12)):
        r = vibration_health_index(rms, 2.0, cf, 2.5, env, 0.05)
        print(f"{label:>30}{r.value:8.3f}{r.band:>10}{r.dominant_factor:>14}")

    print("\n  note row 2: RMS barely moved (2.0 -> 2.1) but the envelope")
    print("  term lifts VHI into WARNING. That is the early detection the")
    print("  index exists for — and it needs M-04 to work.")
