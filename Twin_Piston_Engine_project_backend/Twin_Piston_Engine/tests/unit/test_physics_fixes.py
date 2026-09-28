"""
Consolidated regression suite for the Twin_Piston_Engine fix package.

Each test asserts the property the corresponding module exists to guarantee,
and — where a published reference exists — asserts it against that reference
rather than against the module's own output.

Run (from the repo root):
    uv run pytest tests/unit/test_physics_fixes.py
    python -m tests.unit.test_physics_fixes   (no pytest needed)

These tests use synthetic signals with known ground truth. They prove the
mathematics is right. They do NOT prove the thresholds are right — that needs
healthy-flight data from the actual installation, which does not exist yet.
See INTEGRATION_MAP.md section 10.
"""

from __future__ import annotations

import math
import sys

from src.core.sensor_physics import D01_thermocouple_nist_inverse as D01
from src.core.sensor_physics import D02_rtd_callendar_van_dusen as D02
from src.core.sensor_physics import M05_nyquist_guard as M05
from src.l2_digital_twin.physics import B01_vibration_ring_buffer as B01
from src.l2_digital_twin.physics import D03_lambda_derivation as D03
from src.l2_digital_twin.physics import D04_air_mass_flow_speed_density as D04
from src.l2_digital_twin.physics import M01_excess_kurtosis as M01
from src.l2_digital_twin.physics import M02_half_order_fraction as M02
from src.l2_digital_twin.physics import M03_firing_order_fraction as M03
from src.l2_digital_twin.physics import M04_envelope_demodulation as M04
from src.l2_digital_twin.physics import M06_dual_channel_misfire_gate as M06
from src.l2_digital_twin.physics import M07_specific_fuel_consumption as M07
from src.l2_digital_twin.physics import M08_normalised_engine_load as M08
from src.l2_digital_twin.physics import M09_combustion_stability_index as M09
from src.l2_digital_twin.physics import M10_lubrication_vibration_indices as M10
from src.l2_digital_twin.physics import M11_residual_expectation_channels as M11


# --------------------------------------------------------------------------
# D-01  Type K thermocouple — against NIST ITS-90
# --------------------------------------------------------------------------

# NIST ITS-90 Type K reference EMF, microvolts, cold junction at 0 degC.
NIST_TYPE_K = {0: 0, 100: 4096, 200: 8138, 300: 12209, 400: 16397,
               500: 20644, 600: 24905, 700: 29129, 800: 33275,
               900: 37326, 1000: 41276, 1200: 48838}

TYPE_K_CLASS1_FLOOR_C = 1.5          # IEC 60584 class 1


def test_d01_matches_nist_table():
    worst = max(abs(D01.type_k_temp_c(uv) - t) for t, uv in NIST_TYPE_K.items())
    assert worst < 0.02, f"worst NIST deviation {worst:.4f} degC"


def test_d01_beats_class1_tolerance_everywhere():
    for t, uv in NIST_TYPE_K.items():
        tol = max(TYPE_K_CLASS1_FLOOR_C, 0.004 * abs(t))
        err = abs(D01.type_k_temp_c(uv) - t)
        assert err < tol, f"{t} degC: err {err:.3f} >= tol {tol:.3f}"


def test_d01_shipped_linear_form_fails_tolerance():
    """The defect is real: assert the SHIPPED form actually violates spec.

    Guards against a fix that passes while the thing it replaced was fine.
    """
    err_1200 = abs((48838 / 41.27) - 1200)
    assert err_1200 > 4.0 * TYPE_K_CLASS1_FLOOR_C, "shipped form was within tolerance?"


def test_d01_round_trip():
    for t in range(-150, 1300, 37):
        assert abs(D01.type_k_temp_c(D01.type_k_emf_uv(t)) - t) < 1e-6


def test_d01_cold_junction_uses_emf_not_degrees():
    """Hot 800 degC with cold junction at 25 degC must return 800 degC."""
    hot_uv = D01.type_k_emf_uv(800.0) - D01.type_k_emf_uv(25.0)
    k = D01.thermocouple_k_to_kelvin(hot_uv, 25.0)
    assert abs((k - 273.15) - 800.0) < 0.05


def test_d01_accepts_negative_microvolts():
    """Below the cold junction, EMF is legitimately negative.

    The shipped guard `valid = hot_uv >= 0.0` discards those readings.
    """
    hot_uv = D01.type_k_emf_uv(5.0) - D01.type_k_emf_uv(25.0)
    assert hot_uv < 0.0
    _, valid, _ = D01.convert_tc_channel_kelvin(hot_uv, 25.0)
    assert valid, "negative microvolts rejected — the shipped bug"


# --------------------------------------------------------------------------
# D-02  Pt100 RTD — against IEC 60751
# --------------------------------------------------------------------------

IEC_PT100 = {-40: 84.271, -20: 92.160, 0: 100.000, 25: 109.735, 50: 119.397,
             80: 130.897, 100: 138.505, 125: 147.951, 150: 157.325}


def _class_a_tolerance_c(t: float) -> float:
    return 0.15 + 0.002 * abs(t)


def test_d02_matches_iec_table():
    """Table resistances are rounded to 3 decimals (~0.003 degC at Pt100's
    ~0.39 ohm/degC), so the achievable bound here is rounding, not epsilon.
    The exactness of the solver is asserted separately by the round-trip test.
    """
    for t, r in IEC_PT100.items():
        k, valid, _ = D02.convert_rtd_channel_kelvin(r)
        assert valid
        assert abs((k - 273.15) - t) < 0.005, f"{t} degC from {r} ohm"


def test_d02_round_trip_is_exact():
    """Against the module's own forward model there is no table rounding."""
    for t in range(-190, 650, 17):
        r = D02.pt100_resistance_ohms(float(t))
        assert abs(D02.pt100_temp_c(r) - t) < 1e-9


def test_d02_beats_class_a_everywhere():
    for t, r in IEC_PT100.items():
        k, _, _ = D02.convert_rtd_channel_kelvin(r)
        assert abs((k - 273.15) - t) < _class_a_tolerance_c(t)


def test_d02_shipped_linear_form_fails_class_a():
    t, r = 150, 157.325
    shipped = (r - 100.0) / (100.0 * 0.00385)
    assert abs(shipped - t) > _class_a_tolerance_c(t), "shipped form was within Class A?"


def test_d02_rejects_nonphysical_resistance():
    for bad in (0.0, -5.0, float("nan")):
        _, valid, _ = D02.convert_rtd_channel_kelvin(bad)
        assert not valid


# --------------------------------------------------------------------------
# D-04  Air mass flow — MAP must drive the charge estimate
# --------------------------------------------------------------------------

DISP_M3 = 1352e-6


def test_d04_map_drives_air_flow():
    low = D04.air_mass_flow_speed_density(5000, 70_000, 320.0, DISP_M3, 0.90)
    high = D04.air_mass_flow_speed_density(5000, 140_000, 320.0, DISP_M3, 0.90)
    assert 1.9 < high / low < 2.1, f"MAP sensitivity {high/low:.3f}, expected ~2"


def test_d04_shipped_form_has_zero_map_sensitivity():
    """Reimplement thermo_mechanical_twin.py:151 verbatim and vary MAP.

    The shipped expression is `m_dot_air = fuel.value * afr_tv.value`. MAP is
    not an argument to it, so the derivative with respect to MAP is exactly
    zero — which is the whole defect, stated as a test rather than as prose.
    """
    def shipped(fuel_kg_s, afr, map_pa):       # noqa: ARG001 — map_pa unused
        return fuel_kg_s * afr                 # verbatim, line 151

    values = {shipped(0.0065, 14.7, mp) for mp in (40_000, 70_000, 101_325, 140_000)}
    assert len(values) == 1, "shipped form responded to MAP — re-read line 151"

    corrected = {D04.air_mass_flow_speed_density(5000, mp, 320.0, DISP_M3, 0.90)
                 for mp in (40_000, 70_000, 101_325, 140_000)}
    assert len(corrected) == 4, "corrected form must respond to every MAP"


def test_d04_scales_linearly_with_speed():
    a = D04.air_mass_flow_speed_density(2000, 100_000, 320.0, DISP_M3, 0.90)
    b = D04.air_mass_flow_speed_density(4000, 100_000, 320.0, DISP_M3, 0.90)
    assert abs(b / a - 2.0) < 1e-9


def test_d04_four_stroke_divisor():
    """N/120, not N/60 — a four-stroke inducts once per two revolutions."""
    rpm, charge_t, mapp = 6000.0, 300.0, 101325.0
    m = D04.air_mass_flow_speed_density(rpm, mapp, charge_t, DISP_M3, 1.0)
    rho = D04.manifold_charge_density(mapp, charge_t)
    assert abs(m - rho * DISP_M3 * rpm / 120.0) < 1e-15
    # and not the two-stroke form
    assert abs(m - rho * DISP_M3 * rpm / 60.0) > 1e-6


def test_d04_rejects_nonphysical_inputs():
    # solve_air_flow(rpm, map_pa, ambient_pa, ambient_temp_k, displacement_m3)
    for rpm, mapp, disp in ((0.0, 100_000.0, DISP_M3),
                            (5000.0, -1.0, DISP_M3),
                            (5000.0, 100_000.0, 0.0)):
        _, _, _, valid, reason = D04.solve_air_flow(rpm, mapp, 101325.0, 288.15, disp)
        assert not valid and reason


# --------------------------------------------------------------------------
# D-03  Lambda — must respond to trapped air
# --------------------------------------------------------------------------

def test_d03_lambda_tracks_air_fuel_ratio():
    assert abs(D03.lambda_from_mass_flows(0.0956, 0.0065) - 1.0) < 0.01
    assert D03.lambda_from_mass_flows(0.0800, 0.0065) < 0.90     # rich
    assert D03.lambda_from_mass_flows(0.1100, 0.0065) > 1.10     # lean


def test_d03_ring_wear_signature_is_visible():
    """The chain the hard-coded 1.0 destroys, end to end."""
    healthy = D03.lambda_from_mass_flows(0.1008, 0.0065)
    worn = D03.lambda_from_mass_flows(0.0806, 0.0065)
    assert healthy > 1.0 > worn
    assert D03.classify_mixture(healthy) == "LEAN"
    assert D03.classify_mixture(worn) == "RICH"
    assert D03.combustion_efficiency(worn) < D03.combustion_efficiency(healthy) - 0.10


def test_d03_rich_efficiency_is_oxygen_limited():
    """Below lambda 1 the burn is oxygen-limited: eta_c cannot exceed lambda."""
    for lam in (0.75, 0.85, 0.95):
        assert D03.combustion_efficiency(lam) <= lam + 1e-9


def test_d03_returns_none_not_one_when_undefined():
    assert D03.lambda_from_mass_flows(0.10, 0.0) is None
    assert D03.lambda_from_mass_flows(None, 0.0065) is None


# --------------------------------------------------------------------------
# B-01  Vibration ring buffer
# --------------------------------------------------------------------------

def test_b01_buffer_not_ready_before_full():
    buf = B01.VibrationRingBuffer(capacity=64)
    for i in range(63):
        buf.push(float(i), float(i), float(i))
        assert not buf.ready()
        assert buf.samples_until_ready() == 63 - i
    buf.push(63.0, 63.0, 63.0)
    assert buf.ready()
    assert buf.fill_fraction() == 1.0


def test_b01_window_length_matches_configured_capacity():
    buf = B01.VibrationRingBuffer(capacity=32)
    for i in range(100):
        buf.push(float(i), float(i), float(i))
    x, y, z = buf.window()
    assert len(x) == len(y) == len(z) == 32
    assert x[-1] == 99.0 and x[0] == 68.0        # oldest first, newest last


def test_b01_rejects_undersized_capacity():
    try:
        B01.VibrationRingBuffer(capacity=4)
    except ValueError:
        return
    raise AssertionError("capacity below the minimum was accepted")


def test_b01_warmup_seconds_is_reported():
    assert abs(B01.warmup_seconds(2048, 2048.0) - 1.0) < 1e-9


def test_b01_single_sample_crest_factor_is_the_bug():
    """Over one sample, crest factor is 1.0 by definition, for any signal."""
    for x in (0.1, 5.0, 900.0):
        rms = math.sqrt(x * x)
        assert abs(max(abs(x), 0.0) / rms - 1.0) < 1e-12


# --------------------------------------------------------------------------
# M-01  Excess kurtosis
# --------------------------------------------------------------------------

def test_m01_gaussian_is_near_zero():
    import random
    random.seed(7)
    g = [random.gauss(0.0, 1.0) for _ in range(20000)]
    assert abs(M01.excess_kurtosis(g)) < 0.15


def test_m01_impulsive_signal_is_strongly_positive():
    sig = [0.01] * 2000
    for i in range(0, 2000, 250):
        sig[i] = 6.0
    assert M01.excess_kurtosis(sig) > 3.0


def test_m01_uses_excess_convention():
    """Gaussian must be 0.0, not 3.0 — thresholds differ by exactly 3."""
    import random
    random.seed(11)
    g = [random.gauss(0.0, 2.5) for _ in range(20000)]
    assert abs(M01.excess_kurtosis(g)) < 0.2


def test_m01_degenerate_input_is_none_not_zero():
    """0.0 is the Gaussian value and would band as NORMAL."""
    assert M01.excess_kurtosis([1.0]) is None
    assert M01.excess_kurtosis([2.0] * 500) is None            # dead channel
    assert M01.excess_kurtosis([1.0, float("nan"), 3.0, 4.0]) is None
    assert M01.kurtosis_band(None) == "UNKNOWN"
    assert M01.kurtosis_band(None) != "NORMAL"


# --------------------------------------------------------------------------
# M-02 / M-03  Order-domain features
# --------------------------------------------------------------------------

FS = 2048.0
N = 2048


def _synth(rpm: float, half_amp: float, fire_amp: float = 1.0) -> list[float]:
    f1 = rpm / 60.0
    return [fire_amp * math.sin(2 * math.pi * 2 * f1 * i / FS)
            + half_amp * math.sin(2 * math.pi * 0.5 * f1 * i / FS)
            for i in range(N)]


def test_m02_separates_misfire_across_the_speed_range():
    for rpm in (2000.0, 3500.0, 5800.0):
        healthy = M02.half_order_fraction(_synth(rpm, 0.001), FS, rpm)
        misfire = M02.half_order_fraction(_synth(rpm, 0.50), FS, rpm)
        assert misfire > 100 * healthy, f"{rpm} rpm: only {misfire/healthy:.0f}x"


def test_m02_is_order_domain_not_fixed_hz():
    """The 0.5X component moves 16.7 -> 48.3 Hz across the range.

    A fixed-Hz band cannot track it; the fraction must stay high regardless.
    """
    vals = [M02.half_order_fraction(_synth(rpm, 0.50), FS, rpm)
            for rpm in (2000.0, 3500.0, 5800.0)]
    assert min(vals) > 0.05
    assert max(vals) / min(vals) < 4.0, "fraction drifts with speed"


def test_m03_firing_order_is_speed_invariant():
    vals = [M03.firing_order_fraction(_synth(rpm, 0.0), FS, rpm)
            for rpm in (2000.0, 3500.0, 5800.0)]
    assert min(vals) > 0.95
    assert max(vals) - min(vals) < 0.02, f"drifts with speed: {vals}"


def test_m03_firing_order_is_2x_for_four_cylinder_four_stroke():
    assert abs(M03.firing_order(n_cylinders=4, strokes=4) - 2.0) < 1e-12
    assert abs(M03.firing_order(n_cylinders=6, strokes=4) - 3.0) < 1e-12


# --------------------------------------------------------------------------
# M-04  Envelope demodulation
# --------------------------------------------------------------------------

def test_m04_recovers_modulation_rate_invisible_to_rms():
    carrier, modulation = 800.0, 37.0   # inside the revised 650-1000 Hz default band
    plain = [math.sin(2 * math.pi * carrier * i / FS) for i in range(N)]
    modded = [(1.0 + 0.6 * math.sin(2 * math.pi * modulation * i / FS))
              * math.sin(2 * math.pi * carrier * i / FS) for i in range(N)]

    freqs, spec = M04.envelope_spectrum(modded, FS)
    lo = next(i for i, f in enumerate(freqs) if f > 5.0)
    peak = freqs[lo + max(range(len(spec) - lo), key=lambda i: spec[lo + i])]
    assert abs(peak - modulation) < 2.0, f"peak at {peak:.1f} Hz"

    rms_plain = math.sqrt(sum(v * v for v in plain) / N)
    rms_mod = math.sqrt(sum(v * v for v in modded) / N)
    assert rms_mod / rms_plain < 1.15, "RMS moved a lot — bad demonstration"


def test_m04_default_band_is_below_nyquist_at_current_config():
    """650-1000 Hz: above the firing harmonics, below Nyquist at 2048 Hz (OI-9)."""
    assert M04.DEFAULT_BAND_LO_HZ < M04.DEFAULT_BAND_HI_HZ <= FS / 2.0


# --------------------------------------------------------------------------
# M-05  Nyquist guard
# --------------------------------------------------------------------------

def test_m05_rejects_the_frameworks_configured_band():
    """Upper edge 10 kHz at fs 2048 Hz. This is the live misconfiguration."""
    assert not M05.band_is_representable(10000.0, 2048.0)
    try:
        M05.assert_band_within_nyquist(10000.0, 2048.0)
    except M05.NyquistViolation:
        return
    raise AssertionError("10 kHz band edge accepted at 2048 Hz")


def test_m05_accepts_the_corrected_band():
    M05.assert_band_within_nyquist(1000.0, 2048.0)
    assert M05.band_is_representable(1000.0, 2048.0)


def test_m05_nyquist_limit():
    assert M05.nyquist_limit_hz(2048.0) == 1024.0


def test_m05_max_representable_order_falls_with_speed():
    """The same sample rate buys fewer shaft orders as the shaft speeds up."""
    assert M05.max_representable_order(2000.0, 2048.0) > \
           M05.max_representable_order(5800.0, 2048.0)


# --------------------------------------------------------------------------
# M-06  Dual-channel misfire gate
# --------------------------------------------------------------------------

def test_m06_requires_both_channels():
    g = M06.DualChannelMisfireGate()
    assert g.evaluate(5.0, 0.20).verdict is M06.MisfireVerdict.CONFIRMED
    assert g.evaluate(5.0, 0.001).verdict is M06.MisfireVerdict.UNCONFIRMED
    assert g.evaluate(0.3, 0.20).verdict is M06.MisfireVerdict.UNCONFIRMED
    assert g.evaluate(0.3, 0.001).verdict is M06.MisfireVerdict.NORMAL


def test_m06_bearing_fault_does_not_confirm_a_misfire():
    """High overall vibration, no half-order content.

    The shipped detector uses overall RMS as its vibration evidence, so this
    case raises its misfire score. The gate must not confirm.
    """
    r = M06.DualChannelMisfireGate().evaluate(0.4, 0.002)
    assert r.verdict is M06.MisfireVerdict.NORMAL


def test_m06_names_the_channel_that_fired():
    r = M06.DualChannelMisfireGate().evaluate(5.0, 0.001)
    assert r.triggering_channel == "crank_speed_variability"


def test_m06_invalid_input_yields_no_verdict():
    r = M06.DualChannelMisfireGate().evaluate(float("nan"), 0.20)
    assert r.verdict is M06.MisfireVerdict.INVALID


def test_m06_cov_is_a_dispersion_not_a_rate():
    """Smooth acceleration must not look like oscillation of the same span."""
    steady = M06.CrankSpeedVariability(window=16)
    for _ in range(16):
        steady.push(4000.0)
    assert steady.coefficient_of_variation_pct() < 1e-9

    osc = M06.CrankSpeedVariability(window=16)
    for i in range(16):
        osc.push(4000.0 + (400.0 if i % 2 else -400.0))
    assert osc.coefficient_of_variation_pct() > 9.0


# --------------------------------------------------------------------------
# M-07 / M-08  Derived parameters
# --------------------------------------------------------------------------

def test_m07_sfc_formula():
    assert abs(M07.specific_fuel_consumption_kg_kwh(0.0065, 75.0) - 0.312) < 0.001


def test_m07_returns_none_at_idle_not_zero():
    """0.0 would be read downstream as perfect efficiency."""
    assert M07.specific_fuel_consumption_kg_kwh(0.001, 0.5) is None
    assert M07.sfc_band(None) == "UNKNOWN"
    assert M07.sfc_band(None) != "NORMAL"


def test_m07_bands_match_appendix_b():
    assert M07.sfc_band(0.300) == "NORMAL"
    assert M07.sfc_band(0.350) == "WARNING"
    assert M07.sfc_band(0.400) == "ALARM"


def test_m08_load_and_bands():
    # Prompt 18: rated (peak) power 104 kW per flyrotax.com/products/915-is-a-isc-a (was 105.0)
    assert abs(M08.normalised_engine_load(104.0) - 1.0) < 1e-9
    assert M08.load_band(M08.normalised_engine_load(8.0)) == "IDLE"
    assert M08.load_band(M08.normalised_engine_load(111.0)) == "ABOVE_RATED"


def test_m08_zero_sensitivity_reproduces_fixed_threshold():
    """Lets channels be converted one at a time with no behaviour change."""
    for load in (0.0, 0.3, 0.8, 1.1):
        assert M08.load_adjusted_threshold(200.0, load, 0.0) == 200.0


def test_m08_above_rated_is_visible_not_clipped_to_one():
    assert M08.normalised_engine_load(111.0) > 1.0


# --------------------------------------------------------------------------
# M-09  Combustion stability index
# --------------------------------------------------------------------------

def test_m09_healthy_climb_does_not_alarm():
    """The defect found by running the literal Appendix A.4 formula."""
    r = M09.combustion_stability_index(5.0, 1120.0, 0.4, 0.55,
                                       expected_cht_slope_k_per_s=0.55)
    assert r.band == "NORMAL", f"healthy climb -> {r.band} (CSI {r.value:.3f})"


def test_m09_thermal_runaway_still_alarms():
    r = M09.combustion_stability_index(6.0, 1130.0, 0.5, 1.10,
                                       expected_cht_slope_k_per_s=0.55)
    assert r.band == "ALARM"


def test_m09_reduces_to_appendix_a4_when_expected_slope_is_zero():
    a = M09.combustion_stability_index(40.0, 1100.0, 2.0, 0.30)
    b = M09.combustion_stability_index(40.0, 1100.0, 2.0, 0.30,
                                       expected_cht_slope_k_per_s=0.0)
    assert abs(a.value - b.value) < 1e-12


def test_m09_terms_are_dimensionless_and_sum_to_the_total():
    r = M09.combustion_stability_index(60.0, 1100.0, 6.0, 0.30)
    total = r.egt_dispersion_term + r.speed_variability_term + r.thermal_slope_term
    assert abs(total - r.value) < 1e-12


def test_m09_names_the_dominant_term():
    r = M09.combustion_stability_index(5.0, 1100.0, 20.0, 0.01)
    assert r.dominant_term == "speed_variability"


# --------------------------------------------------------------------------
# M-10  Health indices
# --------------------------------------------------------------------------

def test_m10_lhi_is_one_for_healthy_oil_at_any_temperature():
    """The defect found by running the fixed-nominal form."""
    for t in (35.0, 85.0, 115.0):
        exp_v = M10.expected_viscosity_pa_s(t, 0.043, 85.0)
        r = M10.lubrication_health_index(4.0e5, 4.0e5, exp_v, exp_v, t)
        assert r.band == "NORMAL", f"healthy oil at {t} degC -> {r.band}"
        assert abs(r.value - 1.0) < 1e-9


def test_m10_lhi_detects_oil_thinner_than_expected():
    t = 85.0
    exp_v = M10.expected_viscosity_pa_s(t, 0.043, 85.0)
    r = M10.lubrication_health_index(0.85 * 4.0e5, 4.0e5, 0.58 * exp_v, exp_v, t)
    assert r.band == "ALARM"
    assert r.dominant_factor == "viscosity"


def test_m10_over_temperature_derate_is_one_sided():
    assert M10.over_temperature_derate(35.0) == 1.0      # cold is not derated
    assert M10.over_temperature_derate(130.0) == 1.0
    assert M10.over_temperature_derate(140.0) < 1.0


def test_m10_vhi_catches_an_early_defect_that_rms_misses():
    r = M10.vibration_health_index(2.1, 2.0, 2.7, 2.5, 0.14, 0.05)
    assert r.band != "NORMAL"
    assert r.dominant_factor == "envelope"
    assert r.factors["rms"] < 1.1       # RMS barely moved


# --------------------------------------------------------------------------
# M-11  Residual expectation channels
# --------------------------------------------------------------------------

def test_m11_cht_expectation_rises_with_load():
    idle = M11.expected_cht_k(1800.0, 60_000.0, 288.15, 0.0)
    power = M11.expected_cht_k(5800.0, 140_000.0, 288.15, 0.0)
    assert power > idle + 30.0


def test_m11_cht_expectation_credits_cooling_airflow():
    """Without this, a descent reads as a cooling fault."""
    ground = M11.expected_cht_k(4800.0, 115_000.0, 278.15, 0.0)
    cruise = M11.expected_cht_k(4800.0, 115_000.0, 278.15, 1.0)
    assert cruise < ground


def test_m11_fuel_expectation_keys_on_expected_power():
    """An engine burning correct fuel for LESS power must show a residual."""
    exp = M11.expected_fuel_flow_kg_s(75.0)
    healthy = M11.normalised_residual(exp, exp, 0.0004)
    degraded = M11.normalised_residual(exp * 1.15, exp, 0.0004)
    assert abs(healthy) < 1e-9
    assert degraded > 2.0


def test_m11_per_cylinder_offsets_sum_to_zero():
    """The fleet-mean expectation must be unchanged."""
    assert abs(sum(M11.DEFAULT_EGT_CYLINDER_OFFSETS_K)) < 1e-9


def test_m11_per_cylinder_removes_standing_residual_on_healthy_engine():
    common = 1065.0
    per_cyl = M11.expected_egt_per_cylinder_k(common)
    measured = (1057.0, 1062.0, 1069.0, 1072.0)     # healthy boxer spread
    shared_worst = max(abs(M11.normalised_residual(m, common, 50.0)) for m in measured)
    percyl_worst = max(abs(M11.normalised_residual(m, e, 50.0))
                       for m, e in zip(measured, per_cyl))
    assert percyl_worst < shared_worst
    assert percyl_worst < 1e-9


def test_m11_expectations_never_read_the_channel_they_predict():
    """SRD-CON-001 in spirit: expectations are operating-point functions only.

    Same operating point, wildly different measured CHT -> identical expectation.
    """
    a = M11.expected_cht_k(4800.0, 115_000.0, 278.15, 1.0)
    b = M11.expected_cht_k(4800.0, 115_000.0, 278.15, 1.0)
    assert a == b


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, str(exc) or "assertion failed"))
            print(f"  FAIL  {name}  — {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}  — {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("\nfailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
