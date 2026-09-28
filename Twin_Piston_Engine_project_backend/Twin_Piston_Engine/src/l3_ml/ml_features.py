"""
ML feature schema 2.0.0 (Prompt 16).

Features are built ONLY from the L2/L3 outputs the live pipeline produces
(residuals, twin-derived quantities, vibration, combustion, lubrication,
electrical, injection, coolant and trend states). No raw channel, no
simulator ground truth, no fault configuration, no timestamp, sequence or
onset time. A feature that is not derivable in a record is NaN (never a
substituted value); the trained models handle missing values natively
(LightGBM) or through an imputer fitted on training data (IsolationForest),
and the classifier scales its confidence by the importance-weighted share of
features present.

Every feature belongs to a sensor group, used for the channel-dropout
evaluation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from src.core.provenance import Provenance
from src.l3_ml.ml_infrastructure import MLFeature, MLFeatureVector

ML_FEATURE_SCHEMA_VERSION = "2.0.0"

_TREND_CHANNELS = ("cht", "egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4", "oil_temp", "coolant")

# feature name -> sensor group
FEATURE_GROUPS: dict[str, str] = {
    **{f"res_egt_cyl{i}": "egt" for i in range(1, 5)},
    "res_cht": "cht",
    "res_coolant": "coolant",
    "res_fuel_flow": "fuel",
    "res_oil_pressure": "oil",
    "res_oil_temp": "oil",
    "res_vibration_rms": "vibration",
    "res_brake_power_kw": "fuel",
    "res_fuel_rail_pressure": "fuel",
    "res_fuel_delivery_ratio": "fuel",
    **{f"res_injector_flow_ratio_cyl{i}": "fuel" for i in range(1, 5)},
    "twin_bmep_pa": "fuel",
    "twin_eta_volumetric": "fuel",
    "twin_eta_thermal": "fuel",
    "twin_mean_piston_speed": "crank",
    "twin_lambda": "fuel",
    "twin_sfc": "fuel",
    "twin_engine_load": "fuel",
    "vib_overall_rms": "vibration",
    "vib_crest": "vibration",
    "vib_kurtosis": "vibration",
    "vib_half_order": "vibration",
    "vib_firing_order": "vibration",
    "vib_envelope_rms": "vibration",
    "vib_vhi": "vibration",
    "vib_prop1x_vel_frac_lat": "vibration",
    "vib_prop1x_lat_vert_ratio": "vibration",
    "comb_crank_cov": "crank",
    "comb_csi": "crank",
    "lub_lhi": "oil",
    "lub_pressure_margin": "oil",
    "lub_temp_margin": "oil",
    "elec_charging_residual": "electrical",
    "elec_ripple_pct": "electrical",
    "elec_r_int": "electrical",
    "elec_ehi": "electrical",
    "elec_ripple_order": "electrical",
    "inj_rail_residual": "fuel",
    "inj_mixture_factor": "fuel",
    "inj_delivery_ratio": "fuel",
    **{f"inj_egt_diff_cyl{i}": "egt" for i in range(1, 5)},
    **{f"inj_ign_residual_cyl{i}": "ignition" for i in range(1, 5)},
    "inj_duty_max": "fuel",
    "cool_residual": "coolant",
    "cool_cht_delta": "coolant",
    "cool_cht_delta_residual": "coolant",
    **{f"trend_{c}": ("egt" if c.startswith("egt") else "oil" if c == "oil_temp" else c) for c in _TREND_CHANNELS},
    "egt_spread": "egt",
}
ML_FEATURES: tuple[str, ...] = tuple(FEATURE_GROUPS)
SENSOR_GROUPS: tuple[str, ...] = tuple(sorted(set(FEATURE_GROUPS.values())))


def _v(tv: Any) -> float:
    if tv is None or not getattr(tv, "valid", False) or tv.value is None:
        return math.nan
    x = float(tv.value)
    return x if math.isfinite(x) else math.nan


def extract_features(
    residual_state: Any = None,
    derived_state: Any = None,
    vib_state: Any = None,
    comb_state: Any = None,
    lub_state: Any = None,
    elec_state: Any = None,
    injection_state: Any = None,
    coolant_state: Any = None,
    overheat_state: Any = None,
    egt_result: Any = None,
) -> dict[str, float]:
    """One feature row (NaN where not derivable) from pipeline states."""
    f: dict[str, float] = {name: math.nan for name in ML_FEATURES}
    if residual_state is not None:
        r = residual_state.residuals
        for key in ("egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4", "cht", "coolant", "fuel_flow", "oil_pressure",
                    "oil_temp", "vibration_rms", "brake_power_kw", "fuel_rail_pressure", "fuel_delivery_ratio",
                    "injector_flow_ratio_cyl1", "injector_flow_ratio_cyl2", "injector_flow_ratio_cyl3",
                    "injector_flow_ratio_cyl4"):
            f[f"res_{key}"] = _v(r.get(key))
    if derived_state is not None:
        d = derived_state
        f.update(twin_bmep_pa=_v(d.bmep_pa), twin_eta_volumetric=_v(d.eta_volumetric), twin_eta_thermal=_v(d.eta_thermal),
                 twin_mean_piston_speed=_v(d.mean_piston_speed_m_s), twin_lambda=_v(d.lambda_derived),
                 twin_sfc=_v(d.sfc_kg_kwh), twin_engine_load=_v(d.engine_load))
    if vib_state is not None:
        v = vib_state
        f.update(vib_overall_rms=_v(v.overall_rms_m_s2), vib_crest=_v(v.crest_factor), vib_kurtosis=_v(v.excess_kurtosis),
                 vib_half_order=_v(v.half_order_fraction), vib_firing_order=_v(v.firing_order_fraction),
                 vib_envelope_rms=_v(v.envelope_rms_m_s2), vib_vhi=_v(v.vibration_health_index),
                 vib_prop1x_vel_frac_lat=_v(v.prop_1x_velocity_fraction_lateral),
                 vib_prop1x_lat_vert_ratio=_v(v.prop_1x_lateral_vertical_ratio))
    if comb_state is not None:
        f.update(comb_crank_cov=_v(comb_state.crank_cov_pct), comb_csi=_v(comb_state.csi_value))
    if lub_state is not None:
        f.update(lub_lhi=_v(lub_state.lhi), lub_pressure_margin=_v(lub_state.pressure_margin_pa),
                 lub_temp_margin=_v(lub_state.temperature_margin_k))
    if elec_state is not None:
        e = elec_state
        f.update(elec_charging_residual=_v(e.charging_residual_median_v), elec_ripple_pct=_v(e.voltage_ripple_pct),
                 elec_r_int=_v(e.battery_resistance_mohm), elec_ehi=_v(e.ehi), elec_ripple_order=_v(e.ripple_order))
    if injection_state is not None:
        s = injection_state
        f.update(inj_rail_residual=_v(s.rail_pressure_residual_median_kpa), inj_mixture_factor=_v(s.rail_mixture_factor),
                 inj_delivery_ratio=_v(s.fuel_delivery_ratio_median))
        for i in range(4):
            f[f"inj_egt_diff_cyl{i + 1}"] = _v(s.egt_differential_k[i])
            f[f"inj_ign_residual_cyl{i + 1}"] = _v(s.ign_timing_residual_median_deg[i])
        duties = [_v(t) for t in s.injector_duty_pct]
        finite = [x for x in duties if not math.isnan(x)]
        f["inj_duty_max"] = max(finite) if finite else math.nan
    if coolant_state is not None:
        c = coolant_state
        f.update(cool_residual=_v(c.coolant_residual_median_k), cool_cht_delta=_v(c.coolant_cht_delta_k),
                 cool_cht_delta_residual=_v(c.cht_delta_residual_k))
    if overheat_state is not None:
        for ch in _TREND_CHANNELS:
            t = overheat_state.channels.get(ch)
            f[f"trend_{ch}"] = _v(t.slope_residual_k_min) if t is not None else math.nan
    if egt_result is not None and getattr(egt_result, "spread_egt_k", None) is not None:
        x = float(egt_result.spread_egt_k)
        f["egt_spread"] = x if math.isfinite(x) else math.nan
    return f


def feature_vector_from_row(row: dict[str, float], timestamp: datetime | None = None) -> MLFeatureVector:
    """MLFeatureVector (schema 2.0.0) with NaN for missing features. valid is
    True when at least one feature is present; per-feature validity and the
    coverage are carried along for the models."""
    feats: dict[str, MLFeature] = {}
    values: list[float] = []
    for name in ML_FEATURES:
        x = row.get(name, math.nan)
        ok = x is not None and not math.isnan(x)
        feats[name] = MLFeature(name=name, value=x if ok else math.nan, provenance=Provenance.DERIVED,
                                quality=1.0 if ok else 0.0, valid=ok)
        values.append(x if ok else math.nan)
    coverage = sum(1 for f in feats.values() if f.valid) / len(ML_FEATURES)
    return MLFeatureVector(
        timestamp=timestamp or datetime.now(timezone.utc),
        provenance=Provenance.DERIVED,
        feature_names=list(ML_FEATURES),
        values=values,
        features=feats,
        schema_version=ML_FEATURE_SCHEMA_VERSION,
        valid=coverage > 0.0,
        quality=coverage,
    )
