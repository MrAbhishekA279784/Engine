"""
Healthy Expectation Models and Residual Engine — Original Module 11.

Seventh processing stage of L2 Digital Twin.
Generates deterministic physical healthy expectations based on operating point (RPM, MAP, ambient context)
and computes multi-signal physical residuals:
    Residual = Observed - Expected Healthy
    Normalized Residual = (Observed - Expected Healthy) / Scale

STRICT BOUNDARY CONSTRAINTS:
    - Input: Modules 5–10 outputs (NormalizedSignalRecord, DerivedEngineState, DiagnosticState, etc.)
    - Output: ResidualState (canonical domain schema) & HealthyExpectationResult
    - Zero ML models, zero ML inference, zero ML training
    - Zero simulator internals or ground truth dependencies
    - Zero diagnostic fault labels ("bearing failure", "misfire", etc.)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, HealthyBaselineConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import (
    DerivedEngineState,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    make_tagged,
)
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.physics import M11_residual_expectation_channels as M11

logger = get_logger(__name__)

# Standard ISA Sea Level Constants
STD_PRESSURE_PA = 101325.0
STD_TEMP_K = 288.15


class QuantityResidual(BaseModel):
    """Detailed residual container for a single physical quantity."""

    name: str
    observed: float | None
    expected: float | None
    raw_residual: float | None
    absolute_residual: float | None
    normalized_residual: float | None
    valid: bool
    quality: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class HealthyExpectationResult(BaseModel):
    """Container for expected healthy state and computed residuals."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    operating_point: OperatingPoint
    expected_values: dict[str, float | None]
    residuals: dict[str, QuantityResidual]
    # fleet expectation before the per-engine baseline correction (Prompt 17)
    raw_expected_values: dict[str, float | None] = Field(default_factory=dict)
    baseline_version: str = "fleet-0"

    model_config = ConfigDict(frozen=True)


class HealthyExpectationModel:
    """Deterministic healthy baseline model based on operating context.

    Every expected_* method is a function of the operating point (and, where
    noted, of OTHER channels) only. None may take the channel it predicts as
    input: an expectation derived from the measurement it judges collapses the
    residual toward zero exactly when the fault grows (M-11; guarded by
    tests/unit/test_residual_expectations.py).
    """

    # The healthy expectations are NOT calibrated (to the simulator or to
    # flight data; OI-5). Consumers must not treat them as ground truth.
    CALIBRATED = False

    # There is no airspeed channel in RawSignalRecord/OperatingPoint, so the
    # CHT cooling-air credit is not applied (conservative: no ram air).
    CHT_AIRSPEED_FRACTION = 0.0

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: HealthyBaselineConfig = self._settings.healthy_baseline
        self._cht_cfg = M11.CHTExpectationConfig(
            base_cht_k=self._cfg.base_cht_k,
            k_cht_load_k=self._cfg.k_cht_load_k,
            k_cht_ambient=self._cfg.k_cht_ambient,
            k_cht_airspeed=self._cfg.k_cht_airspeed,
            scale_cht_k=self._cfg.scale_cht_k,
        )
        self._fuel_cfg = M11.FuelExpectationConfig(
            baseline_sfc_kg_kwh=self._cfg.baseline_sfc_kg_kwh,
            idle_fuel_kg_s=self._cfg.idle_fuel_kg_s,
            scale_fuel_kg_s=self._cfg.scale_fuel_kg_s,
        )

    def expected_egt_k(self, op: OperatingPoint) -> float:
        """Expected healthy EGT based on MAP, RPM, and ambient temperature."""
        delta_p = op.map_pressure_pa - STD_PRESSURE_PA
        delta_rpm = max(0.0, op.rpm - 1000.0)
        temp_effect = (op.ambient_temp_k - STD_TEMP_K) * 0.5
        return self._cfg.base_egt_k + (self._cfg.k_egt_map_pa * delta_p) + (self._cfg.k_egt_rpm * delta_rpm) + temp_effect

    def expected_egt_per_cylinder_k(self, op: OperatingPoint) -> tuple[float, float, float, float] | None:
        """Per-cylinder expected EGT: common expectation + installation offset (M-11)."""
        offsets = tuple(self._cfg.egt_cylinder_offsets_k)
        return M11.expected_egt_per_cylinder_k(self.expected_egt_k(op), offsets)  # type: ignore[arg-type]

    def expected_cht_k(self, op: OperatingPoint) -> float | None:
        """Expected healthy CHT from rpm, MAP and ambient temperature (M-11)."""
        return M11.expected_cht_k(
            op.rpm, op.map_pressure_pa, op.ambient_temp_k,
            airspeed_fraction=self.CHT_AIRSPEED_FRACTION, cfg=self._cht_cfg,
        )

    def expected_oil_pressure_pa(self, op: OperatingPoint, oil_temp_k: float = 363.15) -> float:
        """Expected healthy oil pressure based on RPM and oil temperature."""
        nom_p = 400000.0  # 4.0 bar
        rpm_factor = math.sqrt(max(0.1, op.rpm) / 4000.0)
        c = 140.0
        visc_ratio = 1.0
        if oil_temp_k > c:
            num = math.exp(1200.0 / (oil_temp_k - c))
            den = math.exp(1200.0 / (363.15 - c))
            visc_ratio = math.pow(num / den, 0.3) if den > 0 else 1.0

        return nom_p * rpm_factor * visc_ratio

    def expected_oil_temp_k(self, op: OperatingPoint) -> float:
        """Expected healthy oil temperature based on load and ambient temperature."""
        load_factor = (op.map_pressure_pa / STD_PRESSURE_PA) * (op.rpm / 5800.0)
        return op.ambient_temp_k + (self._cfg.base_oil_temp_rise_k * load_factor)

    def expected_vibration_rms_m_s2(self, op: OperatingPoint) -> float:
        """Expected healthy vibration RMS based on engine speed."""
        speed_ratio = op.rpm / 5800.0
        return self._cfg.base_vibration_rms_m_s2 + (self._cfg.k_vib_rpm * (speed_ratio ** 2))

    def expected_brake_power_kw(self, op: OperatingPoint) -> float:
        """Expected healthy brake power based on MAP and RPM."""
        rated_p_kw = self._settings.engine.rated_power_kw
        return rated_p_kw * (op.map_pressure_pa / self._cfg.expected_power_map_ref_pa) * (op.rpm / 5800.0)

    def expected_fuel_flow_kg_s(self, op: OperatingPoint) -> float | None:
        """Expected healthy fuel flow, keyed on EXPECTED brake power (M-11).

        Must never use measured power: an engine producing less power than it
        should would then also get a lower fuel expectation, cancelling the
        fuel residual exactly when the engine degrades.
        """
        return M11.expected_fuel_flow_kg_s(self.expected_brake_power_kw(op), self._fuel_cfg)

    def expected_coolant_temp_k(self, op: OperatingPoint) -> float:
        """Expected healthy coolant temperature (Prompt 14): thermostat-regulated
        level rising with load and ambient. Operating point only."""
        c = self._settings.coolant
        load = (op.map_pressure_pa / STD_PRESSURE_PA) * (op.rpm / 5800.0)
        return c.regulated_c + 273.15 + c.k_load_k * load + c.k_ambient * (op.ambient_temp_k - STD_TEMP_K)

    def expected_fuel_rail_dp_pa(self, op: OperatingPoint, injector_demand_kg_s: float = 0.0) -> float:
        """Expected rail - manifold pressure at the current injector demand:
        regulator setpoint minus droop x demand (Prompt 13). The demand is the
        injector command, never the measured rail pressure."""
        inj = self._settings.injection
        return inj.rail_dp_setpoint_pa - inj.rail_droop_pa_per_kg_s * max(injector_demand_kg_s, 0.0)

    def expected_fuel_delivery_ratio(self, op: OperatingPoint) -> float:
        """Measured / commanded fuel on a healthy system (SRD-FUN-084)."""
        return 1.0

    def expected_injector_flow_ratio(self, op: OperatingPoint) -> float:
        """Fuel burnt / fuel commanded per cylinder on a healthy injector."""
        return 1.0


def operating_point_from_record(record: NormalizedSignalRecord) -> OperatingPoint:
    """Operating point used as the key for every healthy expectation."""
    rpm_val = record.rpm.value if (record.rpm.valid and record.rpm.value >= 0) else 0.0
    map_val = record.map_pressure.value if (record.map_pressure.valid and record.map_pressure.value > 0) else STD_PRESSURE_PA
    amb_p_val = record.ambient_pressure.value if (record.ambient_pressure.valid and record.ambient_pressure.value > 0) else STD_PRESSURE_PA
    amb_t_val = record.ambient_temp.value if (record.ambient_temp.valid and record.ambient_temp.value > 0) else STD_TEMP_K
    # No substitute: None means the throttle is not instrumented.
    throttle_val = record.throttle_position.value if record.throttle_position.valid else None
    altitude_m = max(0.0, 44330.0 * (1.0 - math.pow(amb_p_val / STD_PRESSURE_PA, 0.1903)))
    return OperatingPoint(
        rpm=rpm_val,
        map_pressure_pa=map_val,
        altitude_m=altitude_m,
        ambient_temp_k=amb_t_val,
        ambient_pressure_pa=amb_p_val,
        throttle_pct=throttle_val,
    )


class ResidualEngine:
    """Reusable residual calculation engine for physical actuals vs expected healthy baselines."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: HealthyBaselineConfig = self._settings.healthy_baseline
        self.expectation_model = HealthyExpectationModel(self._settings)
        # Per-engine baseline (Prompt 17): residual channel -> (a, b, ref) so the
        # expectation becomes expected + a + b (expected - ref). Empty = the
        # frozen fleet expectation (baseline_version "fleet-0").
        self.baseline_corrections: dict[str, tuple[float, float, float]] = {}
        self.baseline_version: str = "fleet-0"

    def set_baseline(self, corrections: dict[str, tuple[float, float, float]], version: str) -> None:
        self.baseline_corrections = dict(corrections)
        self.baseline_version = version

    def _adapt(self, channel: str, expected: float | None) -> float | None:
        c = self.baseline_corrections.get(channel)
        if c is None or expected is None:
            return expected
        a, b, ref = c
        return expected + a + b * (expected - ref)

    def compute_residual(
        self,
        name: str,
        observed: float | None,
        observed_valid: bool,
        expected: float | None,
        scale: float,
        quality: float = 1.0,
    ) -> QuantityResidual:
        """Compute raw, absolute, and normalized residual for a single quantity."""
        valid = observed_valid and (observed is not None) and (expected is not None) and not (math.isnan(observed) or math.isnan(expected))
        if not valid or scale <= 0.0:
            return QuantityResidual(
                name=name,
                observed=observed,
                expected=expected,
                raw_residual=None,
                absolute_residual=None,
                normalized_residual=None,
                valid=False,
                quality=0.0,
            )

        raw_res = observed - expected
        abs_res = abs(raw_res)
        norm_res = raw_res / scale

        return QuantityResidual(
            name=name,
            observed=observed,
            expected=expected,
            raw_residual=raw_res,
            absolute_residual=abs_res,
            normalized_residual=norm_res,
            valid=True,
            quality=quality,
        )

    def evaluate(
        self,
        record: NormalizedSignalRecord,
        derived_state: DerivedEngineState | None = None,
        injection_state: Any = None,
    ) -> tuple[ResidualState, HealthyExpectationResult]:
        """Evaluate healthy expectations and generate residuals from telemetry.

        With an InjectionState (Prompt 13) the residuals also carry
        fuel_rail_pressure [Pa], fuel_delivery_ratio and
        injector_flow_ratio_cyl1..4 (ratio - expected ratio; running medians).
        """
        op = operating_point_from_record(record)
        # Expectations are keyed on the operating point. If rpm, MAP or an
        # ambient input is invalid, operating_point_from_record substituted a
        # default; residuals built on that would look plausible but be wrong,
        # so they are marked invalid instead.
        op_invalid = [name for name, ch in (("rpm", record.rpm), ("map_pressure", record.map_pressure),
                                            ("ambient_temp", record.ambient_temp),
                                            ("ambient_pressure", record.ambient_pressure)) if not ch.valid]

        exp_egt = self.expectation_model.expected_egt_k(op)
        exp_egt_cyl = self.expectation_model.expected_egt_per_cylinder_k(op) or (None, None, None, None)
        exp_cht = self.expectation_model.expected_cht_k(op)
        exp_coolant = self.expectation_model.expected_coolant_temp_k(op)
        exp_fuel = self.expectation_model.expected_fuel_flow_kg_s(op)
        exp_oil_p = self.expectation_model.expected_oil_pressure_pa(op, record.oil_temp.value if record.oil_temp.valid else 363.15)
        exp_oil_t = self.expectation_model.expected_oil_temp_k(op)
        exp_vib = self.expectation_model.expected_vibration_rms_m_s2(op)
        exp_power = self.expectation_model.expected_brake_power_kw(op)
        raw_expected = {f"egt_cyl{i + 1}": v for i, v in enumerate(exp_egt_cyl)}
        raw_expected.update(cht=exp_cht, coolant=exp_coolant, fuel_flow=exp_fuel, oil_pressure=exp_oil_p,
                            oil_temp=exp_oil_t, vibration_rms=exp_vib, brake_power_kw=exp_power)
        if self.baseline_corrections:
            exp_egt_cyl = tuple(self._adapt(f"egt_cyl{i + 1}", v) for i, v in enumerate(exp_egt_cyl))
            exp_cht, exp_coolant = self._adapt("cht", exp_cht), self._adapt("coolant", exp_coolant)
            exp_fuel, exp_oil_p = self._adapt("fuel_flow", exp_fuel), self._adapt("oil_pressure", exp_oil_p)
            exp_oil_t, exp_vib = self._adapt("oil_temp", exp_oil_t), self._adapt("vibration_rms", exp_vib)
            exp_power = self._adapt("brake_power_kw", exp_power)

        expected_dict = {
            "egt_common": exp_egt,
            "egt_cyl1": exp_egt_cyl[0],
            "egt_cyl2": exp_egt_cyl[1],
            "egt_cyl3": exp_egt_cyl[2],
            "egt_cyl4": exp_egt_cyl[3],
            "cht": exp_cht,
            "coolant": exp_coolant,
            "fuel_flow": exp_fuel,
            "oil_pressure": exp_oil_p,
            "oil_temp": exp_oil_t,
            "vibration_rms": exp_vib,
            "brake_power_kw": exp_power,
        }

        res_dict: dict[str, QuantityResidual] = {}
        schema_residuals: dict[str, ProvenanceTaggedValue[float | None]] = {}

        egt_channels = [
            ("egt_cyl1", record.egt_cyl_1),
            ("egt_cyl2", record.egt_cyl_2),
            ("egt_cyl3", record.egt_cyl_3),
            ("egt_cyl4", record.egt_cyl_4),
        ]

        for (name, ch), exp_cyl in zip(egt_channels, exp_egt_cyl):
            q_res = self.compute_residual(
                name=name,
                observed=ch.value if ch.valid else None,
                observed_valid=ch.valid,
                expected=exp_cyl,
                scale=self._cfg.scale_egt_k,
                quality=ch.quality,
            )
            res_dict[name] = q_res
            if q_res.valid and q_res.raw_residual is not None:
                schema_residuals[name] = make_tagged(q_res.raw_residual, Provenance.DERIVED, valid=True, quality=q_res.quality)
            else:
                schema_residuals[name] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # CHT: RawSignalRecord carries one CHT thermocouple, reported on cht_cyl_1.
        for name, ch, exp_val, scale in (
            ("cht", record.cht_cyl_1, exp_cht, self._cfg.scale_cht_k),
            ("fuel_flow", record.fuel_flow, exp_fuel, self._cfg.scale_fuel_kg_s),
            ("coolant", record.coolant_temp, exp_coolant, self._settings.coolant.scale_k),
        ):
            q_res = self.compute_residual(
                name=name,
                observed=ch.value if ch.valid else None,
                observed_valid=ch.valid,
                expected=exp_val,
                scale=scale,
                quality=ch.quality,
            )
            res_dict[name] = q_res
            if q_res.valid and q_res.raw_residual is not None:
                schema_residuals[name] = make_tagged(q_res.raw_residual, Provenance.DERIVED, valid=True, quality=q_res.quality)
            else:
                schema_residuals[name] = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0).model_copy(
                    update={"fault_flag": ch.fault_flag or f"{name} channel invalid"})

        oil_p_res = self.compute_residual(
            name="oil_pressure",
            observed=record.oil_pressure.value if record.oil_pressure.valid else None,
            observed_valid=record.oil_pressure.valid,
            expected=exp_oil_p,
            scale=self._cfg.scale_oil_pressure_pa,
            quality=record.oil_pressure.quality,
        )
        res_dict["oil_pressure"] = oil_p_res
        if oil_p_res.valid and oil_p_res.raw_residual is not None:
            schema_residuals["oil_pressure"] = make_tagged(oil_p_res.raw_residual, Provenance.DERIVED, valid=True, quality=oil_p_res.quality)
        else:
            schema_residuals["oil_pressure"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        oil_t_res = self.compute_residual(
            name="oil_temp",
            observed=record.oil_temp.value if record.oil_temp.valid else None,
            observed_valid=record.oil_temp.valid,
            expected=exp_oil_t,
            scale=self._cfg.scale_oil_temp_k,
            quality=record.oil_temp.quality,
        )
        res_dict["oil_temp"] = oil_t_res
        if oil_t_res.valid and oil_t_res.raw_residual is not None:
            schema_residuals["oil_temp"] = make_tagged(oil_t_res.raw_residual, Provenance.DERIVED, valid=True, quality=oil_t_res.quality)
        else:
            schema_residuals["oil_temp"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        vib_res = self.compute_residual(
            name="vibration_rms",
            observed=record.vibration_rms.value if record.vibration_rms.valid else None,
            observed_valid=record.vibration_rms.valid,
            expected=exp_vib,
            scale=self._cfg.scale_vibration_m_s2,
            quality=record.vibration_rms.quality,
        )
        res_dict["vibration_rms"] = vib_res
        if vib_res.valid and vib_res.raw_residual is not None:
            schema_residuals["vibration_rms"] = make_tagged(vib_res.raw_residual, Provenance.DERIVED, valid=True, quality=vib_res.quality)
        else:
            schema_residuals["vibration_rms"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        if derived_state is not None and derived_state.brake_power_kw.valid:
            obs_power = derived_state.brake_power_kw.value
            p_res = self.compute_residual(
                name="brake_power_kw",
                observed=obs_power,
                observed_valid=True,
                expected=exp_power,
                scale=self._cfg.scale_power_kw,
                quality=derived_state.brake_power_kw.quality,
            )
            res_dict["brake_power_kw"] = p_res
            if p_res.valid and p_res.raw_residual is not None:
                schema_residuals["brake_power_kw"] = make_tagged(p_res.raw_residual, Provenance.DERIVED, valid=True, quality=p_res.quality)
            else:
                schema_residuals["brake_power_kw"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        if injection_state is not None:
            inj = injection_state
            total = inj.fuel_cmd_total_kg_s.value if inj.fuel_cmd_total_kg_s.valid else 0.0
            exp_rail = self.expectation_model.expected_fuel_rail_dp_pa(op, total)
            expected_dict["fuel_rail_pressure"] = exp_rail
            quantities = [("fuel_rail_pressure",
                           inj.rail_dp_kpa.value * 1000.0 if inj.rail_dp_kpa.valid else None,
                           exp_rail, self._settings.injection.rail_residual_alarm_pa, inj.rail_dp_kpa)]
            exp_del = self.expectation_model.expected_fuel_delivery_ratio(op)
            expected_dict["fuel_delivery_ratio"] = exp_del
            quantities.append(("fuel_delivery_ratio", inj.fuel_delivery_ratio_median.value, exp_del,
                               self._settings.injection.fuel_delivery_alarm, inj.fuel_delivery_ratio_median))
            exp_flow = self.expectation_model.expected_injector_flow_ratio(op)
            for i, tv in enumerate(inj.injector_flow_ratio_median, start=1):
                expected_dict[f"injector_flow_ratio_cyl{i}"] = exp_flow
                quantities.append((f"injector_flow_ratio_cyl{i}", tv.value, exp_flow,
                                   self._settings.injection.injector_flow_alarm, tv))
            for name, observed, expected, scale, tv in quantities:
                q_res = self.compute_residual(name=name, observed=observed if tv.valid else None,
                                              observed_valid=tv.valid, expected=expected, scale=scale,
                                              quality=tv.quality)
                res_dict[name] = q_res
                if q_res.valid and q_res.raw_residual is not None:
                    schema_residuals[name] = make_tagged(q_res.raw_residual, Provenance.DERIVED, valid=True,
                                                         quality=q_res.quality)
                else:
                    schema_residuals[name] = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0).model_copy(
                        update={"fault_flag": tv.fault_flag or f"{name} not derivable"})

        if op_invalid:
            for name in list(res_dict):
                r = res_dict[name]
                res_dict[name] = r.model_copy(update={"raw_residual": None, "absolute_residual": None,
                                                      "normalized_residual": None, "valid": False, "quality": 0.0})
                schema_residuals[name] = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0).model_copy(
                    update={"fault_flag": f"operating point input invalid: {', '.join(op_invalid)}"})

        exp_result = HealthyExpectationResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            operating_point=op,
            expected_values=expected_dict,
            residuals=res_dict,
            raw_expected_values=raw_expected,
            baseline_version=self.baseline_version,
        )

        residual_state = ResidualState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            operating_point=op,
            residuals=schema_residuals,
        )

        return residual_state, exp_result


def evaluate_residual_engine(
    record: NormalizedSignalRecord,
    derived_state: DerivedEngineState | None = None,
    settings: AppSettings | None = None,
) -> tuple[ResidualState, HealthyExpectationResult]:
    """Convenience function for Module 11 residual engine evaluation."""
    engine = ResidualEngine(settings)
    return engine.evaluate(record, derived_state)
