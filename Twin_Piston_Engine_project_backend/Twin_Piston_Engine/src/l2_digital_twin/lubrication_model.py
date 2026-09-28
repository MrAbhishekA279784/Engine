"""
Lubrication Model — Original Module 8.

Fourth processing stage of L2 Digital Twin.
Derives physics-based lubrication state (viscosity, pressure margins, temperature margins,
operating status, and pressure-temperature interaction) from NormalizedSignalRecord
and optional DerivedEngineState.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (from Module 5) and optional DerivedEngineState (Module 6)
    - Output: LubricationState (canonical schema) and LubricationModelResult
    - Zero simulator internals or ground truth dependencies
    - Zero ML, Advisory, or API dependencies
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, LubricationConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import DerivedEngineState, LubricationState, ProvenanceTaggedValue, make_tagged
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.physics import M10_lubrication_vibration_indices as M10
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, operating_point_from_record

logger = get_logger(__name__)


def _severity_rank(status: DiagnosticStatus) -> int:
    """Helper for severity comparison."""
    ranks = {
        DiagnosticStatus.INVALID: 0,
        DiagnosticStatus.NORMAL: 1,
        DiagnosticStatus.WARNING: 2,
        DiagnosticStatus.CRITICAL: 3,
    }
    return ranks.get(status, 0)


def _max_status(s1: DiagnosticStatus, s2: DiagnosticStatus) -> DiagnosticStatus:
    """Return the status with higher severity."""
    if s1 == DiagnosticStatus.INVALID and s2 != DiagnosticStatus.INVALID:
        return s2
    if s2 == DiagnosticStatus.INVALID and s1 != DiagnosticStatus.INVALID:
        return s1
    return s1 if _severity_rank(s1) >= _severity_rank(s2) else s2


class LubricationModelResult(BaseModel):
    """Detailed result container for the lubrication physics model."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    oil_temperature_k: float | None
    oil_pressure_pa: float | None
    dynamic_viscosity_pa_s: float | None
    pressure_margin_pa: float | None
    temperature_margin_k: float | None
    expected_pressure_pa: float | None
    dp_dt_pa_s: float | None
    dt_dt_k_s: float | None
    status: DiagnosticStatus
    reasons: list[str] = Field(default_factory=list)
    lhi: float | None = None
    lhi_band: str = "UNKNOWN"
    lhi_coverage: float = 0.0
    lhi_components: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


VISCOSITY_NOT_MEASURED = "no measured viscosity channel; derived from oil temp"


class LubricationModel:
    """Deterministic physics-based lubrication model for Rotax 915 iS."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: LubricationConfig = self._settings.lubrication
        self._prev_timestamp: datetime | None = None
        self._prev_oil_p_pa: float | None = None
        self._prev_oil_t_k: float | None = None

        self._expectations = HealthyExpectationModel(self._settings)
        # Precalculate nominal viscosity at nominal temperature (90°C = 363.15 K)
        nom_t_k = self._cfg.nominal_temp_c + 273.15
        self._nom_viscosity_pa_s = self.compute_viscosity(nom_t_k) or 0.0433

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_oil_p_pa = None
        self._prev_oil_t_k = None

    def compute_viscosity(self, temp_k: float) -> float | None:
        """Compute dynamic oil viscosity using the Vogel equation:
        mu(T) = A * exp(B / (T - C)) [Pa*s].
        """
        if temp_k <= self._cfg.vogel_c:
            return None
        try:
            val = self._cfg.vogel_a * math.exp(self._cfg.vogel_b / (temp_k - self._cfg.vogel_c))
            return val if not (math.isnan(val) or math.isinf(val)) else None
        except (OverflowError, ValueError, ZeroDivisionError):
            return None

    def lubrication_health_index(
        self, record: NormalizedSignalRecord
    ) -> tuple[float | None, str, float, dict[str, dict[str, Any]], list[str], float]:
        """M-10 LHI, normalised to the value EXPECTED at the current oil temperature.

        Components (log-space product, M-10 form):
            pressure  = oil pressure / residual_engine.expected_oil_pressure_pa(op, T_oil)
            viscosity = EXCLUDED: valid=False, no measured viscosity channel; the
                        value is derived from oil temperature, so it carries no
                        information. Its value is reported, with zero weight.
            over_temp = M-10 one-sided derate, active only above 130 degC.

        Weights are re-normalised over the valid components in log space:
            LHI = exp( (W_all / W_valid) * sum_valid w_i ln f_i )
        which equals the M-10 product when every component is valid. Coverage
        = W_valid / W_all reports the missing share.

        Returns (lhi, band, coverage, components, evidence, quality).
        """
        c = self._cfg
        weights = {"pressure": c.lhi_weight_pressure, "viscosity": c.lhi_weight_viscosity,
                   "over_temp": c.lhi_weight_over_temp}
        comps: dict[str, dict[str, Any]] = {}
        oil_t, oil_p, rpm = record.oil_temp, record.oil_pressure, record.rpm
        t_ok = oil_t.valid and oil_t.value > 0.0

        # Pressure vs expectation at this rpm and oil temperature
        if t_ok and oil_p.valid and oil_p.value >= 0.0 and rpm.valid and rpm.value > 0.0:
            exp_p = self._expectations.expected_oil_pressure_pa(operating_point_from_record(record), oil_t.value)
            comps["pressure"] = {"factor": oil_p.value / exp_p, "valid": exp_p > 0.0,
                                 "observed_pa": oil_p.value, "expected_pa": exp_p}
        else:
            comps["pressure"] = {"factor": None, "valid": False, "reason": "oil pressure, oil temp or rpm invalid"}

        mu = self.compute_viscosity(oil_t.value) if t_ok else None
        comps["viscosity"] = {"factor": None, "valid": False, "reason": VISCOSITY_NOT_MEASURED,
                              "derived_value_pa_s": mu}

        if t_ok:
            comps["over_temp"] = {"factor": M10.over_temperature_derate(oil_t.value - 273.15, c.lhi_over_temp_limit_c),
                                  "valid": True}
        else:
            comps["over_temp"] = {"factor": None, "valid": False, "reason": "oil temperature invalid"}

        for name, comp in comps.items():
            comp["weight"] = weights[name] if comp["valid"] else 0.0

        w_all = sum(weights.values())
        w_valid = sum(comp["weight"] for comp in comps.values())
        coverage = w_valid / w_all if w_all > 0.0 else 0.0
        evidence = [
            f"viscosity excluded ({VISCOSITY_NOT_MEASURED}); derived value "
            + (f"{mu:.4f} Pa.s" if mu is not None else "unavailable") + ", weight 0"
        ]
        if w_valid <= 0.0 or not comps["pressure"]["valid"]:
            evidence.append("LHI not computed: oil pressure expectation unavailable")
            return None, "UNKNOWN", coverage, comps, evidence, 0.0

        log_sum = 0.0
        for name, comp in comps.items():
            if not comp["valid"]:
                continue
            if comp["factor"] <= 0.0:
                log_sum = float("-inf")
                break
            log_sum += comp["weight"] * math.log(comp["factor"])
        lhi = 0.0 if log_sum == float("-inf") else math.exp((w_all / w_valid) * log_sum)
        for name, comp in comps.items():
            if comp["valid"]:
                evidence.append(f"{name} factor {comp['factor']:.3f} (weight {comp['weight']:.2f})")
        evidence.append(f"coverage {coverage:.2f} (weights re-normalised over valid components)")
        quality = min(oil_t.quality, oil_p.quality, rpm.quality)
        return lhi, M10.lhi_band(lhi), coverage, comps, evidence, quality

    def evaluate(
        self,
        record: NormalizedSignalRecord,
        derived_state: DerivedEngineState | None = None,
    ) -> tuple[LubricationState, LubricationModelResult]:
        """Evaluate lubrication model on a NormalizedSignalRecord.

        Returns canonical LubricationState and detailed LubricationModelResult.
        """
        reasons: list[str] = []
        status = DiagnosticStatus.NORMAL

        # Calculate temporal dt
        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        # ---------------------------------------------------------------------
        # 1. Oil Temperature
        # ---------------------------------------------------------------------
        oil_t_ch = record.oil_temp
        oil_t_valid = oil_t_ch.valid and oil_t_ch.value > 0.0

        if oil_t_valid:
            oil_t_k = oil_t_ch.value
            oil_t_c = oil_t_k - 273.15
            oil_temp_tv = make_tagged(oil_t_k, Provenance.DERIVED, valid=True, quality=oil_t_ch.quality)

            # Temp margin relative to critical max (130°C = 403.15 K)
            crit_max_k = self._cfg.crit_high_temp_c + 273.15
            temp_margin_k = crit_max_k - oil_t_k
            temp_margin_tv = make_tagged(temp_margin_k, Provenance.DERIVED, valid=True, quality=oil_t_ch.quality)

            # Viscosity calculation
            viscosity_pa_s = self.compute_viscosity(oil_t_k)
            if viscosity_pa_s is not None:
                viscosity_tv = make_tagged(viscosity_pa_s, Provenance.DERIVED, valid=True, quality=oil_t_ch.quality)
            else:
                viscosity_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

            # Temp status check
            if oil_t_c >= self._cfg.crit_high_temp_c or oil_t_c <= self._cfg.crit_low_temp_c:
                status = _max_status(status, DiagnosticStatus.CRITICAL)
                reasons.append(f"Oil temperature ({oil_t_c:.1f}°C) at critical threshold")
            elif oil_t_c >= self._cfg.warn_high_temp_c or oil_t_c <= self._cfg.warn_low_temp_c:
                status = _max_status(status, DiagnosticStatus.WARNING)
                reasons.append(f"Oil temperature ({oil_t_c:.1f}°C) at warning threshold")
        else:
            oil_t_k = None
            oil_temp_tv = make_tagged(363.15, Provenance.DERIVED, valid=False, quality=0.0)
            temp_margin_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            viscosity_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            viscosity_pa_s = None

        # ---------------------------------------------------------------------
        # 2. Oil Pressure
        # ---------------------------------------------------------------------
        oil_p_ch = record.oil_pressure
        oil_p_valid = oil_p_ch.valid and oil_p_ch.value >= 0.0

        if oil_p_valid:
            oil_p_pa = oil_p_ch.value
            oil_p_bar = oil_p_pa / 100000.0
            oil_p_tv = make_tagged(oil_p_pa, Provenance.DERIVED, valid=True, quality=oil_p_ch.quality)

            # Pressure margin relative to critical min (1.5 bar = 150,000 Pa)
            crit_min_pa = self._cfg.crit_low_pressure_bar * 100000.0
            press_margin_pa = oil_p_pa - crit_min_pa
            press_margin_tv = make_tagged(press_margin_pa, Provenance.DERIVED, valid=True, quality=oil_p_ch.quality)

            # Pressure status check
            if oil_p_bar <= self._cfg.crit_low_pressure_bar or oil_p_bar >= self._cfg.crit_high_pressure_bar:
                status = _max_status(status, DiagnosticStatus.CRITICAL)
                reasons.append(f"Oil pressure ({oil_p_bar:.2f} bar) at critical threshold")
            elif oil_p_bar <= self._cfg.warn_low_pressure_bar or oil_p_bar >= self._cfg.warn_high_pressure_bar:
                status = _max_status(status, DiagnosticStatus.WARNING)
                reasons.append(f"Oil pressure ({oil_p_bar:.2f} bar) at warning threshold")
        else:
            oil_p_pa = None
            oil_p_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            press_margin_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        if not oil_t_valid and not oil_p_valid:
            status = DiagnosticStatus.INVALID
            reasons.append("Both oil temperature and oil pressure telemetry are invalid")

        # ---------------------------------------------------------------------
        # 3. Pressure-Temperature-RPM Interaction
        # ---------------------------------------------------------------------
        expected_p_pa: float | None = None
        rpm_ch = record.rpm
        if oil_p_valid and oil_t_valid and viscosity_pa_s is not None and rpm_ch.valid and rpm_ch.value > 0:
            rpm_val = rpm_ch.value
            nom_p_pa = self._cfg.nominal_pressure_bar * 100000.0
            # Hydrodynamic pressure correlation: P_exp = P_nom * sqrt(N / N_nom) * (mu / mu_nom)^0.3
            expected_p_pa = nom_p_pa * math.sqrt(rpm_val / 4000.0) * math.pow(viscosity_pa_s / self._nom_viscosity_pa_s, 0.3)

            if oil_p_pa is not None and expected_p_pa > 0:
                p_ratio = oil_p_pa / expected_p_pa
                if p_ratio < 0.60:
                    status = _max_status(status, DiagnosticStatus.WARNING)
                    reasons.append(f"Oil pressure ({oil_p_pa / 1e5:.2f} bar) significantly below expected ({expected_p_pa / 1e5:.2f} bar) for current viscosity and RPM")

        # ---------------------------------------------------------------------
        # 4. Temporal Rates of Change
        # ---------------------------------------------------------------------
        dp_dt: float | None = None
        dt_dt: float | None = None

        if dt_s is not None and 0.0 < dt_s <= 10.0:
            if oil_p_valid and self._prev_oil_p_pa is not None:
                dp_dt = (oil_p_pa - self._prev_oil_p_pa) / dt_s
            if oil_t_valid and self._prev_oil_t_k is not None:
                dt_dt = (oil_t_k - self._prev_oil_t_k) / dt_s

        # Update temporal state
        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        if oil_p_valid:
            self._prev_oil_p_pa = oil_p_pa
        if oil_t_valid:
            self._prev_oil_t_k = oil_t_k

        lhi, lhi_band, lhi_coverage, lhi_comps, lhi_evidence, lhi_q = self.lubrication_health_index(record)

        # Build output objects
        lub_result = LubricationModelResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            oil_temperature_k=oil_t_k,
            oil_pressure_pa=oil_p_pa,
            dynamic_viscosity_pa_s=viscosity_pa_s,
            pressure_margin_pa=oil_p_pa - (self._cfg.crit_low_pressure_bar * 100000.0) if oil_p_valid else None,
            temperature_margin_k=(self._cfg.crit_high_temp_c + 273.15) - oil_t_k if oil_t_valid else None,
            expected_pressure_pa=expected_p_pa,
            dp_dt_pa_s=dp_dt,
            dt_dt_k_s=dt_dt,
            status=status,
            reasons=reasons,
            lhi=lhi,
            lhi_band=lhi_band,
            lhi_coverage=lhi_coverage,
            lhi_components=lhi_comps,
        )

        lub_state = LubricationState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            oil_temperature_k=oil_temp_tv,
            oil_pressure_pa=oil_p_tv,
            dynamic_viscosity_pa_s=viscosity_tv,
            pressure_margin_pa=press_margin_tv,
            temperature_margin_k=temp_margin_tv,
            status=status,
            lhi=make_tagged(lhi, Provenance.DERIVED, valid=lhi is not None, quality=lhi_q if lhi is not None else 0.0),
            lhi_band=lhi_band,
            lhi_coverage=lhi_coverage,
            lhi_evidence=lhi_evidence,
        )

        return lub_state, lub_result


def evaluate_lubrication_model(
    record: NormalizedSignalRecord,
    derived_state: DerivedEngineState | None = None,
    settings: AppSettings | None = None,
) -> tuple[LubricationState, LubricationModelResult]:
    """Convenience function for evaluating Module 8 Lubrication Model."""
    model = LubricationModel(settings)
    return model.evaluate(record, derived_state)
