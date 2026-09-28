"""
Thermodynamic and Mechanical Digital Twin — Original Module 6.

Second processing stage of L2 Digital Twin.
Derives thermodynamic and mechanical engine parameters from engineering-unit
NormalizedSignalRecord produced by Module 5.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (from Module 5)
    - Output: DerivedEngineState (canonical domain schema)
    - Zero simulator internals or ground truth dependencies
    - Zero ML, Advisory, or API dependencies
    - Physical derivations are independent of simulator generator equations
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, Provenance
from src.core.constants import R_AIR
from src.core.schemas import DerivedEngineState, ProvenanceTaggedValue, make_tagged
from src.core.units import isa_density_altitude_m, isa_pressure_altitude_m, isa_temperature
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.lubrication_model import LubricationModel
from src.l2_digital_twin.physics.D03_lambda_derivation import (
    combustion_efficiency,
    lambda_from_mass_flows,
)
from src.l2_digital_twin.physics.D04_air_mass_flow_speed_density import (
    manifold_charge_density,
    solve_air_flow,
)
from src.l2_digital_twin.physics.M07_specific_fuel_consumption import (
    sfc_band,
    sfc_degradation_pct,
    specific_fuel_consumption_kg_kwh,
)
from src.l2_digital_twin.physics.M08_normalised_engine_load import load_band, normalised_engine_load

logger = get_logger(__name__)

# Physical Constants
AIR_GAS_CONSTANT_J_KG_K = 287.058  # Specific gas constant for dry air
STANDARD_SEA_LEVEL_PRESSURE_PA = 101325.0
STANDARD_SEA_LEVEL_TEMP_K = 288.15
CHARGE_TEMP_RISE_K = 28.0  # Post-intercooler charge rise over ambient (D04 default)


class EngineGeometry:
    """Calculates and encapsulates Rotax 915 iS engine geometry."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        cfg = (settings or get_settings()).engine
        self.bore_m: float = cfg.bore_mm / 1000.0
        self.stroke_m: float = cfg.stroke_mm / 1000.0
        self.displacement_m3: float = cfg.displacement_cc * 1e-6
        self.num_cylinders: int = cfg.num_cylinders
        self.compression_ratio: float = cfg.compression_ratio
        self.conrod_length_m: float = cfg.connecting_rod_mm / 1000.0
        self.rated_power_kw: float = cfg.rated_power_kw
        self.rated_rpm: int = cfg.rated_rpm
        self.lhv_j_kg: float = cfg.lhv_mj_per_kg * 1e6
        self.stoichiometric_afr: float = cfg.stoichiometric_afr
        self.sfc_baseline_kg_kwh: float = cfg.sfc_baseline_kg_kwh
        self.eta_mech_plausible_min: float = cfg.eta_mech_plausible_min
        self.eta_mech_plausible_max: float = cfg.eta_mech_plausible_max
        self.fmep_bm_a_bar: float = cfg.fmep_barnes_moss_a_bar
        self.fmep_bm_b_bar: float = cfg.fmep_barnes_moss_b_bar
        self.fmep_bm_c_bar: float = cfg.fmep_barnes_moss_c_bar
        self.fmep_viscosity_exponent: float = cfg.fmep_viscosity_exponent

        # Derived geometry
        self.cylinder_displacement_m3: float = self.displacement_m3 / self.num_cylinders
        self.cylinder_area_m2: float = (math.pi / 4.0) * (self.bore_m ** 2)
        self.crank_radius_m: float = self.stroke_m / 2.0
        self.conrod_ratio: float = self.crank_radius_m / self.conrod_length_m

        # Clearance volume per cylinder: V_c = V_cyl / (r_c - 1)
        self.clearance_volume_m3: float = self.cylinder_displacement_m3 / (self.compression_ratio - 1.0)

    def mean_piston_speed(self, rpm: float) -> float:
        """Mean piston speed S_p = 2 * stroke * (RPM / 60) [m/s]."""
        if rpm < 0:
            return 0.0
        return 2.0 * self.stroke_m * (rpm / 60.0)

    def max_piston_speed(self, rpm: float) -> float:
        """Peak piston speed V_p,max = (pi/2) * S_p * (1 + lambda_crank) [m/s]."""
        s_p = self.mean_piston_speed(rpm)
        return (math.pi / 2.0) * s_p * (1.0 + self.conrod_ratio)


class ThermodynamicResults(BaseModel):
    """Detailed thermodynamic twin intermediate results.

    Causal direction of the air path (never the reverse):
        MAP + rpm + eta_vol  -> air mass flow   (speed density)
        air + measured fuel  -> lambda, AFR
        lambda               -> combustion efficiency

    eta_volumetric is the breathing model's EXPECTED value at this operating
    point (rpm, MAP, ambient), not a measurement, and must not be read as a
    health indicator. A real breathing or fuelling deviation (e.g. ring
    blow-by) shows up as a change in lambda. Undefined quantities carry
    value=None with valid=False.
    """

    boost_pressure_pa: ProvenanceTaggedValue[float]
    pressure_ratio: ProvenanceTaggedValue[float]
    air_density_kg_m3: ProvenanceTaggedValue[float | None]
    air_mass_flow_kg_s: ProvenanceTaggedValue[float | None]
    fuel_energy_rate_w: ProvenanceTaggedValue[float]
    afr: ProvenanceTaggedValue[float | None]
    eta_volumetric: ProvenanceTaggedValue[float | None]
    lambda_derived: ProvenanceTaggedValue[float | None]
    combustion_efficiency: ProvenanceTaggedValue[float | None]

    model_config = ConfigDict(frozen=True)


class MechanicalResults(BaseModel):
    """Detailed mechanical twin intermediate results."""

    crank_angular_velocity_rad_s: ProvenanceTaggedValue[float]
    crank_angular_acceleration_rad_s2: ProvenanceTaggedValue[float]
    mean_piston_speed_m_s: ProvenanceTaggedValue[float]
    max_piston_speed_m_s: ProvenanceTaggedValue[float]
    brake_torque_nm: ProvenanceTaggedValue[float]
    brake_power_kw: ProvenanceTaggedValue[float]
    fmep_pa: ProvenanceTaggedValue[float]
    # eta_mech = (IMEP - FMEP) / IMEP. Outside the plausible window the value
    # is kept but quality is lowered to ETA_MECH_IMPLAUSIBLE_QUALITY.
    eta_mech: ProvenanceTaggedValue[float | None]
    sfc_kg_kwh: ProvenanceTaggedValue[float | None]
    sfc_band: str
    sfc_degradation_pct: ProvenanceTaggedValue[float | None]
    engine_load: ProvenanceTaggedValue[float | None]
    load_band: str

    model_config = ConfigDict(frozen=True)


ETA_MECH_IMPLAUSIBLE_QUALITY = 0.3


class ThermodynamicTwin:
    """Thermodynamic Digital Twin calculations."""

    def __init__(self, geometry: EngineGeometry) -> None:
        self.geom = geometry

    def compute(self, record: NormalizedSignalRecord) -> ThermodynamicResults:
        """Derive thermodynamic parameters from normalized telemetry."""
        # 1. Boost Pressure & Pressure Ratio
        p_map = record.map_pressure
        p_amb = record.ambient_pressure

        if p_map.valid and p_amb.valid and p_amb.value > 0:
            boost = p_map.value - p_amb.value
            pr = p_map.value / p_amb.value
            boost_tv = make_tagged(boost, Provenance.DERIVED, valid=True, quality=min(p_map.quality, p_amb.quality))
            pr_tv = make_tagged(pr, Provenance.DERIVED, valid=True, quality=min(p_map.quality, p_amb.quality))
        else:
            boost_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            pr_tv = make_tagged(1.0, Provenance.DERIVED, valid=False, quality=0.0)

        # 2. Air path by speed density (primary; never derived from fuel flow)
        #    m_dot_air = eta_vol(rpm, MAP, ambient) * rho_charge * V_d * rpm / 120
        #    rpm/120: a four-stroke takes one charge per cylinder every two revs.
        #    Charge temperature = ambient + intercooler rise (D04); there is no
        #    measured intake-air-temperature channel in RawSignalRecord.
        rpm = record.rpm
        t_amb = record.ambient_temp
        air_inputs = (rpm, p_map, p_amb, t_amb)
        undefined = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0)
        if all(ch.valid for ch in air_inputs):
            m_dot_air, eta_v, _, air_ok, _ = solve_air_flow(
                rpm=rpm.value,
                map_pa=p_map.value,
                ambient_pa=p_amb.value,
                ambient_temp_k=t_amb.value,
                displacement_m3=self.geom.displacement_m3,
                charge_temp_rise_k=CHARGE_TEMP_RISE_K,
            )
        else:
            m_dot_air, eta_v, air_ok = 0.0, 0.0, False

        if air_ok and m_dot_air > 0.0:
            q_air = min(ch.quality for ch in air_inputs)
            m_dot_air_tv = make_tagged(m_dot_air, Provenance.DERIVED, valid=True, quality=q_air)
            eta_v_tv = make_tagged(eta_v, Provenance.DERIVED, valid=True, quality=q_air)
        else:
            m_dot_air_tv = undefined
            eta_v_tv = undefined

        # Charge density at the same charge temperature the air flow used.
        rho = (
            manifold_charge_density(p_map.value, t_amb.value + CHARGE_TEMP_RISE_K)
            if (p_map.valid and t_amb.valid)
            else 0.0
        )
        if rho > 0.0:
            rho_tv = make_tagged(rho, Provenance.DERIVED, valid=True, quality=min(p_map.quality, t_amb.quality))
        else:
            rho_tv = undefined

        # 3. Fuel Energy Rate
        fuel = record.fuel_flow
        if fuel.valid and fuel.value >= 0:
            q_in = fuel.value * self.geom.lhv_j_kg
            q_in_tv = make_tagged(q_in, Provenance.DERIVED, valid=True, quality=fuel.quality)
        else:
            q_in_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # 4. Lambda, AFR and combustion efficiency from the mass flows (D-03).
        #    RawSignalRecord has no lambda channel; record.lambda_sensor is not
        #    used. Zero fuel (idle cut-off) leaves lambda undefined, not inf/0.
        lam_val = None
        if m_dot_air_tv.valid and fuel.valid:
            lam_val = lambda_from_mass_flows(
                m_dot_air_tv.value, fuel.value, self.geom.stoichiometric_afr
            )
        if lam_val is not None and math.isfinite(lam_val) and lam_val > 0.0:
            q_lam = min(m_dot_air_tv.quality, fuel.quality)
            lambda_tv = make_tagged(lam_val, Provenance.DERIVED, valid=True, quality=q_lam)
            afr_tv = make_tagged(
                lam_val * self.geom.stoichiometric_afr, Provenance.DERIVED, valid=True, quality=q_lam
            )
            eta_c = combustion_efficiency(lam_val)
            eta_c_tv = (
                make_tagged(eta_c, Provenance.DERIVED, valid=True, quality=q_lam)
                if eta_c is not None
                else undefined
            )
        else:
            lambda_tv = undefined
            afr_tv = undefined
            eta_c_tv = undefined

        return ThermodynamicResults(
            boost_pressure_pa=boost_tv,
            pressure_ratio=pr_tv,
            air_density_kg_m3=rho_tv,
            air_mass_flow_kg_s=m_dot_air_tv,
            fuel_energy_rate_w=q_in_tv,
            afr=afr_tv,
            eta_volumetric=eta_v_tv,
            lambda_derived=lambda_tv,
            combustion_efficiency=eta_c_tv,
        )


FMEP_UNSCALED_QUALITY = 0.5  # FMEP quality cap when oil viscosity is unavailable


class MechanicalTwin:
    """Mechanical Digital Twin calculations."""

    def __init__(self, geometry: EngineGeometry, settings: AppSettings | None = None) -> None:
        self.geom = geometry
        self._lub = LubricationModel(settings)
        self._mu_ref = self._lub.compute_viscosity(
            (settings or get_settings()).lubrication.nominal_temp_c + 273.15
        )

    def compute(
        self,
        record: NormalizedSignalRecord,
        prev_omega: float | None = None,
        dt_s: float | None = None,
    ) -> MechanicalResults:
        """Derive mechanical parameters from normalized telemetry."""
        rpm = record.rpm
        if rpm.valid and rpm.value >= 0:
            omega = (2.0 * math.pi * rpm.value) / 60.0
            omega_tv = make_tagged(omega, Provenance.DERIVED, valid=True, quality=rpm.quality)
            s_p = self.geom.mean_piston_speed(rpm.value)
            s_p_tv = make_tagged(s_p, Provenance.DERIVED, valid=True, quality=rpm.quality)
            v_p_max = self.geom.max_piston_speed(rpm.value)
            v_p_max_tv = make_tagged(v_p_max, Provenance.DERIVED, valid=True, quality=rpm.quality)
        else:
            omega_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            s_p_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            v_p_max_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # Angular acceleration alpha = d(omega)/dt
        if omega_tv.valid and prev_omega is not None and dt_s is not None and 0.0 < dt_s <= 10.0:
            alpha = (omega_tv.value - prev_omega) / dt_s
            alpha_tv = make_tagged(alpha, Provenance.DERIVED, valid=True, quality=rpm.quality)
        else:
            alpha_tv = make_tagged(0.0, Provenance.DERIVED, valid=True, quality=1.0)

        # FMEP: Barnes-Moss total motored friction correlation for 4-stroke SI
        # engines, J. B. Heywood, Internal Combustion Engine Fundamentals (1988),
        # ch. 13. VERIFY: equation number not confirmed against the book.
        #     fmep [bar] = 0.97 + 0.15 (N/1000) + 0.05 (N/1000)^2,  N [rev/min]
        # Scaled by (mu_oil / mu_ref)^0.25 when oil viscosity is available
        # (mu_ref at the nominal oil temperature); otherwise unscaled with
        # quality capped at 0.5. Chen-Flynn is not used: it needs peak cylinder
        # pressure, which L2 does not have. Checked through eta_mech below.
        if rpm.valid and rpm.value >= 0:
            n_k = rpm.value / 1000.0
            fmep = 1e5 * (self.geom.fmep_bm_a_bar + self.geom.fmep_bm_b_bar * n_k
                          + self.geom.fmep_bm_c_bar * n_k * n_k)
            oil_t = record.oil_temp
            mu = self._lub.compute_viscosity(oil_t.value) if (oil_t.valid and oil_t.value > 0) else None
            if mu is not None and self._mu_ref:
                fmep *= (mu / self._mu_ref) ** self.geom.fmep_viscosity_exponent
                q_fmep = min(rpm.quality, oil_t.quality)
            else:
                q_fmep = min(rpm.quality, FMEP_UNSCALED_QUALITY)
            fmep_tv = make_tagged(fmep, Provenance.DERIVED, valid=True, quality=q_fmep)
        else:
            fmep_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # Torque & Power derivations
        fuel = record.fuel_flow
        p_map = record.map_pressure

        bmep = 0.0
        imep = 0.0
        power_kw = 0.0
        torque_nm = 0.0
        power_valid = False
        power_quality = 0.0

        if rpm.valid and rpm.value > 0:
            if fuel.valid and fuel.value > 0:
                q_in = fuel.value * self.geom.lhv_j_kg
                p_i = q_in * 0.42  # Indicated thermal efficiency ~42%
                imep = (120.0 * p_i) / (self.geom.displacement_m3 * rpm.value)
                bmep = imep - fmep_tv.value if fmep_tv.valid else imep * 0.85
                power_w = (bmep * self.geom.displacement_m3 * rpm.value) / 120.0
                power_kw = power_w / 1000.0
                torque_nm = (bmep * self.geom.displacement_m3) / (4.0 * math.pi)
                power_valid = True
                power_quality = min(rpm.quality, fuel.quality, fmep_tv.quality)
            elif (
                p_map.valid and p_map.value > 0
                and record.ambient_pressure.valid and record.ambient_temp.valid
            ):
                # Fuel flow unavailable: estimate fuel from the speed-density air
                # path, assuming stoichiometric fuelling (quality-limited).
                m_dot_air_est = solve_air_flow(
                    rpm=rpm.value,
                    map_pa=p_map.value,
                    ambient_pa=record.ambient_pressure.value,
                    ambient_temp_k=record.ambient_temp.value,
                    displacement_m3=self.geom.displacement_m3,
                    charge_temp_rise_k=CHARGE_TEMP_RISE_K,
                )[0]
                m_dot_fuel_est = m_dot_air_est / self.geom.stoichiometric_afr
                q_in = m_dot_fuel_est * self.geom.lhv_j_kg
                p_i = q_in * 0.42
                imep = (120.0 * p_i) / (self.geom.displacement_m3 * rpm.value)
                bmep = imep - fmep_tv.value if fmep_tv.valid else imep * 0.85
                power_w = (bmep * self.geom.displacement_m3 * rpm.value) / 120.0
                power_kw = power_w / 1000.0
                torque_nm = (bmep * self.geom.displacement_m3) / (4.0 * math.pi)
                power_valid = True
                power_quality = min(0.7, fmep_tv.quality)  # quality-limited estimate

        power_tv = make_tagged(power_kw if power_valid else 0.0, Provenance.DERIVED, valid=power_valid, quality=power_quality)
        torque_tv = make_tagged(torque_nm if power_valid else 0.0, Provenance.DERIVED, valid=power_valid, quality=power_quality)
        undefined = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0)

        # Mechanical efficiency and friction-model plausibility check.
        if power_valid and fmep_tv.valid and imep > 0.0:
            eta_mech = (imep - fmep_tv.value) / imep
            q_eta = power_quality
            lo, hi = self.geom.eta_mech_plausible_min, self.geom.eta_mech_plausible_max
            if not lo <= eta_mech <= hi:
                q_eta = min(q_eta, ETA_MECH_IMPLAUSIBLE_QUALITY)
                logger.warning(
                    f"Implausible mechanical efficiency {eta_mech:.3f} outside [{lo}, {hi}] "
                    f"at {rpm.value:.0f} rpm (IMEP {imep:.0f} Pa, FMEP {fmep_tv.value:.0f} Pa); "
                    "friction model suspect"
                )
            eta_mech_tv = make_tagged(eta_mech, Provenance.DERIVED, valid=True, quality=q_eta)
        else:
            eta_mech_tv = undefined

        # M-07 SFC: measured fuel flow only. In the no-fuel fallback the fuel
        # is itself estimated from air at stoichiometric, so SFC would be circular.
        sfc_val = None
        if power_valid and fuel.valid and fuel.value > 0:
            sfc_val = specific_fuel_consumption_kg_kwh(fuel.value, power_kw)
        if sfc_val is not None:
            q_sfc = min(power_quality, fuel.quality)
            sfc_tv = make_tagged(sfc_val, Provenance.DERIVED, valid=True, quality=q_sfc)
            deg = sfc_degradation_pct(sfc_val, self.geom.sfc_baseline_kg_kwh)
            sfc_deg_tv = make_tagged(deg, Provenance.DERIVED, valid=True, quality=q_sfc) if deg is not None else undefined
        else:
            sfc_tv = undefined
            sfc_deg_tv = undefined

        # M-08 normalised load, clipped at 1.2 so above-rated stays visible.
        load_val = normalised_engine_load(power_kw, self.geom.rated_power_kw) if power_valid else None
        load_tv = (
            make_tagged(load_val, Provenance.DERIVED, valid=True, quality=power_quality)
            if load_val is not None
            else undefined
        )

        return MechanicalResults(
            crank_angular_velocity_rad_s=omega_tv,
            crank_angular_acceleration_rad_s2=alpha_tv,
            mean_piston_speed_m_s=s_p_tv,
            max_piston_speed_m_s=v_p_max_tv,
            brake_torque_nm=torque_tv,
            brake_power_kw=power_tv,
            fmep_pa=fmep_tv,
            eta_mech=eta_mech_tv,
            sfc_kg_kwh=sfc_tv,
            sfc_band=sfc_band(sfc_val),
            sfc_degradation_pct=sfc_deg_tv,
            engine_load=load_tv,
            load_band=load_band(load_val),
        )


class ThermodynamicMechanicalTwin:
    """Integrated L2 Digital Twin for Original Module 6."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.geometry = EngineGeometry(self._settings)
        self.thermo_twin = ThermodynamicTwin(self.geometry)
        self.mech_twin = MechanicalTwin(self.geometry, self._settings)
        self._prev_timestamp: datetime | None = None
        self._prev_omega: float | None = None

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_omega = None

    def evaluate(self, record: NormalizedSignalRecord) -> DerivedEngineState:
        """Evaluate digital twin from normalized telemetry to produce DerivedEngineState."""
        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        thermo_res = self.thermo_twin.compute(record)
        mech_res = self.mech_twin.compute(record, prev_omega=self._prev_omega, dt_s=dt_s)

        # Update temporal state
        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        if mech_res.crank_angular_velocity_rad_s.valid:
            self._prev_omega = mech_res.crank_angular_velocity_rad_s.value

        # Calculate BMEP, IMEP, Thermal Efficiency, Mechanical Efficiency
        rpm = record.rpm
        fmep_tv = mech_res.fmep_pa
        power_tv = mech_res.brake_power_kw
        q_in_tv = thermo_res.fuel_energy_rate_w

        if rpm.valid and rpm.value > 0 and power_tv.valid and power_tv.value > 0:
            bmep_val = (120.0 * (power_tv.value * 1000.0)) / (self.geometry.displacement_m3 * rpm.value)
            bmep_tv = make_tagged(bmep_val, Provenance.DERIVED, valid=True, quality=power_tv.quality)
            imep_val = bmep_val + fmep_tv.value if fmep_tv.valid else bmep_val / 0.85
            imep_tv = make_tagged(imep_val, Provenance.DERIVED, valid=True, quality=power_tv.quality)

            if q_in_tv.valid and q_in_tv.value > 0:
                eta_th_val = (power_tv.value * 1000.0) / q_in_tv.value
                valid_eta_th = 0.0 <= eta_th_val <= 1.0
                eta_th_tv = make_tagged(eta_th_val if valid_eta_th else 0.0, Provenance.DERIVED, valid=valid_eta_th, quality=power_tv.quality if valid_eta_th else 0.0)
            else:
                eta_th_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

            eta_mech_tv = mech_res.eta_mech
        else:
            bmep_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            imep_tv = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0)
            eta_th_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            eta_mech_tv = mech_res.eta_mech

        p_alt_tv, isa_dev_tv, d_alt_tv = environment_parameters(record)

        return DerivedEngineState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            bmep_pa=bmep_tv,
            imep_pa=imep_tv,
            eta_thermal=eta_th_tv,
            eta_volumetric=thermo_res.eta_volumetric,
            afr=thermo_res.afr,
            lambda_derived=thermo_res.lambda_derived,
            combustion_efficiency=thermo_res.combustion_efficiency,
            brake_torque_nm=mech_res.brake_torque_nm,
            brake_power_kw=mech_res.brake_power_kw,
            fmep_pa=fmep_tv,
            eta_mechanical=eta_mech_tv,
            mean_piston_speed_m_s=mech_res.mean_piston_speed_m_s,
            sfc_kg_kwh=mech_res.sfc_kg_kwh,
            sfc_band=mech_res.sfc_band,
            sfc_degradation_pct=mech_res.sfc_degradation_pct,
            engine_load=mech_res.engine_load,
            load_band=mech_res.load_band,
            pressure_altitude_m=p_alt_tv,
            isa_deviation_k=isa_dev_tv,
            density_altitude_m=d_alt_tv,
        )


def environment_parameters(
    record: NormalizedSignalRecord,
) -> tuple[ProvenanceTaggedValue[float | None], ProvenanceTaggedValue[float | None], ProvenanceTaggedValue[float | None]]:
    """(pressure altitude m, ISA deviation K, density altitude m), ISO 2533 ISA.

    ISA deviation = OAT - T_ISA(pressure altitude): positive on a hot day.
    Density altitude: ISA altitude whose density equals rho = p / (R T).
    """
    p, t = record.ambient_pressure, record.ambient_temp
    undefined = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0)
    p_ok = p.valid and p.value is not None and p.value > 0.0
    t_ok = t.valid and t.value is not None and t.value > 0.0
    p_alt = isa_pressure_altitude_m(p.value) if p_ok else None
    p_alt_tv = make_tagged(p_alt, Provenance.DERIVED, valid=True, quality=p.quality) if p_alt is not None else undefined
    if p_alt is None or not t_ok:
        return p_alt_tv, undefined, undefined
    q = min(p.quality, t.quality)
    isa_dev = t.value - isa_temperature(p_alt)
    d_alt = isa_density_altitude_m(p.value / (R_AIR * t.value))
    isa_tv = make_tagged(isa_dev, Provenance.DERIVED, valid=True, quality=q)
    d_alt_tv = make_tagged(d_alt, Provenance.DERIVED, valid=True, quality=q) if d_alt is not None else undefined
    return p_alt_tv, isa_tv, d_alt_tv


def evaluate_digital_twin(
    record: NormalizedSignalRecord,
    settings: AppSettings | None = None,
) -> DerivedEngineState:
    """Convenience function to evaluate digital twin on a NormalizedSignalRecord."""
    twin = ThermodynamicMechanicalTwin(settings)
    return twin.evaluate(record)
