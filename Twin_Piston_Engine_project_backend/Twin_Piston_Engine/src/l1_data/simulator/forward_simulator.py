"""
Forward Physics Model, Fault Injection and Scenario Simulator — Original Module 17.

First stage of L1 Forward Simulation & Scenario Generation.
Provides deterministic forward physics modeling, sensor forward conversion to canonical
RawSignalRecord, controlled 9-class fault injection, and scenario replay capabilities.

STRICT BOUNDARY CONSTRAINTS:
    - Simulator is a source of simulated telemetry ONLY.
    - Ground truth physical state remains in SimulationGroundTruth and MUST NOT be inserted into RawSignalRecord.
    - L2 and L3 MUST NOT import or access simulator ground truth or private internals.
    - Emits canonical RawSignalRecord with Provenance.SIMULATED.
    - 100% deterministic replay given identical scenario parameters and seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import FaultClass, FlightPhase, Provenance
from src.core.schemas import SignalQuality
from src.core.sensor_physics import ntc_resistance_ohm, pt100_resistance_ohms, type_k_emf_uv
from src.core.units import isa_pressure, isa_temperature
from src.l1_data.simulator.vibration_burst import (
    BURST_FS_HZ,
    BURST_SAMPLES,
    CRANK_BURST_REVS,
    accel_to_counts,
    generate_accel_burst,
    generate_crank_burst,
)
from src.l1_data.raw_signal_record import RawSignalRecord

logger = get_logger(__name__)


@dataclass
class SimulationGroundTruth:
    """Isolated, simulation-only ground truth state.

    MUST NOT be inserted into RawSignalRecord or exposed to L2/L3.
    """

    timestamp: datetime
    time_s: float
    sequence_number: int
    rpm: float
    map_pressure_pa: float
    throttle_pct: float
    altitude_m: float
    ambient_temp_k: float
    ambient_pressure_pa: float
    egt_k: list[float] = field(default_factory=lambda: [950.0, 950.0, 950.0, 950.0])
    cht_k: float = 383.15  # 110 °C
    oil_temp_k: float = 363.15  # 90 °C
    oil_pressure_pa: float = 400000.0  # 4 bar
    vibration_rms_m_s2: float = 5.0
    vibration_rms_base_m_s2: float | None = None  # pre-fault level; None if unknown
    # IMBALANCE: propeller-shaft 1X frequency and peak acceleration per axis (x axial, y lateral, z vertical)
    imbalance_hz: float | None = None
    imbalance_accel_peak_m_s2: tuple[float, float, float] | None = None
    electrical: Any = None  # ElectricalTruth when the electrical model runs (ScenarioRunner)
    injection: Any = None   # InjectionTruth (ECU schedules, rail, per-cylinder fuel), Prompt 13
    coolant: Any = None     # CoolantTruth (liquid loop), Prompt 14; None without a coolant circuit
    airspeed_m_s: float | None = None
    vibration_exc_g: float = 0.5
    fuel_flow_kg_s: float = 0.005
    brake_power_kw: float = 80.0
    torque_nm: float = 185.0
    air_mass_flow_kg_s: float | None = None
    lambda_cmd: float | None = None
    imep_pa: float | None = None
    fmep_pa: float | None = None
    active_fault: FaultClass = FaultClass.NOMINAL
    injected_fault_class: FaultClass = FaultClass.NOMINAL
    injected_fault_severity: float = 0.0
    affected_cylinders: list[int] = field(default_factory=lambda: [1])
    provenance: Provenance = Provenance.SIMULATED

    @property
    def power_kw(self) -> float:
        return self.brake_power_kw


@dataclass
class FaultScenarioConfig:
    """Configuration for a single fault injection scenario event."""

    fault_class: FaultClass = FaultClass.NOMINAL
    severity: float = 0.0  # Bounded [0.0, 1.0]
    onset_time_s: float = 0.0  # Start time in simulation [s]
    start_time_s: float | None = None  # Alias for onset_time_s
    duration_s: float = 300.0  # Duration of fault event [s]
    affected_cylinders: list[int] = field(default_factory=lambda: [1])
    affected_cylinder: int | None = None  # Alias for single cylinder
    affected_channel: str | None = None  # For SENSOR_FAULT
    # CHARGING_FAULT: setpoint_drift | output_collapse | open_diode; INJECTOR_FAULT: clog | leak;
    # COOLING_FAULT: radiator_blockage (default) | pump_degradation | coolant_loss
    sub_mode: str | None = None
    # Progression (Prompt 16): with a rate > 0 the effective severity grows
    # from 0 at onset at this rate [1/s] up to `severity` (0: step to severity).
    progression_rate_per_s: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"Severity must be in range [0.0, 1.0], got {self.severity}")

        if self.start_time_s is not None:
            self.onset_time_s = self.start_time_s

        if self.affected_cylinder is not None:
            self.affected_cylinders = [self.affected_cylinder]

    def severity_at(self, time_s: float) -> float:
        """Effective severity at time_s (0 before onset; ramped when progressing)."""
        if time_s < self.onset_time_s:
            return 0.0
        sev = max(0.0, min(1.0, self.severity))
        if self.progression_rate_per_s > 0.0:
            sev = min(sev, self.progression_rate_per_s * (time_s - self.onset_time_s))
        return sev


@dataclass
class SimulationMetadata:
    """Explicit simulation run metadata."""

    scenario_id: str
    seed: int
    simulator_version: str = "1.0.0"
    configuration_version: str = "1.0.0"
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_s: float = 0.0
    total_samples: int = 0
    fault_events: list[FaultScenarioConfig] = field(default_factory=list)


class ForwardPhysicsModel:
    """Deterministic forward-physics engine model for the Rotax 915 iS."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        fault_configs: list[FaultScenarioConfig] | None = None,
        seed: int = 42,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine_params = self._settings.engine
        self._phys = self._settings.simulator.physics
        self._fault_configs = fault_configs or []
        self._seed = seed
        self._injection = None
        if self._settings.simulator.injection.enabled:
            from src.l1_data.simulator.injection import InjectionSimulator
            self._injection = InjectionSimulator(self._settings)
        # thermal-lag states (CHT, oil) and the time they were last advanced
        self._lag_t: float | None = None
        self._cht_lag_k: float | None = None
        self._oil_lag_k: float | None = None
        self._coolant = None
        if self._settings.simulator.cooling.enabled and self._settings.engine.has_coolant_circuit:
            from src.l1_data.simulator.coolant_loop import CoolantLoop
            self._coolant = CoolantLoop(self._settings)

    def compute_ground_truth(
        self,
        time_s: float,
        sequence_number: int = 1,
        rpm: float = 4000.0,
        map_pa: float = 100000.0,
        throttle_pct: float = 50.0,
        altitude_m: float = 0.0,
        ambient_temp_k: float = 288.15,
        ambient_pressure_pa: float = 101325.0,
        fault_scenario: FaultScenarioConfig | None = None,
        timestamp: datetime | None = None,
        airspeed_m_s: float | None = None,
    ) -> SimulationGroundTruth:
        """Compute true physical engine state from operating parameters and fault injections.

        The coolant loop (Prompt 14) is stateful: consecutive calls with
        increasing time_s integrate it; the first call, or a call with time
        going backwards, starts it at the steady state."""
        ts = timestamp or (datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=time_s))

        # Clamp operating inputs
        c_rpm = max(500.0, min(6000.0, rpm))
        c_map = max(self._phys.min_map_pa, min(self._phys.max_map_pa, map_pa))
        c_throttle = max(0.0, min(100.0, throttle_pct))

        active_fault = FaultClass.NOMINAL
        severity = 0.0
        aff_cyls: list[int] = [1]

        # Determine active fault scenario from explicit parameter or fault_configs list
        f_active = fault_scenario
        if f_active is None and self._fault_configs:
            for f_cfg in self._fault_configs:
                f_start = f_cfg.onset_time_s
                f_end = f_start + f_cfg.duration_s
                if f_start <= time_s <= f_end:
                    f_active = f_cfg
                    break

        # Physical perturbations that act on the air path are applied BEFORE the
        # air -> fuel -> IMEP chain (SRD-FUN-152): a boost leak lowers MAP, so
        # the engine traps less air and (speed-density fuelling) burns less fuel.
        if (f_active and f_active.fault_class == FaultClass.INTAKE_BOOST_LEAK
                and f_active.onset_time_s <= time_s <= f_active.onset_time_s + f_active.duration_s):
            leak_sev = f_active.severity_at(time_s)
            c_map = max(40000.0, c_map - leak_sev * 35000.0)

        # Nominal physics derivations based on Rotax 915 iS reference parameters
        base_egt_k = 900.0 + (c_map / 100000.0) * 50.0 + (c_rpm / 6000.0) * 30.0
        # healthy engine state: per-cylinder installation offsets and benign
        # ageing (Prompt 17); zero for the reference engine
        age = self._phys.ageing_hours / 100.0
        egt_k = [base_egt_k + off + age * self._phys.ageing_egt_k_per_100h
                 for off in self._phys.egt_cylinder_offsets_k]

        cht_k = 363.15 + (c_map / 100000.0) * 15.0 + (c_rpm / 6000.0) * 10.0 + age * self._phys.ageing_cht_k_per_100h
        oil_temp_k = 353.15 + (c_rpm / 6000.0) * 20.0
        # Oil pressure at the reference oil temperature; the viscosity factor for
        # the actual (post-fault) oil temperature is applied after the fault block.
        oil_pressure_ref_pa = 200000.0 + (c_rpm / 6000.0) * 300000.0
        oil_pressure_drop_pa = 0.0
        vibration_rms_m_s2 = 3.0 + (c_rpm / 6000.0) * 4.0
        vibration_exc_g = 0.3 + (c_rpm / 6000.0) * 0.4
        # Air -> fuel -> IMEP -> brake power (simulator's own constants only).
        air_kg_s, lambda_cmd, fuel_flow_kg_s, imep_pa, fmep_pa = self._fuel_and_mep(
            c_rpm, c_map, ambient_temp_k
        )
        # ECU injection and fuel rail (Prompt 13): each cylinder burns the fuel
        # its injector actually delivers. Healthy injectors on a regulated rail
        # deliver exactly the ECU target, so every factor below is exactly 1.
        fault_on = bool(f_active and f_active.onset_time_s <= time_s <= f_active.onset_time_s + f_active.duration_s)
        injection = None
        if self._injection is not None:
            injection = self._injection.solve(
                c_rpm, c_map, ambient_pressure_pa, air_kg_s, lambda_cmd,
                fault_mode=f_active.fault_class if fault_on else None,
                sub_mode=f_active.sub_mode if fault_on else None,
                severity=f_active.severity_at(time_s) if fault_on else 0.0,
                cylinders=f_active.affected_cylinders if fault_on else None,
            )
            fuel_mean = sum(injection.fuel_factor) / len(injection.fuel_factor)
            if fuel_mean != 1.0:
                fuel_flow_kg_s *= fuel_mean
            if injection.imep_factor != 1.0:
                imep_pa *= injection.imep_factor
            t_charge = ambient_temp_k + self._phys.charge_temp_rise_k
            egt_k = [e if fct == 1.0 else t_charge + (e - t_charge) * fct
                     for e, fct in zip(egt_k, injection.egt_rise_factor)]
        disp_m3 = self._engine_params.displacement_cc * 1e-6
        brake_power_kw = (imep_pa - fmep_pa) * disp_m3 * c_rpm / 120.0 / 1000.0
        torque_nm = (brake_power_kw * 1000.0) / max(1.0, (2.0 * math.pi * c_rpm / 60.0))

        vibration_rms_base_m_s2 = vibration_rms_m_s2  # before any fault increment
        imbalance_hz: float | None = None
        imbalance_amp: tuple[float, float, float] | None = None

        # Apply active fault injection scenario effects on physical state
        if f_active and f_active.fault_class != FaultClass.NOMINAL:
            f_start = f_active.onset_time_s
            f_end = f_start + f_active.duration_s
            if f_start <= time_s <= f_end:
                active_fault = f_active.fault_class
                severity = f_active.severity_at(time_s)
                aff_cyls = f_active.affected_cylinders

                if active_fault == FaultClass.MISFIRE:
                    for c_id in aff_cyls:
                        if 1 <= c_id <= 4:
                            egt_k[c_id - 1] -= severity * 150.0
                    brake_power_kw *= max(0.2, 1.0 - severity * 0.25)
                    vibration_rms_m_s2 += severity * 8.0
                    vibration_exc_g += severity * 1.5

                elif active_fault == FaultClass.DETONATION_KNOCK:
                    # Knock raises head heat flux; the ECU knock control
                    # retards the knocking cylinder(s) (injection model), which
                    # moves heat to the exhaust through the energy balance.
                    # No accelerometer effect: knock energy (several kHz) lies
                    # above the 1024 Hz Nyquist limit of the 2048 Hz burst and
                    # is removed by the anti-alias filter (Prompt 13).
                    cht_k += severity * 40.0
                    if self._injection is None:
                        for c_id in aff_cyls:
                            if 1 <= c_id <= 4:
                                egt_k[c_id - 1] += severity * 50.0

                elif active_fault == FaultClass.EXHAUST_VALVE_LEAK:
                    for c_id in aff_cyls:
                        if 1 <= c_id <= 4:
                            egt_k[c_id - 1] -= severity * 180.0
                    brake_power_kw *= max(0.5, 1.0 - severity * 0.15)

                elif active_fault == FaultClass.INTAKE_BOOST_LEAK:
                    pass  # MAP already reduced before the air/fuel chain; power follows physically

                elif active_fault == FaultClass.OIL_DEGRADATION:
                    oil_pressure_drop_pa = severity * 220000.0
                    oil_temp_k += severity * 18.0

                elif active_fault == FaultClass.COOLING_FAULT:
                    if self._coolant is None:  # no coolant loop: legacy state perturbation
                        cht_k += severity * 35.0
                        oil_temp_k += severity * 25.0
                    # else: the coolant loop applies the physical fault below

                elif active_fault == FaultClass.BEARING_WEAR:
                    vibration_rms_m_s2 += severity * 22.0
                    vibration_exc_g += severity * 3.0
                    brake_power_kw *= max(0.7, 1.0 - severity * 0.1)

                elif active_fault == FaultClass.IMBALANCE:
                    # Rotating force m e omega^2 in the propeller disc; response
                    # velocity scales with omega; lateral mount softer (VERIFY).
                    ph = self._phys
                    f_prop = c_rpm / 60.0 / ph.propeller_gear_ratio
                    v_lat = severity * ph.imbalance_velocity_m_s_sev1 * (
                        c_rpm / ph.propeller_gear_ratio / ph.imbalance_ref_prop_rpm)
                    a_lat = v_lat * 2.0 * math.pi * f_prop
                    imbalance_hz = f_prop
                    imbalance_amp = (a_lat * ph.imbalance_axial_fraction, a_lat,
                                     a_lat / ph.imbalance_lateral_to_vertical)

                elif active_fault == FaultClass.SENSOR_FAULT:
                    # Physical ground truth unaltered
                    pass

        # Thermal lag (Prompt 14): head and oil follow their steady-state maps
        # with first-order lags; a first call or a time reversal starts settled.
        if self._lag_t is None or self._cht_lag_k is None or time_s <= self._lag_t:
            self._cht_lag_k, self._oil_lag_k = cht_k, oil_temp_k
        else:
            dt_lag = time_s - self._lag_t
            self._cht_lag_k += (cht_k - self._cht_lag_k) * (1.0 - math.exp(-dt_lag / self._phys.cht_time_constant_s))
            self._oil_lag_k += (oil_temp_k - self._oil_lag_k) * (
                1.0 - math.exp(-dt_lag / self._phys.oil_time_constant_s))
        self._lag_t = time_s
        cht_k, oil_temp_k = self._cht_lag_k, self._oil_lag_k

        # Liquid coolant loop (Prompt 14): ambient, airspeed, COOLING_FAULT and
        # thermal lag act on CHT and oil temperature through the loop.
        coolant = None
        if self._coolant is not None:
            cool_fault = fault_on and f_active.fault_class == FaultClass.COOLING_FAULT
            coolant = self._coolant.step(
                time_s, c_rpm, fuel_flow_kg_s * self._phys.lhv_j_kg, ambient_temp_k, ambient_pressure_pa,
                airspeed_m_s=airspeed_m_s,
                sub_mode=f_active.sub_mode if cool_fault else None,
                severity=f_active.severity_at(time_s) if cool_fault else 0.0,
                onset_s=f_active.onset_time_s if cool_fault else 0.0,
            )
            cc = self._settings.simulator.cooling
            cht_k += coolant.cht_loop_delta_k
            oil_temp_k += (cc.oil_coolant_coupling * (coolant.coolant_temp_k - coolant.coolant_ref_ss_k)
                           + cc.oil_cooler_ambient_gain * (ambient_temp_k - cc.reference_ambient_k))

        # Oil pressure follows oil viscosity at the actual oil temperature.
        oil_pressure_pa = max(50000.0, oil_pressure_ref_pa * self.oil_viscosity_ratio(oil_temp_k)
                              - oil_pressure_drop_pa)

        return SimulationGroundTruth(
            timestamp=ts,
            time_s=time_s,
            sequence_number=sequence_number,
            rpm=c_rpm,
            map_pressure_pa=c_map,
            throttle_pct=c_throttle,
            altitude_m=altitude_m,
            ambient_temp_k=ambient_temp_k,
            ambient_pressure_pa=ambient_pressure_pa,
            egt_k=egt_k,
            cht_k=cht_k,
            oil_temp_k=oil_temp_k,
            oil_pressure_pa=oil_pressure_pa,
            vibration_rms_m_s2=vibration_rms_m_s2,
            vibration_rms_base_m_s2=vibration_rms_base_m_s2,
            vibration_exc_g=vibration_exc_g,
            fuel_flow_kg_s=fuel_flow_kg_s,
            brake_power_kw=brake_power_kw,
            torque_nm=torque_nm,
            air_mass_flow_kg_s=air_kg_s,
            lambda_cmd=lambda_cmd,
            imep_pa=imep_pa,
            fmep_pa=fmep_pa,
            injection=injection,
            coolant=coolant,
            imbalance_hz=imbalance_hz,
            imbalance_accel_peak_m_s2=imbalance_amp,
            airspeed_m_s=airspeed_m_s,
            active_fault=active_fault,
            injected_fault_class=active_fault,
            injected_fault_severity=severity,
            affected_cylinders=aff_cyls,
            provenance=Provenance.SIMULATED,
        )

    def oil_viscosity_ratio(self, oil_temp_k: float) -> float:
        """(mu(T) / mu(T_ref))^n with the simulator's own Vogel constants."""
        ph = self._phys
        t = max(oil_temp_k, ph.oil_vogel_c_k + 1.0)
        ln_ratio = ph.oil_vogel_b_k / (t - ph.oil_vogel_c_k) - ph.oil_vogel_b_k / (
            ph.oil_pressure_ref_temp_k - ph.oil_vogel_c_k)
        return math.exp(ph.oil_pressure_viscosity_exponent * ln_ratio)

    def volumetric_efficiency(self, rpm: float, map_pa: float) -> float:
        """Simulator's own eta_v(rpm, MAP); independent of the twin's D-04 model."""
        ph = self._phys
        x = (rpm - ph.eta_v_peak_rpm) / ph.eta_v_peak_rpm
        eta = ph.eta_v_peak - ph.eta_v_curvature * x * x
        eta *= 1.0 + ph.eta_v_map_sensitivity * (map_pa / 101325.0 - 1.0)
        return min(max(eta, ph.eta_v_min), ph.eta_v_max)

    def air_mass_flow(self, rpm: float, map_pa: float, ambient_temp_k: float) -> float:
        """m_dot_air = eta_v * MAP / (R T_charge) * V_d * rpm / 120   [kg/s]"""
        ph = self._phys
        t_charge = ambient_temp_k + ph.charge_temp_rise_k
        rho = map_pa / (ph.r_air_j_kg_k * t_charge)
        disp_m3 = self._engine_params.displacement_cc * 1e-6
        return self.volumetric_efficiency(rpm, map_pa) * rho * disp_m3 * rpm / 120.0

    def lambda_command(self, load: float) -> float:
        """Mixture schedule: lambda vs air-flow load (VERIFY against Rotax manual)."""
        ph = self._phys
        return float(np.interp(load, ph.lambda_schedule_load, ph.lambda_schedule_lambda))

    def fmep_pa(self, rpm: float) -> float:
        """Barnes-Moss, Heywood (1988) ch. 13 (VERIFY eq. number), unscaled [Pa]."""
        ph = self._phys
        n = rpm / 1000.0
        return 1e5 * (ph.fmep_barnes_moss_a_bar + ph.fmep_barnes_moss_b_bar * n
                      + ph.fmep_barnes_moss_c_bar * n * n)

    def _fuel_and_mep(
        self, rpm: float, map_pa: float, ambient_temp_k: float
    ) -> tuple[float, float, float, float, float]:
        """(air kg/s, lambda_cmd, fuel kg/s, IMEP Pa, FMEP Pa) for this operating point.

            m_dot_fuel = m_dot_air / (AFR_st * lambda_cmd)
            IMEP = eta_i * eta_c * m_dot_fuel * LHV * 120 / (V_d * rpm),
            eta_c = eta_c,max * min(1, lambda)  (oxygen-limited when rich)
        """
        ph = self._phys
        air = self.air_mass_flow(rpm, map_pa, ambient_temp_k)
        air_rated = self.air_mass_flow(ph.rated_rpm, ph.max_map_pa, ambient_temp_k)
        load = air / air_rated if air_rated > 0.0 else 0.0
        lam = self.lambda_command(load)
        fuel = air / (ph.stoichiometric_afr * lam)
        eta_c = ph.combustion_efficiency_max * min(1.0, lam)
        disp_m3 = self._engine_params.displacement_cc * 1e-6
        imep = ph.indicated_efficiency * eta_c * fuel * ph.lhv_j_kg * 120.0 / (disp_m3 * rpm)
        return air, lam, fuel, imep, self.fmep_pa(rpm)

    def step(
        self,
        time_s: float = 0.0,
        sequence_number: int = 1,
        rpm: float = 4000.0,
        map_pa: float = 100000.0,
        throttle_pct: float = 50.0,
        altitude_m: float = 0.0,
    ) -> SimulationGroundTruth:
        """Convenience single-step wrapper for compute_ground_truth."""
        return self.compute_ground_truth(
            time_s=time_s,
            sequence_number=sequence_number,
            rpm=rpm,
            map_pa=map_pa,
            throttle_pct=throttle_pct,
            altitude_m=altitude_m,
        )


class SensorForwardModel:
    """Converts physical ground truth into canonical acquisition RawSignalRecord."""

    def __init__(self, seed: int = 42, settings: AppSettings | None = None) -> None:
        self._seed = seed
        self._settings = settings or get_settings()
        self._rng = np.random.default_rng(seed)
        # Separate stream for high-rate bursts so the per-record noise sequence
        # (and therefore every scalar raw field) is unchanged for a given seed.
        self._burst_rng = np.random.default_rng([seed, 0xB0])
        self._elec_rng = np.random.default_rng([seed, 0xE2])
        self._inj_rng = np.random.default_rng([seed, 0xF1])
        self._cool_rng = np.random.default_rng([seed, 0xC1])

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset RNG seed for 100% deterministic replay."""
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._burst_rng = np.random.default_rng([self._seed, 0xB0])
        self._elec_rng = np.random.default_rng([self._seed, 0xE2])
        self._inj_rng = np.random.default_rng([self._seed, 0xF1])
        self._cool_rng = np.random.default_rng([self._seed, 0xC1])

    def convert_to_raw_record(
        self,
        ground_truth: SimulationGroundTruth,
        fault_scenario: FaultScenarioConfig | None = None,
        enable_noise: bool = True,
    ) -> RawSignalRecord:
        """Convert ground truth physical state into canonical RawSignalRecord."""
        # 1. Type-K Thermocouple conversion (EGT & CHT in microvolts)
        # NIST ITS-90: the reading is the difference of the two junction EMFs,
        # E(T_hot) - E(T_cold), not a slope times the temperature difference.
        egt_cold_c = ground_truth.ambient_temp_k - 273.15
        cold_emf_uv = type_k_emf_uv(egt_cold_c)

        egt_uv = [
            type_k_emf_uv(k - 273.15) - cold_emf_uv
            for k in ground_truth.egt_k
        ]

        cht_cold_c = egt_cold_c
        cht_uv = type_k_emf_uv(ground_truth.cht_k - 273.15) - type_k_emf_uv(cht_cold_c)

        # 2. PT100 RTD Oil Temperature conversion (Ohms), IEC 60751 CVD
        oil_temp_c = ground_truth.oil_temp_k - 273.15
        oil_rtd_ohms = pt100_resistance_ohms(oil_temp_c)

        # 3. ADC Pressure conversion (Counts)
        oil_p_counts = int((ground_truth.oil_pressure_pa / 800000.0) * 4095.0)
        map_counts = int((ground_truth.map_pressure_pa / 200000.0) * 4095.0)
        adc_vref_counts = 4095

        # 4. Crankshaft period (microseconds)
        crank_period_us = (60.0 * 1e6) / max(100.0, ground_truth.rpm)

        # 5. Fuel flow pulse frequency (Hz)
        fuel_pulse_hz = ground_truth.fuel_flow_kg_s / 0.0001

        # 6. Accelerometer ADC counts (X, Y, Z)
        counts_per_m_s2 = self._settings.sensor_calibration.accel_counts_per_m_s2
        vib_count = int(ground_truth.vibration_rms_m_s2 * counts_per_m_s2)
        accel_counts_xyz = (vib_count, vib_count, vib_count)
        imb = ground_truth.imbalance_accel_peak_m_s2
        if ground_truth.imbalance_hz and imb is not None:
            # per-axis RMS with the 1X sinusoid added (peak a -> RMS a / sqrt 2)
            accel_counts_xyz = tuple(int(math.sqrt(ground_truth.vibration_rms_m_s2 ** 2 + a * a / 2.0)
                                         * counts_per_m_s2) for a in imb)

        # 6a. Electrical (Prompt 12): divider / Hall sensors -> ADC counts,
        #     ratiometric to the 12-bit reference (4095). None if not simulated.
        elec_fields: dict[str, Any] = {}
        el = ground_truth.electrical
        if el is not None:
            ec = self._settings.simulator.electrical
            nz = (lambda sd: float(self._elec_rng.normal(0.0, sd))) if enable_noise else (lambda sd: 0.0)

            def _counts(x: float) -> int:
                return int(min(max(round(x * 4095.0), 0), 4095))

            elec_fields = {
                "bus_v_counts": _counts((el.bus_voltage_v + nz(ec.bus_v_noise_v)) / ec.bus_v_full_scale_v),
                "alt_i_counts": _counts((el.alternator_current_a + nz(ec.current_noise_a)) / ec.alt_i_full_scale_a),
                "batt_i_counts": _counts((el.battery_current_a + nz(ec.current_noise_a)) / ec.batt_i_full_scale_a + 0.5),
                "bus_v_burst_counts": tuple(_counts(v / ec.bus_v_full_scale_v) for v in el.burst_v),
                "bus_v_burst_fs_hz": el.burst_fs_hz,
            }

        # 6a'. Injection / ignition timer captures and fuel-rail pressure
        #      (Prompt 13): intervals in us from the ECU angles at this rpm
        #      (deg = us x rpm x 6e-6), rail gauge pressure -> ADC counts.
        inj_fields: dict[str, Any] = {}
        inj = ground_truth.injection
        if inj is not None:
            ic = self._settings.simulator.injection
            jit = (lambda: float(self._inj_rng.normal(0.0, ic.timing_noise_us))) if enable_noise else (lambda: 0.0)
            us_per_deg = 1.0 / (ground_truth.rpm * 6e-6)
            p_noise = float(self._inj_rng.normal(0.0, ic.fuel_press_noise_pa)) if enable_noise else 0.0
            inj_fields = {
                "inj_pw_us": tuple(inj.pulse_width_us + jit() for _ in range(4)),
                "inj_soi_delay_us": tuple(inj.soi_deg * us_per_deg + jit() for _ in range(4)),
                "ign_delay_us": tuple(a * us_per_deg + jit() for a in inj.advance_deg),
                "fuel_press_counts": int(min(max(round((inj.rail_gauge_pa + p_noise)
                                                       / ic.fuel_press_full_scale_pa * 4095.0), 0), 4095)),
            }

        # 6a''. Coolant NTC (Prompt 14): Steinhart-Hart forward model -> ohms.
        cool_fields: dict[str, Any] = {}
        if ground_truth.coolant is not None:
            cc = self._settings.simulator.cooling
            r_ohm = ntc_resistance_ohm(ground_truth.coolant.coolant_temp_k - 273.15,
                                       cc.ntc_sh_a, cc.ntc_sh_b, cc.ntc_sh_c)
            if enable_noise:
                r_ohm *= 1.0 + float(self._cool_rng.normal(0.0, cc.ntc_noise_rel))
            cool_fields = {"coolant_ntc_ohms": r_ohm}

        # 6b. High-rate bursts: 2048 samples at 2048 Hz per axis, and the last
        #     32 per-revolution crank periods. Firing-order 2X physics, 0.5X
        #     under misfire, AM bearing carrier under bearing wear, plus noise.
        burst_fault = ground_truth.active_fault
        burst_sev = ground_truth.injected_fault_severity
        bx, by, bz = generate_accel_burst(
            rpm=ground_truth.rpm,
            axis_rms_m_s2=ground_truth.vibration_rms_m_s2,
            fault_mode=burst_fault,
            severity=burst_sev,
            rng=self._burst_rng,
            fs_hz=BURST_FS_HZ,
            n_samples=BURST_SAMPLES,
            enable_noise=enable_noise,
            base_rms_m_s2=ground_truth.vibration_rms_base_m_s2,
        )
        if ground_truth.imbalance_hz and imb is not None:
            # rotating force: lateral and vertical in quadrature, axial in phase with lateral
            t_b = np.arange(BURST_SAMPLES) / BURST_FS_HZ
            w_t = 2.0 * math.pi * ground_truth.imbalance_hz * t_b
            bx = bx + imb[0] * np.sin(w_t)
            by = by + imb[1] * np.sin(w_t)
            bz = bz + imb[2] * np.cos(w_t)
        crank_burst_us = generate_crank_burst(
            rpm=ground_truth.rpm,
            fault_mode=burst_fault,
            severity=burst_sev,
            affected_cylinders=ground_truth.affected_cylinders,
            rng=self._burst_rng,
            n_revs=CRANK_BURST_REVS,
            enable_noise=enable_noise,
        )

        # Ambient
        amb_temp_c = ground_truth.ambient_temp_k - 273.15
        amb_press_pa = ground_truth.ambient_pressure_pa

        # 7. Add deterministic noise if enabled
        if enable_noise:
            noise_egt = self._rng.normal(0.0, 50.0, size=4)
            egt_uv = [u + float(n) for u, n in zip(egt_uv, noise_egt)]
            cht_uv += float(self._rng.normal(0.0, 30.0))
            oil_rtd_ohms += float(self._rng.normal(0.0, 0.05))
            oil_p_counts = int(oil_p_counts + self._rng.normal(0.0, 5.0))
            map_counts = int(map_counts + self._rng.normal(0.0, 5.0))
            crank_period_us += float(self._rng.normal(0.0, 2.0))
            fuel_pulse_hz += float(self._rng.normal(0.0, 0.1))

        # 8. Sensor Fault Injection handling
        signal_quality = SignalQuality(valid=True, score=1.0)
        if fault_scenario and fault_scenario.fault_class == FaultClass.SENSOR_FAULT:
            f_start = fault_scenario.onset_time_s
            f_end = f_start + fault_scenario.duration_s
            if f_start <= ground_truth.time_s <= f_end:
                target_chan = fault_scenario.affected_channel or "egt_hot_junction_mv_c1"
                if target_chan in ("egt_hot_junction_mv_c1", "egt_cyl1_hot_uv"):
                    egt_uv[0] = -999.0  # Dropout / invalid reading
                    # Negative uV is a physical reading, so the dropout must be
                    # flagged explicitly on the channel, not inferred from sign.
                    signal_quality = SignalQuality(
                        valid=False,
                        score=0.0,
                        invalid_channels=["egt_cyl1_hot_uv"],
                        invalid_reasons={"egt_cyl1_hot_uv": "SENSOR_DROPOUT"},
                    )
                elif target_chan == "map_counts":
                    map_counts = 0
                    signal_quality = SignalQuality(
                        valid=False,
                        score=0.0,
                        invalid_channels=["map_counts"],
                        invalid_reasons={"map_counts": "SENSOR_DROPOUT"},
                    )
                elif target_chan == "oil_p_counts":
                    oil_p_counts = 4095  # Saturation
                    signal_quality = SignalQuality(
                        valid=False,
                        score=0.0,
                        invalid_channels=["oil_p_counts"],
                        invalid_reasons={"oil_p_counts": "SENSOR_SATURATION"},
                    )

        # Clamp ADC counts to [0, 4095]
        oil_p_counts = max(0, min(4095, oil_p_counts))
        map_counts = max(0, min(4095, map_counts))
        adc_vref_counts = max(0, min(4095, adc_vref_counts))

        rec = RawSignalRecord(
            timestamp=ground_truth.timestamp,
            sequence_number=ground_truth.sequence_number,
            source_type=Provenance.SIMULATED,
            egt_cyl1_hot_uv=egt_uv[0],
            egt_cyl2_hot_uv=egt_uv[1],
            egt_cyl3_hot_uv=egt_uv[2],
            egt_cyl4_hot_uv=egt_uv[3],
            egt_cold_c=egt_cold_c,
            cht_hot_uv=cht_uv,
            cht_cold_c=cht_cold_c,
            oil_rtd_ohms=oil_rtd_ohms,
            oil_p_counts=oil_p_counts,
            map_counts=map_counts,
            adc_vref_counts=adc_vref_counts,
            crank_period_us=max(100.0, crank_period_us),
            fuel_pulse_hz=max(0.0, fuel_pulse_hz),
            accel_counts_xyz=accel_counts_xyz,
            accel_burst_counts_x=accel_to_counts(bx, counts_per_m_s2),
            accel_burst_counts_y=accel_to_counts(by, counts_per_m_s2),
            accel_burst_counts_z=accel_to_counts(bz, counts_per_m_s2),
            accel_burst_fs_hz=BURST_FS_HZ,
            crank_period_burst_us=crank_burst_us,
            **elec_fields,
            **inj_fields,
            **cool_fields,
            ambient_temp_c=amb_temp_c,
            ambient_press_pa=amb_press_pa,
            signal_quality=signal_quality,
        )

        return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})

    def generate_record(
        self,
        ground_truth: SimulationGroundTruth,
        fault_scenario: FaultScenarioConfig | None = None,
        enable_noise: bool = True,
    ) -> RawSignalRecord:
        """Alias for convert_to_raw_record."""
        return self.convert_to_raw_record(
            ground_truth=ground_truth,
            fault_scenario=fault_scenario,
            enable_noise=enable_noise,
        )


class ScenarioRunner:
    """Deterministic scenario runner for generating raw telemetry sequences and ground truth logs."""

    def __init__(self, settings: AppSettings | None = None, seed: int = 42) -> None:
        self._settings = settings or get_settings()
        self._seed = seed
        self._physics_model = ForwardPhysicsModel(settings=self._settings, seed=seed)

    def run_scenario(
        self,
        duration_s: float = 60.0,
        dt_s: float = 0.1,
        timestep_s: float | None = None,
        rpm_profile: list[float] | None = None,
        throttle_profile: list[float] | None = None,
        operating_profile: list[dict[str, float]] | None = None,
        fault_scenarios: list[FaultScenarioConfig] | None = None,
        seed: int | None = None,
        enable_noise: bool = True,
        isa_deviation_k: float = 0.0,
    ) -> tuple[list[RawSignalRecord], list[SimulationGroundTruth], SimulationMetadata]:
        """Run a deterministic scenario simulation over duration_s.

        Ambient follows the ISA at the profile altitude plus isa_deviation_k
        (a profile segment may set its own "isa_deviation_k", or explicit
        "ambient_temp_k" / "ambient_pressure_pa"); "airspeed_m_s" feeds the
        radiator. Altitude 0 and deviation 0 give 288.15 K / 101325 Pa.

        Returns (raw_telemetry_records, ground_truth_logs, metadata).
        """
        active_seed = seed if seed is not None else self._seed
        sensor_model = SensorForwardModel(seed=active_seed, settings=self._settings)
        electrical = None
        if self._settings.simulator.electrical.enabled:
            from src.l1_data.simulator.electrical import ElectricalSimulator
            electrical = ElectricalSimulator(self._settings, seed=active_seed)
        raw_records: list[RawSignalRecord] = []
        ground_truth_logs: list[SimulationGroundTruth] = []

        actual_dt = dt_s if timestep_s is None else timestep_s
        actual_dt = max(0.01, min(10.0, float(actual_dt)))
        total_steps = min(int(duration_s / actual_dt), 36000)
        base_time = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        for step in range(total_steps):
            t_s = step * actual_dt

            # Default profile: 4000 RPM, 100 kPa MAP, 50% throttle
            rpm = 4000.0
            map_pa = 100000.0
            throttle_pct = 50.0
            altitude_m = 0.0
            isa_dev = isa_deviation_k
            airspeed = None
            amb_t_override = amb_p_override = None

            if rpm_profile and step < len(rpm_profile):
                rpm = rpm_profile[step]

            if throttle_profile and step < len(throttle_profile):
                throttle_pct = throttle_profile[step]

            # Override from operating_profile if specified
            if operating_profile:
                for prof in operating_profile:
                    p_start = prof.get("start_time_s", 0.0)
                    p_end = prof.get("end_time_s", duration_s)
                    if p_start <= t_s <= p_end:
                        rpm = prof.get("rpm", rpm)
                        map_pa = prof.get("map_pa", map_pa)
                        throttle_pct = prof.get("throttle_pct", throttle_pct)
                        altitude_m = prof.get("altitude_m", altitude_m)
                        isa_dev = prof.get("isa_deviation_k", isa_dev)
                        airspeed = prof.get("airspeed_m_s", airspeed)
                        amb_t_override = prof.get("ambient_temp_k", amb_t_override)
                        amb_p_override = prof.get("ambient_pressure_pa", amb_p_override)

            # Find active fault scenario for current timestep
            active_fault: FaultScenarioConfig | None = None
            if fault_scenarios:
                for f_scen in fault_scenarios:
                    if f_scen.onset_time_s <= t_s <= (f_scen.onset_time_s + f_scen.duration_s):
                        active_fault = f_scen
                        break

            ts = base_time + timedelta(seconds=t_s)
            ambient_t = amb_t_override if amb_t_override is not None else isa_temperature(altitude_m) + isa_dev
            ambient_p = amb_p_override if amb_p_override is not None else isa_pressure(altitude_m)
            gt = self._physics_model.compute_ground_truth(
                time_s=t_s,
                sequence_number=step + 1,
                rpm=rpm,
                map_pa=map_pa,
                throttle_pct=throttle_pct,
                altitude_m=altitude_m,
                ambient_temp_k=ambient_t,
                ambient_pressure_pa=ambient_p,
                fault_scenario=active_fault,
                timestamp=ts,
                airspeed_m_s=airspeed,
            )

            if electrical is not None:
                gt.electrical = electrical.step(
                    t_s, gt.rpm,
                    fault_mode=active_fault.fault_class if active_fault else None,
                    sub_mode=active_fault.sub_mode if active_fault else None,
                    severity=active_fault.severity_at(t_s) if active_fault else 0.0,
                    onset_s=active_fault.onset_time_s if active_fault else 0.0,
                    enable_noise=enable_noise,
                )

            raw_rec = sensor_model.convert_to_raw_record(
                ground_truth=gt,
                fault_scenario=active_fault,
                enable_noise=enable_noise,
            )

            ground_truth_logs.append(gt)
            raw_records.append(raw_rec)

        metadata = SimulationMetadata(
            scenario_id=f"scen_{active_seed}_{int(duration_s)}s",
            seed=active_seed,
            duration_s=duration_s,
            total_samples=len(raw_records),
            fault_events=fault_scenarios or [],
        )

        return raw_records, ground_truth_logs, metadata
