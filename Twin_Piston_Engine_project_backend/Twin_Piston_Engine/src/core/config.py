"""
Config — Externalized configuration system using Pydantic Settings.

Loads configuration from config/default.yaml with environment variable
overrides. No physical constant or threshold is hardcoded in business
logic — everything comes from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.provenance import CommunicationState, DeploymentRole, OverflowPolicy


# =============================================================================
# Configuration Sub-Models
# =============================================================================


class EngineConfig(BaseModel):
    """Rotax 915 iS engine geometric and operational parameters."""

    name: str = "Rotax 915 iS"
    bore_mm: float = 84.0
    stroke_mm: float = 61.0
    displacement_cc: float = 1352.0
    num_cylinders: int = 4
    compression_ratio: float = 10.5
    # Peak (take-off) power 104 kW / 141 hp; the page gives no rpm for it, so
    # rated_rpm stays the project value (VERIFY against the Operators Manual).
    # Source: BRP-Rotax 915 iS A / iSc A product page, https://www.flyrotax.com/products/915-is-a-isc-a (read 2026-09-28). Was 105.0 (unsourced).
    rated_power_kw: float = 104.0
    # Maximum continuous power 99.0 kW / 135 hp at 5500 1/min; TBO 1,200 h.
    # Source: BRP-Rotax 915 iS A / iSc A product page, https://www.flyrotax.com/products/915-is-a-isc-a (read 2026-09-28).
    max_continuous_power_kw: float = 99.0
    max_continuous_rpm: int = 5500
    tbo_hours: float = 1200.0
    rated_rpm: int = 5800
    max_rpm: int = 5800
    firing_order: list[int] = Field(default_factory=lambda: [1, 3, 2, 4])
    connecting_rod_mm: float = 105.0
    intake_valve_close_btdc_deg: float = 50.0
    exhaust_valve_open_bbdc_deg: float = 55.0
    piston_mass_kg: float = 0.35
    conrod_mass_kg: float = 0.30
    fuel_type: str = "avgas_100ll"
    lhv_mj_per_kg: float = 43.5
    stoichiometric_afr: float = 14.7
    # Healthy-engine SFC baseline for sfc_degradation_pct (M-07). Default is the
    # Rotax 915 iS published figure at continuous power.
    sfc_baseline_kg_kwh: float = 0.312
    # Propeller gearbox reduction ratio (crank rpm / propeller rpm), 2.54:1.
    # VERIFIED: "propeller speed reduction gearbox i = 2,54", BRP-Rotax 915 iS A / iSc A product page, https://www.flyrotax.com/products/915-is-a-isc-a (read 2026-09-28).
    # None would make propeller_speed valid=False.
    gearbox_ratio: float | None = 2.54
    # Engine hours at installation of the monitoring system, from the engine
    # logbook. None = unknown: engine_hours is then valid=False (the time
    # accumulated since monitoring started is still tracked).
    engine_hours_at_install: float | None = None
    # Largest gap between consecutive records that is counted as running time.
    # Calibration basis: project choice, 10 x the 1 Hz record interval, so a
    # telemetry outage is not counted as engine running time.
    engine_hours_max_gap_s: float = 10.0
    # Internal generator (Prompt 12). The pole count is NOT known: None means
    # L2 cannot compute the expected ripple frequency and never reports a
    # dominant ripple frequency as valid (aliasing cannot be excluded).
    # VERIFY against Rotax 915 iS Operators Manual.
    generator_poles: int | None = None
    # Cooling layout (Prompt 14). True = liquid coolant circuit (cooled heads,
    # radiator, thermostat). Basis: the existing CoolingConfig (50/50 glycol,
    # thermostat 85/95 C, radiator effectiveness), the README and
    # simulator/cooling.py describe one; none of them cites a source.
    # VERIFY against Rotax 915 iS Operators Manual. False -> every coolant
    # parameter is reported unavailable (SRD-FUN-045 pattern), never inferred.
    has_coolant_circuit: bool = True
    generator_phases: int = 3            # VERIFY against Rotax 915 iS Operators Manual
    generator_drive_ratio: float = 1.0   # generator rpm / crank rpm; VERIFY against Rotax 915 iS Operators Manual
    # Plausible mechanical-efficiency window; outside it the friction model is
    # suspect and eta_mech is emitted with quality 0.3.
    eta_mech_plausible_min: float = 0.65
    eta_mech_plausible_max: float = 0.92
    # Twin FMEP: Barnes-Moss (Heywood 1988, ch. 13; VERIFY eq. number),
    # fmep[bar] = a + b (N/1000) + c (N/1000)^2, scaled by
    # (oil viscosity / viscosity at lubrication.nominal_temp_c)^exponent.
    fmep_barnes_moss_a_bar: float = 0.97
    fmep_barnes_moss_b_bar: float = 0.15
    fmep_barnes_moss_c_bar: float = 0.05
    fmep_viscosity_exponent: float = 0.25


class TurbochargerConfig(BaseModel):
    """Turbocharger operational parameters."""

    max_boost_bar: float = 1.45
    wastegate_target_bar: float = 1.35
    compressor_efficiency: float = 0.72
    turbine_efficiency: float = 0.68
    compressor_pr_max: float = 2.5
    turbine_pr_max: float = 2.0
    shaft_inertia_kg_m2: float = 0.00005
    wastegate_kp: float = 2.0
    wastegate_ki: float = 0.5


class LubricationConfig(BaseModel):
    """Lubrication system parameters."""

    oil_type: str = "AeroShell Sport Plus 4"
    nominal_pressure_bar: float = 4.0
    min_pressure_bar: float = 1.5
    max_pressure_bar: float = 7.0
    warn_low_pressure_bar: float = 2.0
    crit_low_pressure_bar: float = 1.5
    warn_high_pressure_bar: float = 6.0
    crit_high_pressure_bar: float = 7.0
    nominal_temp_c: float = 90.0
    max_temp_c: float = 130.0
    min_temp_c: float = 40.0
    warn_high_temp_c: float = 120.0
    crit_high_temp_c: float = 130.0
    warn_low_temp_c: float = 50.0
    crit_low_temp_c: float = 40.0
    vogel_a: float = 0.0002
    vogel_b: float = 1200.0
    vogel_c: float = 140.0
    # Lubrication Health Index (M-10) component weights (log-space exponents).
    # Weights are re-normalised over the VALID components. Viscosity is never
    # valid: there is no measured viscosity channel (OI-8).
    lhi_weight_pressure: float = 1.0
    lhi_weight_viscosity: float = 1.0
    lhi_weight_over_temp: float = 1.0
    lhi_over_temp_limit_c: float = 130.0


class CoolingConfig(BaseModel):
    """Cooling system parameters."""

    coolant_type: str = "50/50 ethylene glycol"   # VERIFY against Rotax 915 iS Operators Manual
    thermostat_open_c: float = 85.0               # VERIFY against Rotax 915 iS Operators Manual
    thermostat_full_open_c: float = 95.0          # VERIFY against Rotax 915 iS Operators Manual
    nominal_cht_c: float = 100.0
    # CHT alarm limit; simulator/rotax_915is_params.py cites "OM 4.3" for 135 C,
    # not checked here. VERIFY against Rotax 915 iS Operators Manual.
    max_cht_c: float = 135.0
    # Coolant alarm limit: placeholder. VERIFY against Rotax 915 iS Operators Manual.
    max_coolant_c: float = 120.0
    coolant_flow_rate_kg_s: float = 1.0
    coolant_specific_heat_j_kg_k: float = 3500.0
    radiator_effectiveness: float = 0.65


class TelemetryConfig(BaseModel):
    """Telemetry acquisition settings."""

    sample_rate_hz: float = 1.0
    num_channels: int = 38


class SensorNoiseConfig(BaseModel):
    """Simulator sensor noise standard deviations."""

    rpm_std: float = 5.0
    temperature_std_k: float = 1.5
    pressure_std_pa: float = 500.0
    vibration_std_m_s2: float = 0.2
    fuel_flow_std_kg_s: float = 0.0001
    voltage_std_v: float = 0.05
    lambda_std: float = 0.005


class SimulatorPhysicsConfig(BaseModel):
    """Forward-simulator ground-truth constants.

    Deliberately separate from the twin's constants (engine.*): the simulator
    is the reference the twin is tested against, so it must not share the
    twin's model choices. Values marked VERIFY are engineering estimates to be
    checked against the Rotax 915 iS operator/maintenance manual and deck.
    """

    r_air_j_kg_k: float = 287.05
    charge_temp_rise_k: float = 28.0          # post-intercooler charge over ambient
    min_map_pa: float = 30000.0
    max_map_pa: float = 160000.0              # simulator MAP clamp = "maximum MAP"
    # eta_v_sim(rpm, MAP) = peak - curvature ((rpm - peak_rpm)/peak_rpm)^2,
    # times (1 + map_sensitivity (MAP/101325 - 1)), clipped to [min, max].
    eta_v_peak: float = 0.92                  # VERIFY (4-valve, turbocharged)
    eta_v_peak_rpm: float = 4500.0
    eta_v_curvature: float = 0.10
    eta_v_map_sensitivity: float = 0.02
    eta_v_min: float = 0.60
    eta_v_max: float = 0.98
    stoichiometric_afr: float = 14.7
    # lambda command vs load (load = air flow / air flow at rated rpm and
    # max MAP). VERIFY against the Rotax 915 iS manual (mixture schedule).
    lambda_schedule_load: list[float] = Field(default_factory=lambda: [0.0, 0.75, 1.0])
    lambda_schedule_lambda: list[float] = Field(default_factory=lambda: [1.00, 1.00, 0.87])
    rated_rpm: float = 5800.0
    indicated_efficiency: float = 0.40        # gross indicated, VERIFY
    combustion_efficiency_max: float = 0.98   # times min(1, lambda): O2-limited when rich
    lhv_j_kg: float = 43.5e6                  # Avgas 100LL
    # Oil pressure viscosity dependence (simulator's own oil model; not L2's):
    # p = p_ref(rpm) * (mu(T)/mu(T_ref))^exponent, mu = exp(B / (T - C)).
    # VERIFY against the oil grade / Rotax oil-pressure data.
    oil_vogel_b_k: float = 1100.0
    oil_vogel_c_k: float = 145.0
    oil_pressure_ref_temp_k: float = 363.15
    oil_pressure_viscosity_exponent: float = 0.3
    # Friction: Barnes-Moss (Heywood 1988, ch. 13; VERIFY eq. number), unscaled.
    fmep_barnes_moss_a_bar: float = 0.97
    fmep_barnes_moss_b_bar: float = 0.15
    fmep_barnes_moss_c_bar: float = 0.05
    # Thermal lag of the head and the oil (Prompt 14): the CHT and oil maps
    # are steady-state values; the simulated temperatures follow them with
    # first-order lags. VERIFY (no Rotax thermal-mass data).
    cht_time_constant_s: float = 90.0
    oil_time_constant_s: float = 300.0
    # IMBALANCE (Prompt 15): rotating propeller imbalance at the propeller
    # shaft frequency (crank / gear ratio). Force ~ omega^2, so the response
    # velocity ~ omega. Peak lateral velocity at severity 1 and the reference
    # propeller speed: 1.0 in/s (25.4 mm/s), basis: general-aviation propeller
    # dynamic-balance guidance treats ~0.2 in/s as the acceptable limit and
    # ~1 in/s as severe. Lateral mounts are assumed softer than vertical.
    # Accelerometer axes: x axial (crankshaft), y lateral, z vertical.
    # All VERIFY against the installation.
    propeller_gear_ratio: float = 2.54
    # Healthy engine-to-engine installation differences and benign ageing
    # (Prompt 17; state of the engine, not a sensor offset). EGT per cylinder
    # [K] and a slow EGT/CHT map rise with engine hours (deposits, wear-in).
    # Defaults 0: the reference engine. Rates VERIFY (no Rotax ageing data).
    egt_cylinder_offsets_k: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    ageing_hours: float = 0.0
    ageing_egt_k_per_100h: float = 0.0
    ageing_cht_k_per_100h: float = 0.0
    imbalance_velocity_m_s_sev1: float = 0.0254
    imbalance_ref_prop_rpm: float = 2300.0
    imbalance_lateral_to_vertical: float = 2.0
    imbalance_axial_fraction: float = 0.1


class SimulatorElectricalConfig(BaseModel):
    """Simulator electrical forward model (Prompt 12). Simulator-only constants,
    independent of the L2 electrical expectations (electrical.*).

    VERIFY-marked values are placeholders to check against the Rotax 915 iS
    Operators Manual / installed battery; others state their basis.
    """

    enabled: bool = True
    regulator_setpoint_v: float = 14.2           # VERIFY against Rotax 915 iS Operators Manual
    regulator_droop_ohm: float = 0.005           # finite regulator stiffness (V/A droop); VERIFY
    regulator_time_constant_s: float = 0.3       # regulator response; VERIFY
    cut_in_rpm: float = 2400.0                   # generator rpm; VERIFY against Rotax 915 iS Operators Manual
    rated_current_a: float = 30.0                # VERIFY against Rotax 915 iS Operators Manual
    drive_ratio: float = 1.0                     # generator/crank; VERIFY against Rotax 915 iS Operators Manual
    # SIMULATION PLACEHOLDER - NOT a Rotax figure. The 915 iS internal
    # generator pole count is unknown (engine.generator_poles = None, VERIFY).
    generator_poles_placeholder: int = 12
    phases: int = 3
    # Battery (basis: generic 12 V AGM lead-acid; VERIFY against installed battery)
    battery_capacity_ah: float = 17.0
    battery_r_int_ohm: float = 0.015
    # Charge polarisation (RC): builds up only while charging, so a nearly full
    # lead-acid battery accepts a few amps at 14 V in steady state while its
    # instantaneous resistance stays R_int. VERIFY against installed battery.
    battery_charge_pol_ohm: float = 0.3
    battery_charge_pol_tau_s: float = 30.0
    battery_ocv_soc: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    battery_ocv_v: list[float] = Field(default_factory=lambda: [11.8, 12.2, 12.4, 12.6, 12.8])
    initial_soc: float = 0.9
    # Electrical load (avionics, pumps): random steps (basis: project choice)
    base_load_a: float = 12.0
    load_min_a: float = 4.0
    load_max_a: float = 22.0
    load_step_min_a: float = 3.0
    load_step_max_a: float = 8.0
    load_step_mean_interval_s: float = 6.0      # 0 disables load steps
    # Ripple: fraction of the rectifier ripple reaching the bus (battery-filtered)
    ripple_filter_gain: float = 0.1
    bus_v_noise_v: float = 0.005
    # Voltage burst: fs >= 2.5 x max ripple fundamental (6 x f_e at 5800 rpm with
    # the placeholder poles = 3480 Hz -> >= 8700 Hz); 10240 Hz chosen.
    bus_v_burst_fs_hz: float = 10240.0
    bus_v_burst_samples: int = 2048
    integration_step_s: float = 0.01
    # Sensor chain (simulator's own copy; mirrors the hardware calibration)
    bus_v_full_scale_v: float = 20.0
    alt_i_full_scale_a: float = 50.0
    batt_i_full_scale_a: float = 100.0
    current_noise_a: float = 0.05
    # Fault parameters (physical perturbations, SRD-FUN-152)
    setpoint_drift_max_v: float = 2.0           # at severity 1
    setpoint_drift_ramp_s: float = 30.0
    open_diode_capacity_loss: float = 1.0 / 3.0
    battery_r_int_growth: float = 4.0           # R_int multiplier at severity 1, end of ramp
    battery_capacity_fade: float = 0.5
    battery_degradation_ramp_s: float = 120.0


class SimulatorInjectionConfig(BaseModel):
    """Simulator ECU injection/ignition schedules, fuel rail and per-cylinder
    combustion energy balance (Prompt 13). Simulator-only constants.

    The SOI and ignition-advance maps are ECU CONTROL SCHEDULES, not engine
    properties. Every value here is a placeholder: no Rotax 915 iS ECU
    calibration, injector or fuel-pump data was available.
    """

    enabled: bool = True
    # Injector (VERIFY against Rotax 915 iS Operators Manual / injector data)
    injector_static_flow_kg_s: float = 0.004     # at the reference rail differential pressure; VERIFY
    injector_dead_time_us: float = 900.0         # opening delay at nominal bus voltage; VERIFY
    rail_dp_ref_pa: float = 300000.0             # regulated rail - manifold pressure; VERIFY
    # The ECU computes PW for the reference rail pressure (no rail-pressure
    # compensation modelled). VERIFY whether the 915 iS ECU compensates.
    # Fuel pump: Q = Q_max * health * (1 - p_gauge / p_stall); the regulator
    # returns the excess while the pump can hold rail_dp_ref. VERIFY.
    pump_max_flow_kg_s: float = 0.03
    pump_stall_gauge_pa: float = 600000.0
    # ECU start-of-injection map, degrees after the TDC reference edge (placeholder; VERIFY)
    soi_rpm_axis: list[float] = Field(default_factory=lambda: [1500.0, 3000.0, 4500.0, 5800.0])
    soi_map_axis_pa: list[float] = Field(default_factory=lambda: [60000.0, 160000.0])
    soi_table_deg: list[list[float]] = Field(default_factory=lambda: [
        [300.0, 300.0], [330.0, 330.0], [350.0, 350.0], [370.0, 370.0]])
    # ECU ignition-advance map, degrees BTDC (placeholder; VERIFY)
    ign_rpm_axis: list[float] = Field(default_factory=lambda: [1500.0, 2500.0, 3500.0, 4500.0, 5800.0])
    ign_map_axis_pa: list[float] = Field(default_factory=lambda: [60000.0, 100000.0, 140000.0, 160000.0])
    ign_table_deg: list[list[float]] = Field(default_factory=lambda: [
        [20.0, 18.0, 15.0, 13.0],
        [26.0, 24.0, 20.0, 17.0],
        [30.0, 28.0, 24.0, 20.0],
        [32.0, 30.0, 26.0, 22.0],
        [33.0, 31.0, 27.0, 23.0]])
    # Timer input-capture jitter on the injection/ignition intervals
    timing_noise_us: float = 1.0
    # Per-cylinder exhaust energy balance: EGT rise ∝ x(λ, retard) · η_c(λ) · f / (1 + f),
    # f = 1 / (AFR_st λ), η_c = η_c,max · min(1, λ) (simulator.physics).
    # x = fraction of the released heat leaving with the exhaust. It falls
    # rich of stoichiometric and rises lean (slower burn), saturating, so the
    # peak EGT lies slightly lean of stoichiometric. Basis: general SI/aviation
    # leaning behaviour (peak EGT slightly lean of stoichiometric); values VERIFY.
    egt_x_stoich: float = 0.30
    egt_x_rich_slope: float = 0.20
    egt_x_lean_gain: float = 0.08
    egt_x_lean_scale: float = 0.08
    egt_x_per_deg_retard: float = 0.012         # later combustion -> more heat to exhaust; VERIFY
    imep_loss_per_deg_retard: float = 0.005     # VERIFY
    # Faults (physical parameter perturbations, SRD-FUN-152)
    # INJECTOR_FAULT: severity = fractional change of the cylinder's flow
    # coefficient; sub_mode "clog" lowers it, "leak" raises it.
    fuel_pump_capacity_loss: float = 0.85       # FUEL_SYSTEM_FAULT at severity 1
    knock_max_retard_deg: float = 8.0           # ECU knock retard at DETONATION_KNOCK severity 1; VERIFY
    # Sensor chain (simulator copy of the hardware calibration)
    fuel_press_full_scale_pa: float = 1000000.0  # gauge
    fuel_press_noise_pa: float = 1500.0


class SimulatorCoolingConfig(BaseModel):
    """Simulator liquid-coolant loop (Prompt 14). Simulator-only constants,
    independent of L2. All VERIFY: no Rotax 915 iS cooling data was available.

    Heat from the heads Q = head_heat_fraction x fuel power flows into a lumped
    coolant mass; a thermostat (min leak .. fully open) routes coolant through
    a crossflow radiator: G = epsilon(NTU, C_r) C_min, with the air side
    rho v A_face f_duct and UA scaling with airspeed^0.8. CHT = coolant +
    Q / hA_head (hA scaling with coolant flow^0.8). The healthy steady state
    at the reference ambient and airspeed reproduces the simulator's CHT map;
    departures from it (ambient, airspeed, faults, thermal lag) are added.
    """

    enabled: bool = True
    head_heat_fraction: float = 0.10          # of fuel power to the coolant (heads only liquid-cooled)
    coolant_flow_ref_kg_s: float = 1.0        # at pump_ref_rpm; pump flow proportional to rpm
    pump_ref_rpm: float = 5800.0
    coolant_cp_j_kg_k: float = 3500.0          # 50/50 glycol
    coolant_mass_kg: float = 2.5
    head_heat_capacity_j_k: float = 9000.0     # wetted head metal
    radiator_face_area_m2: float = 0.06
    duct_flow_fraction: float = 0.3
    radiator_ua_ref_w_k: float = 1500.0        # at airspeed_ref_m_s
    airspeed_ref_m_s: float = 45.0             # also the default airspeed when a profile gives none
    reference_ambient_k: float = 288.15
    thermostat_open_c: float = 85.0
    thermostat_full_open_c: float = 95.0
    thermostat_min_open: float = 0.02
    head_ha_ref_w_k: float = 1450.0            # head-to-coolant at coolant_flow_ref_kg_s
    oil_coolant_coupling: float = 0.5          # oil temperature change per K of coolant change
    oil_cooler_ambient_gain: float = 1.0       # air-cooled oil cooler at fixed load: dT_oil/dT_amb
    integration_step_s: float = 0.5
    # COOLING_FAULT sub-modes (physical parameter perturbations, SRD-FUN-152)
    blockage_max: float = 0.8                  # radiator air path blocked at severity 1
    pump_loss_max: float = 0.8                 # coolant flow lost at severity 1
    coolant_loss_max: float = 0.7              # coolant mass lost at severity 1
    coolant_loss_head_ha_factor: float = 0.6   # head-to-coolant hA lost at severity 1 (vapour/air)
    fault_ramp_s: float = 0.0                  # severity ramps from onset over this time (0: step)
    # NTC (simulator copy of the sensor datasheet) and resistance noise
    ntc_sh_a: float = 1.129148e-3
    ntc_sh_b: float = 2.34125e-4
    ntc_sh_c: float = 8.76741e-8
    ntc_noise_rel: float = 0.001


class SimulatorConfig(BaseModel):
    """Physics simulator configuration."""

    time_step_s: float = 0.001
    cycles_per_output: int = 1
    default_rpm: float = 4000.0
    default_throttle_pct: float = 50.0
    default_altitude_m: float = 1000.0
    default_ambient_temp_k: float = 288.15
    default_ambient_pressure_pa: float = 101325.0
    sensor_noise: SensorNoiseConfig = Field(default_factory=SensorNoiseConfig)
    sensor_dropout_probability: float = 0.001
    physics: SimulatorPhysicsConfig = Field(default_factory=SimulatorPhysicsConfig)
    electrical: SimulatorElectricalConfig = Field(default_factory=SimulatorElectricalConfig)
    injection: SimulatorInjectionConfig = Field(default_factory=SimulatorInjectionConfig)
    cooling: SimulatorCoolingConfig = Field(default_factory=SimulatorCoolingConfig)


class PipelineConfig(BaseModel):
    """Main processing pipeline settings."""

    cycle_timeout_ms: int = 500
    max_invalid_channels_before_halt: int = 10
    residual_window_size: int = 60
    enable_persistence: bool = True
    persistence_interval_s: float = 5.0


class HealthIndexWeights(BaseModel):
    """Weights for health index aggregation."""

    thermal: float = 0.25
    mechanical: float = 0.25
    combustion: float = 0.25
    lubrication: float = 0.15
    vibration: float = 0.10


class DegradationThresholds(BaseModel):
    """Health index thresholds for degradation state classification."""

    healthy: float = 0.8
    watch: float = 0.6
    caution: float = 0.4
    warning: float = 0.2
    critical: float = 0.0


class MLConfig(BaseModel):
    """Machine learning pipeline configuration."""

    anomaly_threshold: float = 0.85
    fault_confidence_threshold: float = 0.70
    # Trained models (Prompt 16): None = no trained models loaded (L3 reports
    # MODEL_UNAVAILABLE and the rule fallback runs). Set to a version under
    # model_dir (e.g. "v1") to load <model_dir>/<version>/*.joblib.
    model_version: str | None = None
    # Lifetime RUL bundle (Prompt 17): <model_dir>/<rul_model_version>/rul_lifetime.joblib.
    # Used only when set AND the engine-hours channel is valid; otherwise the
    # unvalidated baseline RUL is reported, marked validated=False.
    rul_model_version: str | None = None
    rul_horizon_hours: float = 500.0
    health_index_weights: HealthIndexWeights = Field(default_factory=HealthIndexWeights)
    degradation_states: DegradationThresholds = Field(default_factory=DegradationThresholds)


class AdvisorySeverityConfig(BaseModel):
    """Health-index thresholds for advisory severity levels."""

    monitor: float = 0.7
    schedule_maintenance: float = 0.5
    immediate_inspection: float = 0.3
    ground_aircraft: float = 0.1


class AdvisoryConfig(BaseModel):
    """Advisory layer configuration."""

    severity_levels: AdvisorySeverityConfig = Field(default_factory=AdvisorySeverityConfig)


class ChannelRangeConfig(BaseModel):
    """Min/max range for a single channel."""

    min: float
    max: float


class SecurityConfig(BaseModel):
    """Telemetry integrity and security configuration."""

    hmac_algorithm: str = "sha256"
    secret_key_env_var: str = "TELEMETRY_SECRET_KEY"
    default_secret_key: str = "dev_prototype_secret_key_change_in_prod"
    require_signature: bool = True
    max_sequence_gap: int = 1000
    stale_timeout_s: float = 5.0
    # Bearer token for state-changing ML endpoints (POST /ml/rollback; Prompt 17).
    # Read from this environment variable; if it is unset those endpoints refuse (fail closed).
    api_bearer_token_env_var: str = "TWIN_API_BEARER_TOKEN"
    per_channel_stale_timeouts_s: dict[str, float] = Field(
        default_factory=lambda: {
            "egt_cyl1_hot_uv": 3.0,
            "egt_cyl2_hot_uv": 3.0,
            "egt_cyl3_hot_uv": 3.0,
            "egt_cyl4_hot_uv": 3.0,
            "crank_period_us": 2.0,
            "oil_p_counts": 3.0,
        }
    )


class PersistenceConfig(BaseModel):
    """Raw telemetry persistence configuration."""

    enabled: bool = True
    storage_backend: str = "in_memory"
    storage_path: str = "data/raw_telemetry.db"
    max_records: int = 100000


class SensorCalibrationConfig(BaseModel):
    """Sensor calibration coefficients and constants for inverse modelling."""

    # type_k_uv_per_c and pt100_alpha_per_c are nominal sensitivities only,
    # kept for reference/compatibility. They are NOT used for conversion:
    # thermocouples use NIST ITS-90 Type K and the RTD uses IEC 60751
    # Callendar-Van Dusen (src.core.sensor_physics).
    type_k_uv_per_c: float = 41.27
    pt100_r0_ohms: float = 100.0
    pt100_alpha_per_c: float = 0.00385
    map_full_scale_pa: float = 200000.0
    oil_p_full_scale_pa: float = 800000.0
    fuel_flow_kg_s_per_hz: float = 0.0001
    accel_counts_per_m_s2: float = 10.0
    # Electrical sensor chain (Prompt 12), ratiometric against adc_vref_counts
    # like the pressure sensors (SRD-FUN-004). Calibration basis: divider and
    # Hall-sensor selection of the monitoring hardware - VERIFY against the
    # hardware selection. Not Rotax figures.
    bus_v_full_scale_v: float = 20.0      # divider: 20 V at the ADC reference
    alt_i_full_scale_a: float = 50.0      # unidirectional Hall: 0 A .. 50 A over full scale
    batt_i_full_scale_a: float = 100.0    # bidirectional Hall: -50 A .. +50 A over full scale
    batt_i_zero_fraction: float = 0.5     # zero current at mid-scale; positive = discharge
    # Fuel rail pressure (Prompt 13): gauge sensor, ratiometric; VERIFY against the hardware selection
    fuel_press_full_scale_pa: float = 1000000.0
    # Coolant NTC (Prompt 14), Steinhart-Hart 1/T = A + B ln R + C (ln R)^3.
    # Basis: the widely published example coefficients for a 10 kOhm (25 C)
    # NTC thermistor (Steinhart & Hart, Deep-Sea Research 15, 1968, equation
    # form). NOT the Rotax coolant sensor: VERIFY against the fitted sensor's
    # datasheet and replace.
    ntc_sh_a: float = 1.129148e-3
    ntc_sh_b: float = 2.34125e-4
    ntc_sh_c: float = 8.76741e-8


class EGTDiagnosticConfig(BaseModel):
    """Configuration for per-cylinder EGT diagnostics."""

    egt_max_k: float = 1200.0
    egt_warning_temp_k: float = 1123.15      # 850°C
    egt_critical_temp_k: float = 1173.15     # 900°C
    egt_spread_warning_k: float = 50.0
    egt_spread_critical_k: float = 80.0
    egt_dev_warning_k: float = 35.0
    egt_dev_critical_k: float = 60.0
    egt_rate_warning_k_s: float = 15.0
    egt_rate_critical_k_s: float = 30.0


class VibrationConfig(BaseModel):
    """Vibration signal processing configuration."""

    sampling_frequency_hz: float = 2048.0
    window_size: int = 2048
    overlap_pct: float = 0.0
    window_function: str = "hanning"
    freq_band_low_max_hz: float = 100.0
    freq_band_mid_max_hz: float = 500.0
    freq_band_high_max_hz: float = 1000.0
    # Envelope-demodulation band for bearing early warning (M-04).
    # Above the firing harmonics (6X at 5800 rpm = 580 Hz), below Nyquist (OI-9).
    envelope_band_low_hz: float = 650.0
    envelope_band_high_hz: float = 1000.0
    envelope_taper_hz: float = 50.0
    envelope_time_taper_alpha: float = 0.1
    # Accelerometer axis orientation (SRD-FUN-070; Prompt 15): which burst axis
    # is lateral and which vertical. VERIFY against the sensor installation.
    axis_lateral: str = "y"
    axis_vertical: str = "z"
    # Rotating imbalance evidence (M-12): 1X of the propeller shaft is
    # "dominant" when it holds more than half of the lateral velocity energy,
    # and lateral when its lateral/vertical amplitude ratio exceeds 1.
    imbalance_dominance_fraction: float = 0.5
    imbalance_lateral_ratio_min: float = 1.0
    # Vibration Health Index (M-10) references, interpolated in rpm.
    # HEALTHY INSTALLATION BASELINE -- PROVISIONAL, SIMULATOR-DERIVED
    # (scripts/derive_vhi_references.py); replace with a baseline captured
    # from the first healthy flights (OI-9). Units: overall RMS of the
    # acceleration magnitude [m/s^2], crest [-], envelope RMS [m/s^2].
    # Nominal simulator at MAP 110 kPa, re-derived after the M-04 band moved to
    # 650-1000 Hz with a tapered band-pass (OI-9). Healthy VHI over
    # 2000-5800 rpm x 80-135 kPa: 0.995-1.003 (was 0.609-1.877 with the
    # 300-1000 Hz brick-wall band). The envelope reference is now the sensor
    # noise floor (~0.071 m/s^2), nearly independent of rpm.
    vhi_ref_rpm: list[float] = Field(default_factory=lambda: [2000.0, 3000.0, 4000.0, 5000.0, 5800.0])
    vhi_ref_rms_m_s2: list[float] = Field(default_factory=lambda: [7.443, 8.59, 9.726, 10.9, 11.819])
    vhi_ref_crest: list[float] = Field(default_factory=lambda: [1.583, 1.58, 1.555, 1.563, 1.549])
    vhi_ref_envelope_m_s2: list[float] = Field(default_factory=lambda: [0.0706, 0.0711, 0.072, 0.0713, 0.072])
    # M-10 self-test references, kept for documentation only (NOT active).
    # M-10 labels the RMS as m/s^2, but 2.0 is typical of ISO 10816 velocity
    # severity in mm/s; unconfirmed, not converted (OI-9).
    vhi_m10_reference_rms: float = 2.0
    vhi_m10_reference_crest: float = 2.5
    vhi_m10_reference_envelope: float = 0.05


class MisfireConfig(BaseModel):
    """Configuration for combustion stability and misfire diagnostics."""

    egt_drop_threshold_k: float = 50.0
    egt_dev_threshold_k: float = 40.0
    egt_rate_drop_threshold_k_s: float = 20.0
    rpm_std_threshold: float = 50.0
    alpha_std_threshold: float = 15.0
    vibration_rms_threshold_m_s2: float = 15.0
    misfire_confidence_threshold: float = 0.70
    # Combustion Stability Index (M-09). CHT term uses the slope RESIDUAL
    # against the slope of residual_engine.expected_cht_k over the window.
    csi_window_s: float = 30.0            # least-squares CHT slope window
    csi_min_samples: int = 3
    csi_reference_cht_slope_k_s: float = 0.5
    # Transient gate for the CSI thermal term: computed only when brake power
    # and MAP are steady over the window; otherwise excluded (invalid).
    csi_transient_max_power_rate_kw_s: float = 0.5
    csi_transient_max_map_rate_pa_s: float = 500.0
    unstable_confidence_threshold: float = 0.40
    egt_evidence_weight: float = 0.40
    crank_evidence_weight: float = 0.35
    vibration_evidence_weight: float = 0.25
    # Dual-channel AND gate (M-06 defaults). A misfire is CONFIRMED only when
    # crank-period CoV (torsional) AND half-order fraction (structural) both
    # reach their thresholds.
    # Channel A limit = max(gate_crank_cov_floor_pct,
    #                       gate_baseline_multiplier x gate_healthy_crank_cov_pct).
    # VERIFY: the 0.30 % floor is an engineering estimate (one missed power
    # stroke on this driveline ~0.8 % CoV, OI-7); healthy CoV is unknown until
    # flight data exists (None -> floor applies).
    gate_crank_cov_floor_pct: float = 0.30
    gate_healthy_crank_cov_pct: float | None = None
    gate_baseline_multiplier: float = 3.0            # M-06
    # M-06 default fixed threshold, documented reference only (NOT active):
    gate_crank_cov_threshold_pct: float = 2.0
    gate_half_order_threshold: float = 0.03
    gate_crank_window_revs: int = 32
    # CONFIRMED -> CRITICAL (alarm) when both channels reach this multiple of
    # their limits; otherwise WARNING. Not an M-06 value; project choice.
    gate_alarm_margin: float = 2.0


class ElectricalConfig(BaseModel):
    """L2 electrical-health model constants (Prompt 12).

    Expectations for the charging system and battery. Values marked VERIFY
    are placeholders to be checked against the Rotax 915 iS Operators Manual
    and the installed battery; the rest state their calibration basis.
    """

    regulator_setpoint_v: float = 14.2       # VERIFY against Rotax 915 iS Operators Manual
    cut_in_rpm: float = 2400.0               # engine rpm; VERIFY against Rotax 915 iS Operators Manual
    battery_ocv_nominal_v: float = 12.7      # resting OCV, charged 12 V battery; VERIFY against battery data
    battery_r_int_nominal_mohm: float = 15.0 # healthy internal resistance; VERIFY against battery data
    # Bands (project choice): charging residual |V - expected|
    charging_warning_v: float = 0.5
    charging_alarm_v: float = 1.0
    # Ripple: healthy level and bands as ratio to it (project choice; VERIFY in flight)
    ripple_healthy_pct: float = 0.5
    ripple_warning_ratio: float = 2.0
    ripple_alarm_ratio: float = 3.0
    # Internal-resistance bands as ratio to nominal (project choice)
    r_int_warning_ratio: float = 1.5
    r_int_alarm_ratio: float = 2.5
    # R_int estimator: -dV/dI_batt over load steps
    r_est_min_di_a: float = 1.0          # battery-current step needed (A)
    r_est_max_rpm_step: float = 50.0     # rpm change allowed across the step
    r_est_median_n: int = 7              # robust median over the last N steps
    r_est_min_steps: int = 3             # steps before R_int is reported
    r_est_max_dt_s: float = 1.0          # the two samples must be at most this far apart
    # Observable only while the battery supplies the step: below cut-in, or
    # alternator current below this (alternator not in control). Project choice.
    r_est_max_alt_i_a: float = 1.0
    # Charging residual band decisions use the median of the last N records,
    # so a single load-step transient does not trip the band (project choice).
    residual_median_n: int = 10
    # Ripple order (ripple Hz / generator rev/s) must be within this of an
    # integer for the ripple to be shaft-locked (rpm-scaling check).
    ripple_order_tolerance: float = 0.25
    # Ripple peak must exceed the in-band median power by this factor to count
    # as a line (not a noise peak). Project choice.
    ripple_peak_snr: float = 100.0  # white-noise peak/median over ~800 bins is ~10; real line >1e4
    # Electrical Health Index: component factor f = 0.5 ** d, d = deviation
    # normalised so d = 1 at that component's alarm threshold; EHI is the
    # weighted log-space product re-normalised over valid components (LHI
    # pattern). Bands and weights are project choices.
    ehi_weight_charging: float = 0.5
    ehi_weight_ripple: float = 0.25
    ehi_weight_r_int: float = 0.25
    ehi_warning: float = 0.75
    ehi_alarm: float = 0.5
    # Ripple analysis band; must lie below the voltage-burst Nyquist limit (M-05)
    ripple_band_low_hz: float = 50.0
    ripple_band_high_hz: float = 4000.0
    bus_v_burst_fs_nominal_hz: float = 10240.0


class InjectionConfig(BaseModel):
    """L2 injection, ignition and fuel-system model (Prompt 13).

    Injector and regulator values are component-datasheet placeholders (the
    same source a real installation would use); the ignition map is a copy of
    the ECU control schedule. All VERIFY against Rotax 915 iS Operators Manual
    / ECU calibration. Bands are project choices.
    """

    injector_static_flow_kg_s: float = 0.004     # K_inj at rail_dp_ref_pa; VERIFY
    injector_dead_time_us: float = 900.0         # VERIFY
    rail_dp_ref_pa: float = 300000.0             # regulated rail - manifold pressure; VERIFY
    # Expected rail differential at the current demand: setpoint - droop x demand
    rail_dp_setpoint_pa: float = 300000.0        # VERIFY
    rail_droop_pa_per_kg_s: float = 0.0          # ideal regulator unless data says otherwise; VERIFY
    rail_residual_warning_pa: float = 30000.0
    rail_residual_alarm_pa: float = 60000.0
    duty_saturation_pct: float = 85.0
    fuel_delivery_warning: float = 0.05          # |ratio - 1|
    fuel_delivery_alarm: float = 0.10
    injector_flow_warning: float = 0.05          # |flow ratio - 1|
    injector_flow_alarm: float = 0.10
    median_n: int = 10                           # records in the running medians
    stoichiometric_afr: float = 14.7
    # L2's own EGT-vs-mixture energy balance (independent of the simulator's):
    # rise ∝ x(λ) η_c(λ) f/(1+f), x = x_st - s_r (1-λ) rich, x_st + g (1 - exp(-(λ-1)/w)) lean.
    egt_x_stoich: float = 0.32
    egt_x_rich_slope: float = 0.25
    egt_x_lean_gain: float = 0.10
    egt_x_lean_scale: float = 0.10
    combustion_efficiency_max: float = 0.98
    lambda_search_min: float = 0.55
    # The flow ratio is derived only while the operating mixture is rich of the
    # model's peak-EGT λ by at least this margin (EGT monotonic in λ there).
    lambda_peak_margin: float = 0.03
    charge_temp_rise_k: float = 30.0             # charge over ambient (no IAT channel); VERIFY
    # Ignition: ECU schedule copy (placeholder; VERIFY) and knock evidence
    ign_rpm_axis: list[float] = Field(default_factory=lambda: [1500.0, 2500.0, 3500.0, 4500.0, 5800.0])
    ign_map_axis_pa: list[float] = Field(default_factory=lambda: [60000.0, 100000.0, 140000.0, 160000.0])
    ign_table_deg: list[list[float]] = Field(default_factory=lambda: [
        [20.0, 18.0, 15.0, 13.0],
        [26.0, 24.0, 20.0, 17.0],
        [30.0, 28.0, 24.0, 20.0],
        [32.0, 30.0, 26.0, 22.0],
        [33.0, 31.0, 27.0, 23.0]])
    knock_retard_threshold_deg: float = 2.0      # consistent retard (median) on a cylinder
    knock_egt_support_k: float = 15.0            # differential EGT rise supporting knock retard
    knock_cht_support_k: float = 10.0            # CHT residual rise over the running minimum


class CoolantModelConfig(BaseModel):
    """L2 coolant expectation and bands (Prompt 14). Project choices / VERIFY.

    Expected coolant temperature (M-11 style, operating point only): the
    thermostat holds the coolant near its regulated temperature, rising with
    load and ambient once it opens further.
    """

    regulated_c: float = 90.0                 # VERIFY (thermostat band 85-95 C)
    k_load_k: float = 6.0                      # per unit load (MAP/rated x rpm/rated)
    k_ambient: float = 0.1                     # per K ambient above ISA SL
    scale_k: float = 5.0
    residual_warning_k: float = 8.0
    residual_alarm_k: float = 15.0
    cht_delta_warning_k: float = 15.0          # (CHT - coolant) above its expectation
    cht_delta_alarm_k: float = 30.0
    median_n: int = 10


class DriftConfig(BaseModel):
    """Residual drift monitoring (Prompt 17): Population Stability Index of the
    recent residual window vs the baseline reference histogram per channel."""

    # PSI on unchanged data averages about (bins - 1)(1/n_ref + 1/n_window)
    # (small-sample bias). 5 bins and >= 200 records each keep that near 0.04,
    # well below the 0.10 band; 10 bins with 60/100 records gave ~0.24.
    bins: int = 5
    psi_moderate: float = 0.10          # < moderate: stable
    psi_significant: float = 0.25       # >= significant: significant
    reference_min_records: int = 200    # records needed to build a reference
    window_records: int = 200           # recent records compared with the reference
    min_window_records: int = 200
    epsilon: float = 1e-4               # probability floor in the PSI


class AdaptationConfig(BaseModel):
    """Gated per-engine baseline adaptation (Prompt 17). Project choices, VERIFY
    against fleet data; the gate itself is not tunable away (see adaptation.py)."""

    commissioning_flights: int = 3
    # Target of the correction (Prompt 17b): the fleet's HEALTHY residual at the
    # same operating point (src/l3_ml/fleet_reference.py), read from
    # <model_dir>/<fleet_reference_version>/fleet_residual_reference.json.
    # None: no reference, so no engine is commissioned (nothing is learned).
    fleet_reference_version: str | None = None
    # |a| bound for the commissioning fit: the engine's deviation from the fleet
    # reference (installation offset plus ageing so far), NOT the expectation
    # bias any more (Prompt 17b); b bound for the linear term. VERIFY.
    commissioning_max_offset: dict[str, float] = Field(default_factory=lambda: {
        "egt": 60.0, "cht": 30.0, "coolant": 15.0, "oil_temp": 20.0, "oil_pressure": 50000.0,
        "fuel_flow": 0.002, "vibration_rms": 5.0, "brake_power_kw": 15.0})
    max_linear_gain: float = 0.5
    # after commissioning: offset change per adapted flight and total vs commissioning
    rate_limit_per_flight: dict[str, float] = Field(default_factory=lambda: {
        "egt": 3.0, "cht": 1.5, "coolant": 1.0, "oil_temp": 1.5, "oil_pressure": 3000.0,
        "fuel_flow": 0.0001, "vibration_rms": 0.3, "brake_power_kw": 1.0})
    max_total_from_commissioning: dict[str, float] = Field(default_factory=lambda: {
        "egt": 40.0, "cht": 15.0, "coolant": 8.0, "oil_temp": 12.0, "oil_pressure": 30000.0,
        "fuel_flow": 0.0006, "vibration_rms": 2.0, "brake_power_kw": 6.0})
    fault_confidence_threshold: float = 0.70   # ML class other than NOMINAL above this -> ineligible
    challenger_after_flights: int = 20
    promotion_max_f1_drop: float = 0.02
    promotion_max_pr_auc_drop: float = 0.02
    promotion_max_fpr: float = 0.05
    store_dir: str = "models/adaptation"


class OverheatTrendConfig(BaseModel):
    """Overheating trend prediction (Prompt 14). Theil-Sen slope of the
    residual (measured - lagged M-11 expectation) over a rolling window; a
    time-to-limit is reported only for a positive slope whose confidence
    interval excludes zero and exceeds min_slope_k_per_min. Limits are the
    Appendix B alarm limits held elsewhere in config (the Appendix B text is
    not in the repository; VERIFY each):
        CHT      cooling.max_cht_c
        EGT      egt_diagnostics.egt_critical_temp_k
        oil      lubrication.lhi_over_temp_limit_c
        coolant  cooling.max_coolant_c
    """

    window_s: float = 180.0
    min_samples: int = 20
    confidence: float = 0.95
    min_slope_k_per_min: float = 0.2           # practical significance; project choice
    alert_horizon_s: float = 1800.0            # advisory when time-to-limit is within this
    # Expectation lag (L2's own thermal time constants; VERIFY): the expected
    # trajectory follows the operating point with this first-order lag, so a
    # healthy climb gives a slope residual near zero.
    lag_cht_s: float = 60.0
    lag_coolant_s: float = 90.0
    lag_oil_s: float = 180.0
    lag_egt_s: float = 5.0
    # An abrupt operating-point change between consecutive records restarts
    # the trend windows: the residual then jumps (an offset, not a rate).
    # Project choice (throttle movement, not a climb).
    reset_rpm_step: float = 500.0
    reset_map_step_pa: float = 20000.0
    reset_ambient_step_k: float = 5.0


class HealthyBaselineConfig(BaseModel):
    """Healthy expectation baseline parameters and normalization scales.

    Not calibrated to the simulator: these are the model defaults, and they
    disagree with it (docs/OPEN_ITEMS.md, OI-5). They need healthy-flight data.
    """

    base_egt_k: float = 950.0
    k_egt_map_pa: float = 0.002
    k_egt_rpm: float = 0.05
    base_oil_temp_rise_k: float = 65.0
    base_vibration_rms_m_s2: float = 5.0
    k_vib_rpm: float = 10.0
    scale_egt_k: float = 50.0
    scale_oil_pressure_pa: float = 50000.0
    scale_oil_temp_k: float = 10.0
    scale_vibration_m_s2: float = 5.0
    scale_power_kw: float = 15.0
    # CHT expectation (M-11). Starting values; re-fit to healthy flights.
    base_cht_k: float = 398.15
    k_cht_load_k: float = 65.0
    k_cht_ambient: float = 0.85
    k_cht_airspeed: float = -18.0
    scale_cht_k: float = 12.0
    # Fuel-flow expectation (M-11), keyed on EXPECTED brake power.
    baseline_sfc_kg_kwh: float = 0.312
    # Expected brake power = rated * (MAP / expected_power_map_ref_pa) * (rpm / 5800)
    expected_power_map_ref_pa: float = 140000.0
    idle_fuel_kg_s: float = 0.00035
    scale_fuel_kg_s: float = 0.0004
    # Per-cylinder EGT installation offsets, cylinders 1..4 [K]; rear pair of
    # the boxer four runs hotter. Sum to zero. Re-fit per airframe.
    egt_cylinder_offsets_k: list[float] = Field(default_factory=lambda: [-8.0, -3.0, 4.0, 7.0])


class HealthConfig(BaseModel):
    """Configuration for engine health index and degradation supervision."""

    # Rebalanced in Prompt 12 to add electrical health: electrical 0.10 and the
    # six existing weights scaled by 0.9 (ratios kept). Sum = 1.0.
    # Previous: thermal 0.20, lubrication 0.20, vibration 0.20, combustion 0.15,
    # performance 0.10, anomaly_fault 0.15.
    weight_thermal: float = 0.18
    weight_lubrication: float = 0.18
    weight_vibration: float = 0.18
    weight_combustion: float = 0.135
    weight_performance: float = 0.09
    weight_anomaly_fault: float = 0.135
    weight_electrical: float = 0.10

    # Thresholds for DegradationState
    healthy_threshold: float = 0.85
    watch_threshold: float = 0.70
    caution_threshold: float = 0.50
    warning_threshold: float = 0.30

    # Hysteresis delta (prevents flickering near state boundaries)
    hysteresis_delta: float = 0.03

    # Minimum quality threshold required for valid health assessment
    min_evidence_quality: float = 0.20

    # Health-index trend: Theil-Sen slope over a rolling window with Sen's CI
    # (the Prompt 14 trend code). Window 120 s: project choice, basis - at the
    # 1 Hz record rate it holds ~120 samples for a stable CI and matches the
    # 120 s audit scenarios; the window restarts on operating-point steps
    # (overheat.reset_*). The rates keep the thresholds of the former one-step
    # dHI/dt method (DEGRADATION below -0.0001 /s = 0.006 /min, RAPID below
    # -0.01 /s = 0.6 /min), now applied to a robust windowed slope and only when
    # its confidence interval excludes zero.
    trend_window_s: float = 120.0
    trend_min_samples: int = 20
    trend_confidence: float = 0.95
    trend_min_rate_per_min: float = 0.006
    trend_rapid_rate_per_min: float = 0.6


class RULConfig(BaseModel):
    """Configuration for RUL prediction and history tracking."""

    min_history_samples: int = 3
    max_prediction_horizon_hours: float = 2000.0
    default_operating_assumption: str = "constant_cruise_operating_profile"
    confidence_level: float = 0.95
    history_window_max_samples: int = 1000
    baseline_degradation_target_hi: float = 0.20


class MissionConfig(BaseModel):
    """Configuration for mission phase classification and mission risk assessment."""

    # Phase detection thresholds
    ground_max_rpm: float = 1200.0
    ground_max_throttle_pct: float = 15.0
    takeoff_min_rpm: float = 5000.0
    takeoff_min_map_pa: float = 110000.0
    climb_min_rpm: float = 4500.0
    climb_min_vrate_m_s: float = 1.0
    descent_max_rpm: float = 3800.0
    descent_max_vrate_m_s: float = -1.0
    landing_max_altitude_m: float = 200.0
    # Vertical rate for the phase rules (Prompt 18, OI-23): least-squares slope
    # of the derived pressure altitude over this window. Calibration basis:
    # project choice, long enough (10 records at 1 Hz) that altitude noise does
    # not read as a climb or descent. VERIFY on flight data.
    vertical_rate_window_s: float = 10.0
    vertical_rate_min_points: int = 3

    # Phase risk multipliers
    phase_risk_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "TAKEOFF": 1.5,
            "CLIMB": 1.3,
            "LANDING": 1.2,
            "CRUISE": 1.0,
            "DESCENT": 1.0,
            "GROUND": 0.5,
            "UNKNOWN": 1.2,
        }
    )

    # Risk level thresholds
    low_risk_threshold: float = 0.25
    moderate_risk_threshold: float = 0.50
    high_risk_threshold: float = 0.75

    # Phase transition debounce / minimum persistence samples
    phase_debounce_samples: int = 2


class DeploymentConfig(BaseModel):
    """Deployment role and transport settings for Edge/Ground partition (Module 21)."""

    role: DeploymentRole = DeploymentRole.SIMULATION
    transport_backend: str = "in_memory"
    endpoint_url: str = "http://localhost:8000/api/v1/telemetry"
    timeout_s: float = 5.0
    retry_attempts: int = 3
    retry_backoff_s: float = 1.0
    buffer_capacity: int = 1000
    buffer_overflow_policy: OverflowPolicy = OverflowPolicy.DISCARD_OLDEST
    reconnect_interval_s: float = 2.0
    freshness_threshold_s: float = 5.0


# =============================================================================
# Top-Level Settings
# =============================================================================


class AppSettings(BaseSettings):
    """Top-level application settings.

    Load order (highest priority wins):
        1. Environment variables (prefixed APP_)
        2. .env file
        3. config/default.yaml

    Usage:
        settings = load_settings()
        bore = settings.engine.bore_mm
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Sub-configs ---
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    turbocharger: TurbochargerConfig = Field(default_factory=TurbochargerConfig)
    lubrication: LubricationConfig = Field(default_factory=LubricationConfig)
    cooling: CoolingConfig = Field(default_factory=CoolingConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    advisory: AdvisoryConfig = Field(default_factory=AdvisoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    sensor_calibration: SensorCalibrationConfig = Field(default_factory=SensorCalibrationConfig)
    egt_diagnostics: EGTDiagnosticConfig = Field(default_factory=EGTDiagnosticConfig)
    vibration: VibrationConfig = Field(default_factory=VibrationConfig)
    misfire: MisfireConfig = Field(default_factory=MisfireConfig)
    healthy_baseline: HealthyBaselineConfig = Field(default_factory=HealthyBaselineConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    rul: RULConfig = Field(default_factory=RULConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    electrical: ElectricalConfig = Field(default_factory=ElectricalConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    coolant: CoolantModelConfig = Field(default_factory=CoolantModelConfig)
    overheat: OverheatTrendConfig = Field(default_factory=OverheatTrendConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    channel_ranges: dict[str, ChannelRangeConfig] = Field(default_factory=dict)
    raw_channel_ranges: dict[str, ChannelRangeConfig] = Field(default_factory=dict)

    # --- Application-level ---
    version: str = "1.0.0"
    app_env: str = "development"
    log_level: str = "INFO"
    config_path: str = "config/default.yaml"
    database_url: str = "sqlite+aiosqlite:///./digital_twin.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    telemetry_source: str = "simulator"
    csv_replay_path: str = "data/sample_flights/"
    model_dir: str = "models/"


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Returns an empty dict if the file does not exist, so the system
    can start with pure defaults or environment variables.
    """
    config_file = Path(path)
    if config_file.exists():
        with config_file.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def _validate_vibration_bands(vib: VibrationConfig, error_cls: type[Exception]) -> None:
    """Fail at load time if any vibration analysis band is not representable.

    Uses M-05 validate_vibration_config, plus an explicit "at or above fs/2"
    check: M-05 only rejects edges strictly above Nyquist, but an edge exactly
    at fs/2 has no usable bin width either.
    """
    from src.core.sensor_physics.M05_nyquist_guard import validate_vibration_config

    fs = vib.sampling_frequency_hz
    problems = validate_vibration_config(
        sampling_frequency_hz=fs,
        freq_band_low_max_hz=vib.freq_band_low_max_hz,
        freq_band_mid_max_hz=vib.freq_band_mid_max_hz,
        freq_band_high_max_hz=vib.freq_band_high_max_hz,
        envelope_band_high_hz=vib.envelope_band_high_hz,
    )
    edges = {
        "freq_band_low_max_hz": vib.freq_band_low_max_hz,
        "freq_band_mid_max_hz": vib.freq_band_mid_max_hz,
        "freq_band_high_max_hz": vib.freq_band_high_max_hz,
        "envelope_band_low_hz": vib.envelope_band_low_hz,
        "envelope_band_high_hz": vib.envelope_band_high_hz,
    }
    for name, edge in edges.items():
        if fs > 0 and edge >= fs / 2.0 and not any(name in p for p in problems):
            problems.append(f"{name}: {edge:.1f} Hz is at or above the Nyquist limit {fs / 2.0:.1f} Hz")
    if not 0.0 < vib.envelope_band_low_hz < vib.envelope_band_high_hz:
        problems.append("envelope band must satisfy 0 < envelope_band_low_hz < envelope_band_high_hz")
    if problems:
        raise error_cls("Vibration configuration not representable at "
                        f"fs={fs:.1f} Hz: " + "; ".join(problems))


def _validate_ripple_band(elec: "ElectricalConfig", error_cls: type[Exception]) -> None:
    """M-05 for the bus-voltage ripple band against the nominal burst rate."""
    from src.core.sensor_physics.M05_nyquist_guard import assert_band_within_nyquist, NyquistViolation

    fs = elec.bus_v_burst_fs_nominal_hz
    try:
        assert_band_within_nyquist(elec.ripple_band_high_hz, fs, "ripple_band_high_hz")
    except NyquistViolation as exc:
        raise error_cls(f"Electrical ripple band not representable: {exc}") from exc
    if elec.ripple_band_high_hz >= fs / 2.0 or not 0.0 < elec.ripple_band_low_hz < elec.ripple_band_high_hz:
        raise error_cls(f"Electrical ripple band must satisfy 0 < low < high < fs/2 ({fs / 2.0:.0f} Hz)")


def load_settings(config_path: str | Path = "config/default.yaml") -> AppSettings:
    """Load and validate application settings.

    Merges YAML file values with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppSettings instance.

    Raises:
        ConfigurationError: If configuration values fail validation.
    """
    from pydantic import ValidationError
    from src.core.exceptions import ConfigurationError

    yaml_config = _load_yaml_config(config_path)
    try:
        settings = AppSettings(**yaml_config)
        # Custom sanity validation
        if settings.engine.bore_mm <= 0 or settings.engine.num_cylinders <= 0:
            raise ConfigurationError("Engine bore and cylinder count must be positive")
        if settings.telemetry.sample_rate_hz <= 0:
            raise ConfigurationError("Sample rate must be positive")
        _validate_vibration_bands(settings.vibration, ConfigurationError)
        _validate_ripple_band(settings.electrical, ConfigurationError)
        if settings.app_env.lower() in ("production", "prod"):
            import os
            env_secret = os.environ.get(settings.security.secret_key_env_var)
            if not env_secret or env_secret == "dev_prototype_secret_key_change_in_prod":
                raise ConfigurationError(
                    f"Production mode requires a secret key injected via environment variable '{settings.security.secret_key_env_var}'. The default development secret key is prohibited in production."
                )
        return settings
    except (ValidationError, ValueError) as e:
        raise ConfigurationError(f"Failed to validate configuration from {config_path}: {e}") from e



# Module-level singleton (lazily initialized)
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get the global settings singleton.

    Initializes from default config path on first call.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings singleton (useful for testing)."""
    global _settings  # noqa: PLW0603
    _settings = None
