"""
Simulator electrical forward model (L1, simulation only; Prompt 12).

Bus model: the alternator is a regulated source (setpoint V_set) with finite
stiffness (droop resistance R_reg) and a first-order response time, in
parallel with the battery (open-circuit voltage OCV(SoC) behind internal
resistance R_int), feeding a load that steps randomly:

    battery    :  E = OCV(SoC) + V_pol;  V = E - I_batt * R_int
                  I_batt = I_load - I_alt  (positive = discharge)
                  dV_pol/dt = (max(-I_batt, 0) * R_pol - V_pol) / tau_pol
                  (charge polarisation: a nearly full battery accepts only a
                  few amps in steady state; instantaneous resistance is R_int)
    static node:  V = (V_set/R_reg + E/R_int - I_load) / (1/R_reg + 1/R_int)
    alternator :  I_alt -> clamp((V_set - V)/R_reg, 0, I_max), lag tau

Below cut-in (generator rpm) the alternator delivers nothing and the battery
carries the load. Ripple comes from a 3-phase full-bridge rectifier:
rectified output = max(upper phases) - min(lower phases), scaled onto the bus
by a filter gain; an open diode removes one phase from the upper set, which
deepens the ripple (a topology change, not an offset: SRD-FUN-152).

Faults (FaultMode, applied as physical parameter changes):
    CHARGING_FAULT, sub_mode "setpoint_drift" : regulator setpoint drifts down
    CHARGING_FAULT, sub_mode "output_collapse": alternator capacity -> 0 above
                                                cut-in (drive/winding failure)
    CHARGING_FAULT, sub_mode "open_diode"     : one rectifier diode open
    BATTERY_DEGRADATION                       : R_int rises, capacity fades
All constants are in config (simulator.electrical), independent of the L2
constants (electrical.*), so the twin is tested against a separate model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.core.config import AppSettings, get_settings
from src.l1_data.simulator.fault_injection import FaultMode

SUB_MODES = ("setpoint_drift", "output_collapse", "open_diode")


@dataclass
class ElectricalTruth:
    """Electrical ground truth at one record time (simulation only)."""

    bus_voltage_v: float
    alternator_current_a: float
    battery_current_a: float          # positive = discharge
    load_current_a: float
    battery_ocv_v: float
    battery_soc: float
    battery_r_int_ohm: float
    regulator_setpoint_v: float
    alternator_available: bool
    ripple_frequency_hz: float | None  # rectifier ripple fundamental (None when alternator off)
    burst_v: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    burst_fs_hz: float = 0.0


class ElectricalSimulator:
    """Stateful electrical model; call step() once per record, in time order."""

    def __init__(self, settings: AppSettings | None = None, seed: int = 42) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.simulator.electrical
        self._rng = np.random.default_rng([seed, 0xE1])
        self._t: float | None = None
        self._soc = self.cfg.initial_soc
        self._i_alt: float | None = None
        self._load = self.cfg.base_load_a
        self._next_step_t: float | None = None
        self._v_pol: float | None = None

    # ------------------------------------------------------------------ model pieces
    def ocv(self, soc: float) -> float:
        return float(np.interp(soc, self.cfg.battery_ocv_soc, self.cfg.battery_ocv_v))

    def generator_rpm(self, engine_rpm: float) -> float:
        return engine_rpm * self.cfg.drive_ratio

    def ripple_fundamental_hz(self, engine_rpm: float, open_diode: bool = False) -> float:
        """Full-bridge ripple fundamental: 2 * phases * f_e (healthy); an open
        diode makes the pattern repeat once per electrical cycle (f_e)."""
        f_e = (self.cfg.generator_poles_placeholder / 2.0) * self.generator_rpm(engine_rpm) / 60.0
        return f_e if open_diode else 2.0 * self.cfg.phases * f_e

    def _fault_state(self, t: float, fault_mode: int | None, sub_mode: str | None,
                     severity: float, onset_s: float) -> dict:
        c = self.cfg
        state = {"setpoint": c.regulator_setpoint_v, "capacity": 1.0, "open_diode": False,
                 "r_int": c.battery_r_int_ohm, "capacity_ah": c.battery_capacity_ah}
        if fault_mode is None or t < onset_s:
            return state
        sev = min(max(severity, 0.0), 1.0)
        if int(fault_mode) == FaultMode.CHARGING_FAULT:
            mode = sub_mode or "setpoint_drift"
            if mode == "setpoint_drift":
                progress = min(1.0, (t - onset_s) / c.setpoint_drift_ramp_s)
                state["setpoint"] -= sev * c.setpoint_drift_max_v * progress
            elif mode == "output_collapse":
                state["capacity"] = 1.0 - sev
            elif mode == "open_diode":
                state["open_diode"] = True
                state["capacity"] = 1.0 - sev * c.open_diode_capacity_loss
            else:
                raise ValueError(f"unknown CHARGING_FAULT sub_mode {mode!r}; expected one of {SUB_MODES}")
        elif int(fault_mode) == FaultMode.BATTERY_DEGRADATION:
            progress = min(1.0, (t - onset_s) / c.battery_degradation_ramp_s)
            state["r_int"] = c.battery_r_int_ohm * (1.0 + sev * (c.battery_r_int_growth - 1.0) * progress)
            state["capacity_ah"] = c.battery_capacity_ah * (1.0 - sev * c.battery_capacity_fade * progress)
        return state

    def _advance_load(self, t: float) -> None:
        c = self.cfg
        if not c.load_step_mean_interval_s or c.load_step_mean_interval_s <= 0.0:
            return
        if self._next_step_t is None:
            self._next_step_t = t + self._rng.exponential(c.load_step_mean_interval_s)
        while self._next_step_t <= t:
            step = self._rng.uniform(c.load_step_min_a, c.load_step_max_a) * self._rng.choice([-1.0, 1.0])
            self._load = float(np.clip(self._load + step, c.load_min_a, c.load_max_a))
            self._next_step_t += self._rng.exponential(c.load_step_mean_interval_s)

    # ------------------------------------------------------------------ step
    def step(self, time_s: float, engine_rpm: float, fault_mode: int | None = None,
             sub_mode: str | None = None, severity: float = 0.0, onset_s: float = 0.0,
             enable_noise: bool = True) -> ElectricalTruth:
        c = self.cfg
        t_prev = self._t if self._t is not None else time_s
        fs = self._fault_state(time_s, fault_mode, sub_mode, severity, onset_s)
        available = self.generator_rpm(engine_rpm) >= c.cut_in_rpm and fs["capacity"] > 0.0
        i_max = c.rated_current_a * fs["capacity"] if available else 0.0

        g_reg = 1.0 / c.regulator_droop_ohm

        def target(emf: float, load: float, r_b: float) -> float:
            if not available:
                return 0.0
            v_node = (fs["setpoint"] * g_reg + emf / r_b - load) / (g_reg + 1.0 / r_b)
            return float(np.clip((fs["setpoint"] - v_node) * g_reg, 0.0, i_max))

        if self._i_alt is None:  # start at the steady state (polarisation settled)
            ocv0 = self.ocv(self._soc)
            self._i_alt = target(ocv0, self._load, fs["r_int"] + c.battery_charge_pol_ohm)
            self._v_pol = max(0.0, self._i_alt - self._load) * c.battery_charge_pol_ohm

        # integrate from the previous record time in small steps
        n_sub = max(1, int(round((time_s - t_prev) / c.integration_step_s)))
        dt = (time_s - t_prev) / n_sub if time_s > t_prev else 0.0
        for k in range(1, n_sub + 1):
            t_k = t_prev + k * dt
            self._advance_load(t_k)
            emf = self.ocv(self._soc) + self._v_pol
            self._i_alt += (target(emf, self._load, fs["r_int"]) - self._i_alt) * (
                1.0 - math.exp(-dt / c.regulator_time_constant_s))
            i_batt = self._load - self._i_alt
            self._v_pol += (max(-i_batt, 0.0) * c.battery_charge_pol_ohm - self._v_pol) * (
                1.0 - math.exp(-dt / c.battery_charge_pol_tau_s))
            self._soc = float(np.clip(self._soc - i_batt * dt / (fs["capacity_ah"] * 3600.0), 0.0, 1.0))
        self._advance_load(time_s)
        self._t = time_s

        ocv = self.ocv(self._soc)
        i_batt = self._load - self._i_alt
        v_bus = ocv + self._v_pol - i_batt * fs["r_int"]
        burst, f_rip = self._burst(engine_rpm, v_bus, available, fs["open_diode"], enable_noise)
        return ElectricalTruth(
            bus_voltage_v=v_bus, alternator_current_a=self._i_alt, battery_current_a=i_batt,
            load_current_a=self._load, battery_ocv_v=ocv, battery_soc=self._soc,
            battery_r_int_ohm=fs["r_int"], regulator_setpoint_v=fs["setpoint"],
            alternator_available=available, ripple_frequency_hz=f_rip,
            burst_v=burst, burst_fs_hz=c.bus_v_burst_fs_hz,
        )

    def _burst(self, engine_rpm: float, v_bus: float, available: bool, open_diode: bool,
               enable_noise: bool) -> tuple[np.ndarray, float | None]:
        c = self.cfg
        n, fs = c.bus_v_burst_samples, c.bus_v_burst_fs_hz
        t = np.arange(n) / fs
        v = np.full(n, v_bus)
        f_rip = None
        if available:
            f_e = (c.generator_poles_placeholder / 2.0) * self.generator_rpm(engine_rpm) / 60.0
            phase0 = self._rng.uniform(0.0, 2.0 * math.pi)
            phases = [np.sin(2.0 * math.pi * f_e * t + phase0 - 2.0 * math.pi * k / c.phases)
                      for k in range(c.phases)]
            upper = phases[1:] if open_diode else phases
            rect = np.max(upper, axis=0) - np.min(phases, axis=0)
            frac = (rect - rect.mean()) / rect.mean()
            v = v + c.ripple_filter_gain * v_bus * frac
            f_rip = self.ripple_fundamental_hz(engine_rpm, open_diode)
        if enable_noise:
            v = v + self._rng.normal(0.0, c.bus_v_noise_v, n)
        return v, f_rip
