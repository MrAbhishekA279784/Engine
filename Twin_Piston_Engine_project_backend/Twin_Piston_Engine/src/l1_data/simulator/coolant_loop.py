"""
Simulator liquid-coolant loop (L1, simulation only; Prompt 14).

    heat from the heads   Q  = head_heat_fraction x fuel power
    pump                  m_c = m_ref x rpm / rpm_ref x pump health
    thermostat            θ(T_c) = min leak .. 1 between open and fully open
    radiator (crossflow, both fluids unmixed; Incropera eq. 11.33 form)
        C_air = ρ_amb v A_face f_duct c_p,air x (1 - blockage)
        UA    = UA_ref (v / v_ref)^0.8 x (1 - blockage)^0.8
        ε     = 1 - exp[(NTU^0.22 / C_r)(exp(-C_r NTU^0.78) - 1)]
        Q_rad = θ ε C_min (T_c - T_amb)
    coolant               (m_cool c_p + C_head) dT_c/dt = Q - Q_rad
    head                  CHT_loop = T_c + Q / hA_head,  hA ∝ (m_c / m_ref)^0.8

The simulator's CHT map (ForwardPhysicsModel) is the healthy steady state at
the reference ambient and airspeed. The loop adds the departure of
CHT_loop from its healthy reference steady state at the same operating point,
so ambient, airspeed, faults and thermal lag change CHT through the loop
while a steady healthy reference run is unchanged.

COOLING_FAULT sub-modes (physical parameters, SRD-FUN-152):
    radiator_blockage  air path and UA reduced
    pump_degradation   coolant flow reduced
    coolant_loss       coolant mass and head-to-coolant hA reduced
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.config import AppSettings, get_settings

COOLING_SUB_MODES = ("radiator_blockage", "pump_degradation", "coolant_loss")
CP_AIR = 1005.0
R_AIR = 287.05


@dataclass
class CoolantTruth:
    coolant_temp_k: float
    coolant_ref_ss_k: float        # healthy reference steady state, same operating point
    cht_loop_delta_k: float        # CHT_loop - healthy reference CHT_loop
    head_heat_w: float
    radiator_conductance_w_k: float
    thermostat_open: float
    coolant_flow_kg_s: float
    fault_severity: float


@dataclass
class _Params:
    blockage: float = 0.0
    pump: float = 1.0
    mass: float = 1.0
    head_ha: float = 1.0


class CoolantLoop:
    """Stateful coolant loop. step() is called with the simulation time;
    a first call, or time going backwards, starts at the steady state."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.simulator.cooling
        self._t: float | None = None
        self._temp_k: float | None = None

    # ------------------------------------------------------------------ components
    def _fault_params(self, sub_mode: str | None, severity: float) -> _Params:
        c = self.cfg
        p = _Params()
        if severity <= 0.0:
            return p
        mode = sub_mode or "radiator_blockage"
        if mode == "radiator_blockage":
            p.blockage = severity * c.blockage_max
        elif mode == "pump_degradation":
            p.pump = 1.0 - severity * c.pump_loss_max
        elif mode == "coolant_loss":
            p.mass = 1.0 - severity * c.coolant_loss_max
            p.head_ha = 1.0 - severity * c.coolant_loss_head_ha_factor
        else:
            raise ValueError(f"unknown COOLING_FAULT sub_mode {mode!r}; expected one of {COOLING_SUB_MODES}")
        return p

    def coolant_flow(self, rpm: float, p: _Params) -> float:
        c = self.cfg
        return max(c.coolant_flow_ref_kg_s * rpm / c.pump_ref_rpm * p.pump, 1e-6)

    def radiator_conductance(self, rpm: float, ambient_k: float, ambient_pa: float, airspeed: float,
                             p: _Params) -> float:
        """ε C_min [W/K] with the thermostat fully open."""
        c = self.cfg
        open_area = max(1.0 - p.blockage, 1e-6)
        rho = ambient_pa / (R_AIR * ambient_k)
        c_air = rho * max(airspeed, 0.1) * c.radiator_face_area_m2 * c.duct_flow_fraction * CP_AIR * open_area
        c_cool = self.coolant_flow(rpm, p) * c.coolant_cp_j_kg_k
        c_min, c_max = min(c_air, c_cool), max(c_air, c_cool)
        cr = c_min / c_max
        ua = c.radiator_ua_ref_w_k * (max(airspeed, 0.1) / c.airspeed_ref_m_s) ** 0.8 * open_area ** 0.8
        ntu = ua / c_min
        eps = 1.0 - math.exp((ntu ** 0.22 / cr) * (math.exp(-cr * ntu ** 0.78) - 1.0))
        return eps * c_min

    def thermostat(self, temp_k: float) -> float:
        c = self.cfg
        x = (temp_k - 273.15 - c.thermostat_open_c) / (c.thermostat_full_open_c - c.thermostat_open_c)
        return c.thermostat_min_open + (1.0 - c.thermostat_min_open) * min(max(x, 0.0), 1.0)

    def head_ha(self, rpm: float, p: _Params) -> float:
        c = self.cfg
        return c.head_ha_ref_w_k * (self.coolant_flow(rpm, p) / c.coolant_flow_ref_kg_s) ** 0.8 * p.head_ha

    def steady_state_k(self, q_w: float, g_w_k: float, ambient_k: float) -> float:
        """T_c with θ(T_c) G (T_c - T_amb) = Q (monotonic in T_c)."""
        lo, hi = ambient_k, ambient_k + 2000.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.thermostat(mid) * g_w_k * (mid - ambient_k) < q_w:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ step
    def step(self, time_s: float, rpm: float, fuel_power_w: float, ambient_k: float, ambient_pa: float,
             airspeed_m_s: float | None = None, sub_mode: str | None = None, severity: float = 0.0,
             onset_s: float = 0.0) -> CoolantTruth:
        c = self.cfg
        v = c.airspeed_ref_m_s if airspeed_m_s is None else airspeed_m_s
        sev = min(max(severity, 0.0), 1.0)
        if sev > 0.0 and c.fault_ramp_s > 0.0:
            sev *= min(1.0, max(0.0, (time_s - onset_s) / c.fault_ramp_s))
        p = self._fault_params(sub_mode, sev)
        q = c.head_heat_fraction * fuel_power_w
        g = self.radiator_conductance(rpm, ambient_k, ambient_pa, v, p)

        # healthy reference at the reference ambient and airspeed, same operating point
        p0 = _Params()
        g_ref = self.radiator_conductance(rpm, c.reference_ambient_k, 101325.0, c.airspeed_ref_m_s, p0)
        t_ref = self.steady_state_k(q, g_ref, c.reference_ambient_k)
        cht_ref = t_ref + q / self.head_ha(rpm, p0)

        if self._temp_k is None or self._t is None or time_s <= self._t:
            self._temp_k = self.steady_state_k(q, g, ambient_k)
        else:
            cap = c.coolant_mass_kg * p.mass * c.coolant_cp_j_kg_k + c.head_heat_capacity_j_k
            n = max(1, int(math.ceil((time_s - self._t) / c.integration_step_s)))
            dt = (time_s - self._t) / n
            temp = self._temp_k
            for _ in range(n):
                temp += dt * (q - self.thermostat(temp) * g * (temp - ambient_k)) / cap
            self._temp_k = temp
        self._t = time_s
        cht_loop = self._temp_k + q / self.head_ha(rpm, p)
        return CoolantTruth(
            coolant_temp_k=self._temp_k,
            coolant_ref_ss_k=t_ref,
            cht_loop_delta_k=cht_loop - cht_ref,
            head_heat_w=q,
            radiator_conductance_w_k=g,
            thermostat_open=self.thermostat(self._temp_k),
            coolant_flow_kg_s=self.coolant_flow(rpm, p),
            fault_severity=sev,
        )
