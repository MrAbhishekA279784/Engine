"""
Simulator ECU injection/ignition, fuel rail and per-cylinder fuel delivery
(L1, simulation only; Prompt 13).

ECU (control schedules, not engine properties):
    target fuel per cycle per cylinder  m_t = m_dot_air / (AFR_st λ_cmd) / (n_cyl rpm / 120)
    pulse width                         PW  = t_dead + m_t / (K_nom sqrt(dP_ecu / dP_ref))
        with dP_ecu = dP_ref: the ECU assumes the regulated rail (no rail-
        pressure compensation; VERIFY for the 915 iS ECU)
    start of injection, ignition advance: (rpm, MAP) maps in config;
    knock control retards the knocking cylinder(s).

Fuel rail: a pump Q = Q_max h (1 - p_gauge / p_stall) feeds a manifold-
referenced regulator that returns the excess while it can hold dP_ref. When the
pump cannot supply the injectors at dP_ref, the rail settles where pump flow
equals injector flow: Σ K_i sqrt(dP/dP_ref) (PW - t_dead) rpm/120.

Delivered fuel per cylinder:  m_i = K_i sqrt(dP/dP_ref) (PW - t_dead).
The cylinder burns what it receives, so its λ_i = λ_cmd m_t / m_i, and its
exhaust temperature rise follows the energy balance
    ΔT_exh ∝ x(λ, retard) η_c(λ) f / (1 + f),   f = 1/(AFR_st λ)
(x = fraction of the released heat that leaves with the exhaust). A degraded
injector therefore changes that cylinder's EGT through the cycle, not as an
offset (SRD-FUN-152).

Faults: INJECTOR_FAULT (K_i of the affected cylinder: sub_mode "clog" lowers
it, "leak" raises it, by the severity fraction), FUEL_SYSTEM_FAULT (pump
health h = 1 - severity x capacity loss), DETONATION_KNOCK (ECU knock retard on
the affected cylinders).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.core.config import AppSettings, get_settings
from src.core.units import interp2d_clamped
from src.l1_data.simulator.fault_injection import FaultMode

INJECTOR_SUB_MODES = ("clog", "leak")


@dataclass
class InjectionTruth:
    """Injection / fuel-system ground truth at one record (simulation only)."""

    pulse_width_us: float
    soi_deg: float
    advance_deg: list[float]
    retard_deg: list[float]
    rail_dp_pa: float
    rail_gauge_pa: float
    pump_limited: bool
    flow_factor: list[float]           # K_i / K_nom
    fuel_factor: list[float]           # delivered / ECU target, per cylinder
    lambda_cyl: list[float]
    target_fuel_per_cycle_kg: float
    egt_rise_factor: list[float] = field(default_factory=lambda: [1.0] * 4)
    imep_factor: float = 1.0


class InjectionSimulator:
    """Stateless per-record injection and fuel-rail model."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.simulator.injection
        self._phys = self._settings.simulator.physics
        self._n_cyl = self._settings.engine.num_cylinders

    # ------------------------------------------------------------------ schedules
    def soi_deg(self, rpm: float, map_pa: float) -> float:
        c = self.cfg
        return interp2d_clamped(c.soi_rpm_axis, c.soi_map_axis_pa, c.soi_table_deg, rpm, map_pa)

    def scheduled_advance_deg(self, rpm: float, map_pa: float) -> float:
        c = self.cfg
        return interp2d_clamped(c.ign_rpm_axis, c.ign_map_axis_pa, c.ign_table_deg, rpm, map_pa)

    # ------------------------------------------------------------------ combustion energy balance
    def exhaust_fraction(self, lam: float, retard_deg: float = 0.0) -> float:
        c = self.cfg
        if lam <= 1.0:
            x = c.egt_x_stoich - c.egt_x_rich_slope * (1.0 - lam)
        else:
            x = c.egt_x_stoich + c.egt_x_lean_gain * (1.0 - math.exp(-(lam - 1.0) / c.egt_x_lean_scale))
        return x * (1.0 + c.egt_x_per_deg_retard * retard_deg)

    def exhaust_energy(self, lam: float, retard_deg: float = 0.0) -> float:
        """Exhaust heat per unit charge mass, up to a constant: x η_c f / (1 + f)."""
        f = 1.0 / (self._phys.stoichiometric_afr * lam)
        eta_c = self._phys.combustion_efficiency_max * min(1.0, lam)
        return self.exhaust_fraction(lam, retard_deg) * eta_c * f / (1.0 + f)

    # ------------------------------------------------------------------ rail
    def _rail_dp(self, demand_at_ref_kg_s: float, map_pa: float, ambient_pa: float, pump_health: float) -> tuple[float, bool]:
        c = self.cfg

        def pump(dp: float) -> float:
            gauge = dp + map_pa - ambient_pa
            return c.pump_max_flow_kg_s * pump_health * max(0.0, 1.0 - gauge / c.pump_stall_gauge_pa)

        def demand(dp: float) -> float:
            return demand_at_ref_kg_s * math.sqrt(max(dp, 0.0) / c.rail_dp_ref_pa)

        if pump(c.rail_dp_ref_pa) >= demand(c.rail_dp_ref_pa):
            return c.rail_dp_ref_pa, False
        lo, hi = 0.0, c.rail_dp_ref_pa
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if pump(mid) >= demand(mid):
                lo = mid
            else:
                hi = mid
        return lo, True

    # ------------------------------------------------------------------ step
    def solve(self, rpm: float, map_pa: float, ambient_pa: float, air_kg_s: float, lambda_cmd: float,
              fault_mode: int | None = None, sub_mode: str | None = None, severity: float = 0.0,
              cylinders: list[int] | None = None) -> InjectionTruth:
        c = self.cfg
        sev = min(max(severity, 0.0), 1.0)
        cyls = [i for i in (cylinders or []) if 1 <= i <= self._n_cyl]
        mode = int(fault_mode) if fault_mode is not None else None

        k = [1.0] * self._n_cyl
        retard = [0.0] * self._n_cyl
        pump_health = 1.0
        if mode == FaultMode.INJECTOR_FAULT:
            sm = sub_mode or "clog"
            if sm not in INJECTOR_SUB_MODES:
                raise ValueError(f"unknown INJECTOR_FAULT sub_mode {sm!r}; expected one of {INJECTOR_SUB_MODES}")
            for i in cyls:
                k[i - 1] = 1.0 - sev if sm == "clog" else 1.0 + sev
        elif mode == FaultMode.FUEL_SYSTEM_FAULT:
            pump_health = max(0.0, 1.0 - sev * c.fuel_pump_capacity_loss)
        elif mode == FaultMode.DETONATION_KNOCK:
            for i in cyls:
                retard[i - 1] = sev * c.knock_max_retard_deg

        cycles_per_s = rpm / 120.0
        m_target = air_kg_s / (self._phys.stoichiometric_afr * lambda_cmd) / (self._n_cyl * cycles_per_s)
        demand_at_ref = sum(k) * m_target * cycles_per_s  # injector flow at dP_ref with the ECU's PW
        dp, limited = self._rail_dp(demand_at_ref, map_pa, ambient_pa, pump_health)
        # ECU pulse width for the reference rail pressure (no compensation)
        dp_factor = 1.0 if dp == c.rail_dp_ref_pa else math.sqrt(dp / c.rail_dp_ref_pa)
        pw_us = c.injector_dead_time_us + m_target / c.injector_static_flow_kg_s * 1e6

        fuel_factor = [ki * dp_factor for ki in k]
        lam = [lambda_cmd / ff if ff > 0.0 else math.inf for ff in fuel_factor]
        advance_sched = self.scheduled_advance_deg(rpm, map_pa)
        healthy = all(ff == 1.0 for ff in fuel_factor) and not any(retard)
        if healthy:
            egt_factor = [1.0] * self._n_cyl
            imep_factor = 1.0
        else:
            e_ref = self.exhaust_energy(lambda_cmd)
            egt_factor = [self.exhaust_energy(l_i, r_i) / e_ref if math.isfinite(l_i) else 0.0
                          for l_i, r_i in zip(lam, retard)]
            eta_ref = min(1.0, lambda_cmd)
            imep_factor = sum(min(1.0, l_i) * ff * (1.0 - c.imep_loss_per_deg_retard * r_i)
                              for l_i, ff, r_i in zip(lam, fuel_factor, retard)
                              if math.isfinite(l_i)) / (self._n_cyl * eta_ref)
        return InjectionTruth(
            pulse_width_us=pw_us,
            soi_deg=self.soi_deg(rpm, map_pa),
            advance_deg=[advance_sched - r for r in retard],
            retard_deg=retard,
            rail_dp_pa=dp,
            rail_gauge_pa=dp + map_pa - ambient_pa,
            pump_limited=limited,
            flow_factor=k,
            fuel_factor=fuel_factor,
            lambda_cyl=lam,
            target_fuel_per_cycle_kg=m_target,
            egt_rise_factor=egt_factor,
            imep_factor=imep_factor,
        )
