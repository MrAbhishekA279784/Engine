"""
Coolant-circuit diagnosis (L2; Prompt 14).

    coolant_temp_c      from the NTC (sensor_inverse, Steinhart-Hart)
    coolant residual    coolant - expected coolant (operating point only, M-11 style)
    coolant_cht_delta   CHT - coolant: rises when head-to-coolant heat transfer
                        degrades (coolant loss, flow loss, deposits)

Without a liquid coolant circuit (engine.has_coolant_circuit = False) every
parameter is unavailable with that reason (SRD-FUN-045 pattern), never inferred.
The status flags overheating (positive residuals); a coolant running cooler
than expected is reported as evidence only (e.g. thermostat stuck open).
"""

from __future__ import annotations

from collections import deque
from statistics import median

from src.core.config import AppSettings, get_settings
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import CoolantState, ProvenanceTaggedValue, make_tagged
from src.l1_data.signal_record import NormalizedSignalRecord
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, operating_point_from_record

NO_CIRCUIT = "No liquid coolant circuit (engine.has_coolant_circuit = False): coolant parameters unavailable"


def _invalid(reason: str) -> ProvenanceTaggedValue[float | None]:
    return ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False,
                                 validity_reason=ChannelValidity.MISSING, quality=0.0, fault_flag=reason)


def _valid(v: float, q: float = 1.0) -> ProvenanceTaggedValue[float | None]:
    return make_tagged(v, Provenance.DERIVED, valid=True, quality=q)


def _positive_band(x: float | None, warning: float, alarm: float) -> DiagnosticStatus:
    if x is None:
        return DiagnosticStatus.INVALID
    if x >= alarm:
        return DiagnosticStatus.CRITICAL
    if x >= warning:
        return DiagnosticStatus.WARNING
    return DiagnosticStatus.NORMAL


class CoolantModel:
    """Stateful (running median of the coolant residual)."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.coolant
        self.expectations = HealthyExpectationModel(self._settings)
        self._res_hist: deque[float] = deque(maxlen=self.cfg.median_n)

    def reset_state(self) -> None:
        self._res_hist.clear()

    def evaluate(self, record: NormalizedSignalRecord) -> CoolantState:
        c = self.cfg
        if not self._settings.engine.has_coolant_circuit:
            tv = _invalid(NO_CIRCUIT)
            return CoolantState(timestamp=record.timestamp, coolant_temp_c=tv, expected_coolant_c=tv,
                                coolant_residual_k=tv, coolant_residual_median_k=tv, coolant_cht_delta_k=tv,
                                expected_cht_delta_k=tv, cht_delta_residual_k=tv, evidence=[NO_CIRCUIT])
        evidence: list[str] = []
        op = operating_point_from_record(record)
        op_ok = all(ch.valid for ch in (record.rpm, record.map_pressure, record.ambient_pressure, record.ambient_temp))
        cool, cht = record.coolant_temp, record.cht_cyl_1

        cool_tv = _valid(cool.value - 273.15, cool.quality) if cool.valid else _invalid(
            cool.fault_flag or "coolant temperature invalid")
        if op_ok:
            exp_cool = self.expectations.expected_coolant_temp_k(op)
            exp_tv = _valid(exp_cool - 273.15)
        else:
            exp_cool = None
            exp_tv = _invalid("operating point invalid")

        res_med = None
        if cool.valid and exp_cool is not None:
            res = cool.value - exp_cool
            self._res_hist.append(res)
            res_med = median(self._res_hist)
            res_tv, res_med_tv = _valid(res, cool.quality), _valid(res_med)
        else:
            res_tv = res_med_tv = _invalid(cool.fault_flag if not cool.valid else "operating point invalid")

        delta_res = None
        if cool.valid and cht.valid:
            delta = cht.value - cool.value
            delta_tv = _valid(delta, min(cool.quality, cht.quality))
            if op_ok:
                exp_delta = self.expectations.expected_cht_k(op) - exp_cool
                delta_res = delta - exp_delta
                exp_delta_tv, delta_res_tv = _valid(exp_delta), _valid(delta_res)
            else:
                exp_delta_tv = delta_res_tv = _invalid("operating point invalid")
        else:
            reason = "CHT or coolant temperature invalid"
            delta_tv = exp_delta_tv = delta_res_tv = _invalid(reason)

        s_res = _positive_band(res_med, c.residual_warning_k, c.residual_alarm_k)
        s_delta = _positive_band(delta_res, c.cht_delta_warning_k, c.cht_delta_alarm_k)
        valid = [x for x in (s_res, s_delta) if x != DiagnosticStatus.INVALID]
        rank = {DiagnosticStatus.NORMAL: 1, DiagnosticStatus.WARNING: 2, DiagnosticStatus.CRITICAL: 3}
        status = max(valid, key=rank.get) if valid else DiagnosticStatus.INVALID
        if s_res in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
            evidence.append(f"coolant {res_med:+.1f} K above expectation for this operating point")
        if s_delta in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
            evidence.append(f"CHT - coolant {delta_res:+.1f} K above expectation: head-to-coolant heat "
                            "transfer degraded (coolant loss, flow loss or deposits)")
        if res_med is not None and res_med <= -c.residual_alarm_k:
            evidence.append(f"coolant {res_med:+.1f} K below expectation (thermostat stuck open or sensor)")

        return CoolantState(
            timestamp=record.timestamp,
            coolant_temp_c=cool_tv,
            expected_coolant_c=exp_tv,
            coolant_residual_k=res_tv,
            coolant_residual_median_k=res_med_tv,
            coolant_cht_delta_k=delta_tv,
            expected_cht_delta_k=exp_delta_tv,
            cht_delta_residual_k=delta_res_tv,
            status=status,
            evidence=evidence,
        )
