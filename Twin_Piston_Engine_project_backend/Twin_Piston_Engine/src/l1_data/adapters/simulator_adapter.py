"""
Physics Simulator Adapter — Wraps the physics simulator as an L1 data source adapter.

Converts the RotaxEngineSimulator ground truth state through the sensor-forward
path into canonical RawSignalRecord instances tagged with Provenance.SIMULATED.
"""

from __future__ import annotations

import numpy as np

from src.core.config import get_settings
from src.core.provenance import FlightPhase, Provenance
from src.l1_data.adapters.base import SourceAdapter
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.sensor_forward import normalized_to_raw
from src.l1_data.simulator.engine_model import EngineOperatingPoint, RotaxEngineSimulator
from src.l1_data.simulator.fault_injection import FaultScenario
from src.l1_data.simulator.vibration_burst import (
    BURST_FS_HZ,
    accel_to_counts,
    generate_accel_burst,
    generate_crank_burst,
)


class PhysicsSimulatorAdapter(SourceAdapter):
    """Source adapter wrapping the Rotax 915 iS simulator.

    Usage:
        async with PhysicsSimulatorAdapter() as adapter:
            raw_record = await adapter.read_next()
    """

    def __init__(
        self,
        rpm: float = 4000.0,
        throttle_pct: float = 50.0,
        altitude_m: float = 1000.0,
        flight_phase: FlightPhase = FlightPhase.CRUISE,
        fault_scenario: FaultScenario | None = None,
        sample_rate_hz: float = 1.0,
    ) -> None:
        self._simulator = RotaxEngineSimulator()
        self._operating_point = EngineOperatingPoint(
            rpm=rpm,
            throttle_pct=throttle_pct,
            altitude_m=altitude_m,
            flight_phase=flight_phase,
        )
        self._fault_scenario = fault_scenario
        self._sample_rate_hz = sample_rate_hz
        self._connected = False
        self._engine_hours = 0.0
        self._counts_per_m_s2 = get_settings().sensor_calibration.accel_counts_per_m_s2
        self._burst_rng = np.random.default_rng(0xB0)

    async def connect(self) -> None:
        """Initialize the simulator."""
        if self._fault_scenario:
            self._simulator.configure_fault(self._fault_scenario)
        self._connected = True

    async def read_next(self) -> RawSignalRecord | None:
        """Run one simulation step and convert output to RawSignalRecord.

        Returns:
            RawSignalRecord with Provenance.SIMULATED.
        """
        if not self._connected:
            return None

        self._operating_point.engine_hours = self._engine_hours

        # Generate normalized telemetry from simulator
        norm_record = self._simulator.step(self._operating_point)

        # Advance engine hours
        self._engine_hours += 1.0 / (self._sample_rate_hz * 3600.0)

        # Pass through sensor forward model to produce canonical RawSignalRecord
        raw_record = normalized_to_raw(
            record=norm_record,
            sequence_number=norm_record.sequence_number,
            source_type=Provenance.SIMULATED,
        )

        # Attach the per-record high-rate bursts (2048 Hz accel, 32 crank revs)
        # generated from the simulator's operating point and active fault.
        scenario = self._simulator.fault_injector.scenario
        severity = self._simulator.fault_injector.current_severity(self._simulator.sim_time_s)
        rpm = self._operating_point.rpm
        bx, by, bz = generate_accel_burst(
            rpm=rpm,
            axis_rms_m_s2=(
                norm_record.vibration_x.value,
                norm_record.vibration_y.value,
                norm_record.vibration_z.value,
            ),
            fault_mode=scenario.mode,
            severity=severity,
            rng=self._burst_rng,
        )
        crank_burst = generate_crank_burst(
            rpm=rpm,
            fault_mode=scenario.mode,
            severity=severity,
            affected_cylinders=scenario.affected_cylinders,
            rng=self._burst_rng,
        )
        raw_record = raw_record.model_copy(update={
            "accel_burst_counts_x": accel_to_counts(bx, self._counts_per_m_s2),
            "accel_burst_counts_y": accel_to_counts(by, self._counts_per_m_s2),
            "accel_burst_counts_z": accel_to_counts(bz, self._counts_per_m_s2),
            "accel_burst_fs_hz": BURST_FS_HZ,
            "crank_period_burst_us": crank_burst,
        })
        return raw_record.model_copy(update={"integrity_hash": raw_record.compute_integrity_hash()})

    async def disconnect(self) -> None:
        """Clean up simulator resources."""
        self._connected = False

    @property
    def source_type(self) -> Provenance:
        return Provenance.SIMULATED

    @property
    def source_id(self) -> str:
        return "rotax_915is_simulator"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_operating_point(
        self,
        rpm: float | None = None,
        throttle_pct: float | None = None,
        altitude_m: float | None = None,
        flight_phase: FlightPhase | None = None,
    ) -> None:
        """Update the operating point for subsequent steps."""
        if rpm is not None:
            self._operating_point.rpm = rpm
        if throttle_pct is not None:
            self._operating_point.throttle_pct = throttle_pct
        if altitude_m is not None:
            self._operating_point.altitude_m = altitude_m
        if flight_phase is not None:
            self._operating_point.flight_phase = flight_phase


# Backward-compatible alias
SimulatorAdapter = PhysicsSimulatorAdapter
