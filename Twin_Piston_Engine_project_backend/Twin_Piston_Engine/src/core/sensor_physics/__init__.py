"""
Core — Shared sensor transfer-function physics.

Standards-based sensor characteristics used on both sides of the L1/L2
boundary: the L1 simulator forward model (temperature -> raw uV / ohms) and
the L2 inverse model (raw uV / ohms -> temperature). Living in core keeps one
source of truth for each curve without L1 importing from L2.

    D01_thermocouple_nist_inverse  Type K thermocouple, NIST ITS-90
    D02_rtd_callendar_van_dusen    Pt RTD, IEC 60751 Callendar-Van Dusen
    D03_ntc_steinhart_hart         NTC thermistor, Steinhart-Hart (coolant)
    M05_nyquist_guard              Nyquist checks for vibration analysis bands;
                                   used by the settings loader, which L1 also
                                   calls, so it cannot live in L2.
"""

from src.core.sensor_physics.D01_thermocouple_nist_inverse import (
    TYPE_K_TEMP_MAX_C,
    TYPE_K_TEMP_MIN_C,
    thermocouple_k_to_celsius,
    thermocouple_k_to_kelvin,
    type_k_emf_uv,
    type_k_temp_c,
)
from src.core.sensor_physics.D02_rtd_callendar_van_dusen import (
    pt100_resistance_ohms,
    pt100_temp_c,
    pt100_temp_kelvin,
)

from src.core.sensor_physics.D03_ntc_steinhart_hart import (
    ntc_resistance_ohm,
    ntc_temp_c,
    ntc_temp_k,
)

__all__ = [
    "ntc_resistance_ohm",
    "ntc_temp_c",
    "ntc_temp_k",
    "TYPE_K_TEMP_MAX_C",
    "TYPE_K_TEMP_MIN_C",
    "pt100_resistance_ohms",
    "pt100_temp_c",
    "pt100_temp_kelvin",
    "thermocouple_k_to_celsius",
    "thermocouple_k_to_kelvin",
    "type_k_emf_uv",
    "type_k_temp_c",
]
