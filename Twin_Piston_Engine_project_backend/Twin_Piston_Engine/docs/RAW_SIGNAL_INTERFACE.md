# Raw signal interface (L1 → L2)

`RawSignalRecord` (`src/l1_data/raw_signal_record.py`) is the only artifact that
crosses from L1 to L2 (SRD-CON-001). This table lists every raw data field, its
unit, rate, transport, and the L2 channels derived from it
(`sensor_inverse.RAW_FIELD_DEPENDENTS`). Fields added in Prompt 11 are Optional:
`None` means "not instrumented", and every L2 channel that depends on a `None`
field is `valid=False`.

Core fields are **required but nullable**: every record states each one, and
`None` means "no data" (e.g. the CAN frame carrying it has not arrived, or the
source does not carry it). Every L2 channel derived from a `None` field is
`valid=False`; nothing is substituted.

Record rate: 1 Hz (`telemetry.sample_rate_hz`). Where a signal is sampled faster,
the record carries the latest value (or a burst) at the stated rate.

## Existing fields

| Field | Unit | Rate | CAN | Other transport | L2 channels |
|---|---|---|---|---|---|
| egt_cyl1..4_hot_uv | µV | 1 Hz | 0x101 (µV + 10 000 offset, u16 ×4) | API, edge, CSV, repository | egt_cyl_1..4 |
| egt_cold_c | °C † | 1 Hz | — (None from CAN) | API, edge, CSV, repository | egt_cyl_1..4 |
| cht_hot_uv | µV | 1 Hz | 0x102 (µV + 10 000 offset, u16) | all | cht_cyl_1..4 |
| cht_cold_c | °C † | 1 Hz | — (None from CAN) | all | cht_cyl_1..4 |
| oil_rtd_ohms | Ω | 1 Hz | 0x102 (Ω×10, u16) | all | oil_temp |
| oil_p_counts | ADC counts | 1 Hz | 0x102 (u16) | all | oil_pressure |
| map_counts | ADC counts | 1 Hz | 0x102 (u16) | all | map_pressure, boost_pressure |
| adc_vref_counts | ADC counts | 1 Hz | — | all | map_pressure, boost_pressure, oil_pressure |
| crank_period_us | µs | 1 Hz | 0x100 (u32) | all | rpm, propeller_speed |
| fuel_pulse_hz | Hz | 1 Hz | 0x100 (Hz×10, u16) | all | fuel_flow |
| accel_counts_xyz | counts | 1 Hz | — | all | vibration_x/y/z/rms |
| accel_burst_counts_x/y/z | counts | 2048 Hz burst | — (bulk only) | API, edge, repository | vibration_burst_x/y/z |
| accel_burst_fs_hz | Hz | per record | — | API, edge, repository | vibration_burst_x/y/z |
| crank_period_burst_us | µs | 32 revolutions | — (bulk only) | API, edge, repository | crank_period_burst |
| ambient_temp_c | °C † | 1 Hz | — (None from CAN) | all | ambient_temp |
| ambient_press_pa | Pa † | 1 Hz | — (None from CAN) | all | ambient_pressure, boost_pressure, altitude |

† **Accepted SRD-CON-001 exception.** These four fields come from digital smart
sensors whose raw output already is °C or Pa: the thermocouple cold-junction
temperature from a MAX31855-type thermocouple interface chip (internal
cold-junction sensor, digital °C), and ambient pressure/temperature from an
MS5611-type digital barometer (compensated Pa and °C over SPI/I²C). There is no
lower-level electrical quantity available at the interface, so the value is
passed through as the sensor reports it. VERIFY against the actual hardware
selection; if an analogue part is chosen instead, carry its raw quantity
(ohms / counts) and convert in L2. See OI-18.

## Prompt 11 fields (Optional, default None)

| Field | Unit | Rate | CAN | Other transport | L2 channels (derivation) |
|---|---|---|---|---|---|
| bus_v_counts | ADC counts (via divider, 20 V full scale) | 10 Hz | 0x103 word 1 (u16) | API, edge, CSV, repository | voltage |
| alt_i_counts | ADC counts (unidirectional Hall, 0–50 A full scale) | 10 Hz | 0x103 word 2 (u16) | API, edge, CSV, repository | current (alternator output) |
| batt_i_counts | ADC counts (bidirectional Hall, ±50 A; zero at 0.5 × ADC reference; positive = discharge) — added in Prompt 12 | 10 Hz | 0x103 word 3 (u16) | API, edge, CSV, repository | battery_current |
| fuel_press_counts | ADC counts (gauge sensor, 1 MPa full scale; VERIFY) | 10 Hz | 0x103 word 0 (u16) | API, edge, CSV, repository | fuel_pressure [Pa gauge] |
| coolant_ntc_ohms | Ω | 1 Hz | 0x107 (Ω×10, u32) | API, edge, CSV, repository | coolant_temp [K] (Steinhart-Hart; unavailable if engine.has_coolant_circuit = False) |
| inj_pw_us | µs, ×4 cylinders | per cycle, reported at 10 Hz | 0x104 (u16 µs ×4) | API, edge, CSV (`inj_pw_us_1..4`), repository | injector_pulse_width_cyl_1..4 [s] |
| inj_soi_delay_us | µs, ×4 (TDC-ref edge → injector open) | per cycle, reported at 10 Hz | 0x105 (u16 µs ×4) | API, edge, CSV (`inj_soi_delay_us_1..4`), repository | injection_timing_cyl_1..4 [s], injection_soi_deg_cyl_1..4 [deg after TDC ref] |
| ign_delay_us | µs, ×4 (coil fire edge → TDC-ref edge; negative = after TDC) | per cycle, reported at 10 Hz | 0x106 (int16 µs ×4, signed) | API, edge, CSV (`ign_delay_us_1..4`), repository | ignition_timing_cyl_1..4 [deg BTDC] |
| bus_v_burst_counts | ADC counts | burst, 2048 samples at 10 240 Hz (simulator; the rate travels in bus_v_burst_fs_hz) | — (bulk only) | API, edge, repository | bus_voltage_burst (ripple) |
| bus_v_burst_fs_hz | Hz | per record | — | API, edge, repository | bus_voltage_burst (M-05 check) |

Electrical (Prompt 12): L2 converts the counts ratiometrically to the 12-bit
reference (4095) using `sensor_calibration.*_full_scale_*`. Each channel is
valid=False ("not instrumented") when its field is None, and every quantity
derived from it (charging residual, R_int, ripple, EHI) is then invalid too.
The bus-voltage burst must satisfy M-05. The simulator's ripple fundamental is
2 × phases × f_e, with f_e = poles/2 × generator rpm/60 and the SIMULATION
placeholder of 12 poles: 1200 / 1800 / 2400 / 3000 / 3480 Hz at 2000 / 3000 /
4000 / 5000 / 5800 rpm. That is below the 5120 Hz Nyquist limit of the
10 240 Hz burst, and fs ≥ 2.5 × 3480 Hz = 8700 Hz. The L2 analysis band
(50–4000 Hz) is checked against Nyquist at config load and again per burst.

Injection, ignition and fuel rail (Prompt 13). L2 converts each interval to crank
angle with the same record's rpm (from crank_period_us):
deg = delay_us × rpm × 6e-6. For example, 1000 µs at 5000 rpm is 30.0°. The
angle channels are therefore invalid when rpm is invalid. The intervals are
kept as well, in seconds. The fuel pressure is a gauge sensor. L2 forms the
rail differential as p_gauge + p_ambient − MAP, which assumes a manifold-
referenced regulator (VERIFY). Each channel is valid=False ("not instrumented")
when its raw field is None, and everything derived from it (commanded fuel,
duty, rail residual, delivery ratio, injector flow ratio, ignition residual)
is invalid too.

Coolant (Prompt 14). L2 converts coolant_ntc_ohms to temperature with the
Steinhart-Hart equation, 1/T = A + B ln R + C (ln R)^3, using the coefficients
in `sensor_calibration.ntc_sh_*`. The default coefficients are the widely
published example for a 10 kΩ (25 °C) NTC, not the Rotax sensor (VERIFY). When
`engine.has_coolant_circuit` is False, the coolant channel and every coolant
parameter are unavailable, and an NTC value is not interpreted.

Injection and ignition are time intervals against the crank TDC reference
(e.g. STM32 timer input capture), the same class of raw signal as
`crank_period_us`.

## CAN frames (CAN 2.0, big-endian, 8 data bytes)

| ID | Content | Scaling | Limits (decoded as invalid) |
|---|---|---|---|
| 0x100 | crank_period_us, fuel_pulse_hz | u32 µs; u16 Hz×10; 2 bytes reserved | crank 0 or 0xFFFFFFFF |
| 0x101 | egt_cyl1..4_hot_uv | u16 ×4, wire = µV + 10 000 (1 µV/count): −10 000 … 55 535 µV | wire 0 or 65535 |
| 0x102 | cht_hot_uv, oil_rtd_ohms, map_counts, oil_p_counts | u16 (µV + 10 000), u16 Ω×10, u16, u16 | CHT wire 0 or 65535 |
| 0x103 | fuel_press_counts, bus_v_counts, alt_i_counts, batt_i_counts | u16 ×4 (word 3 was the spare until Prompt 12) | — |
| 0x104 | inj_pw_us cyl 1–4 | u16 µs ×4 | 0xFFFF |
| 0x105 | inj_soi_delay_us cyl 1–4 | u16 µs ×4 | 0xFFFF |
| 0x106 | ign_delay_us cyl 1–4 | int16 µs ×4 (signed: negative = coil fire after TDC ref) | −32768 or 32767 |
| 0x107 | coolant_ntc_ohms | u32 Ω×10 (0 … 429 496 729.4 Ω) + 4 bytes reserved | 0 (short circuit) or 0xFFFFFFFF |

Live adapter (`LiveTelemetryAdapter`): every field starts as `None`. Until the
first frame of a group arrives its fields are `None` and flagged "No CAN frame
0x1xx received yet"; fields no frame carries (cold junctions, ADC reference,
accelerometer, ambient) are `None` and flagged "Not carried on the CAN bus".
No placeholder values are emitted.

Bursts (accelerometer, bus voltage) are too large for classic CAN and travel
only on the bulk path: API ingest, edge HTTP transport and the repository.
CSV replay cannot carry bursts (OI-3).

## Integrity and signature

Bursts and the Prompt 11 fields enter the SHA-256 integrity hash and the HMAC
payload (SRD-SEC-001) only when present (`RawSignalRecord.burst_payload`), so
records without them keep their original hash and signature.

## ECU cross-check values (DS-12)

The Rotax 915 iS ECU CAN parameter list (DS-12 in the source register) is not
available in this repository, so it is not known which of these quantities the
ECU broadcasts. If it does, they must be ingested as SEPARATE cross-check fields
tagged with their source (e.g. `ecu_*`), never written into the raw fields above:
ECU-computed engineering values in the raw record would break SRD-CON-001.
