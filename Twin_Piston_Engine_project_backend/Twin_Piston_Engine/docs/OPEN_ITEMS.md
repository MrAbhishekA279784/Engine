# Open items

Known issues found during the physics audit (branch `fix/physics-audit`) that
are deliberately not fixed yet. Each entry says where the problem is, why it
matters, and what closing it needs.

## OI-1 — Indicated efficiency is a fixed 42 % in the twin

- **Where:** `src/l2_digital_twin/thermo_mechanical_twin.py`, `MechanicalTwin.compute`
  (`p_i = q_in * 0.42`, both the measured-fuel and speed-density branches).
- **Effect:** twin brake power and SFC depend on an assumed constant rather than
  the operating point. On the nominal simulator the twin's SFC is
  0.213–0.297 kg/kWh (2000–5800 rpm, MAP 80–160 kPa), below the Rotax 915 iS
  published figure of about 0.312 kg/kWh, so `sfc_band` reads `BELOW_EXPECTED`
  at most points. The simulator uses its own 40 % (`simulator.physics.indicated_efficiency`,
  also VERIFY), so the two disagree by construction.
- **Needs:** the Rotax 915 iS performance deck (power, fuel flow and MAP vs rpm)
  to replace the constant with a validated indicated-efficiency map, then
  recheck `sfc_band` and `eta_mech` against it.

## OI-2 — Edge HTTP transport omits the signed timestamp

- **Where:** `src/edge_ground/transport.py`, `HttpTransport.send_telemetry` payload.
- **Effect:** the HMAC payload (`src/l1_data/telemetry_security.py`,
  `serialize_canonical_payload`) includes `record.timestamp`, but the HTTP
  payload does not send it. The ground API (`src/api/v1/telemetry.py`) then
  substitutes its own receive time, so a signature produced on the edge node
  cannot verify on the ground.
- **Needs:** send `timestamp` (ISO 8601) in the payload and accept it in
  `TelemetryIngestRequest` (the field already exists); add a transport
  round-trip signature test.

## OI-3 — CSV replay cannot carry accelerometer/crank bursts

- **Where:** `src/l1_data/adapters/csv_replay_adapter.py`.
- **Effect:** one CSV row per record cannot reasonably hold 3 × 2048
  accelerometer samples plus 32 crank periods, so replayed records have no
  `accel_burst_*` / `crank_period_burst_us`. L2 then reports every spectral
  vibration feature (RMS, crest factor, kurtosis, order fractions, envelope
  RMS) as `valid=False` for CSV replay. The live CAN adapter has the same
  limitation (no accelerometer frame on IDs 0x100–0x102).
- **Needs:** a replay format that stores bursts (e.g. per-record JSON lines or
  a sidecar binary file keyed by sequence number), or explicit acceptance that
  CSV replay is scalar-only.

## OI-4 — ML feature builder maps invalid derived values to 0.0

- **Where:** `src/l3_ml/ml_infrastructure.py`, `MLFeatureVectorBuilder`
  (`val = attr.value if (attr.valid and is_finite) else 0.0`, around lines 145–180).
- **Effect:** an invalid `eta_volumetric` (or any invalid residual/derived
  feature) enters the feature vector as 0.0. The `MLFeature.valid` flag is set
  to False, but models that read only the numeric vector see a plausible-looking
  value, which conflicts with SRD-CON-001 ("undefined is valid=False, never a
  plausible constant"). Note also that `eta_volumetric` is now the breathing
  model's expected value, not a measurement, so its value as an ML health
  feature should be reviewed.
- **Needs:** an explicit missing-value encoding (NaN plus a mask, or a
  validity-indicator feature per input) and retraining, or dropping
  `eta_volumetric` from `STANDARD_FEATURE_ORDER`.

## OI-5 — Healthy expectations disagree with the simulator; need flight data

- **Where:** `src/core/config.py`, `HealthyBaselineConfig`, used by
  `src/l2_digital_twin/residual_engine.py`, `HealthyExpectationModel`.
- **Status:** the expectations use the model defaults (EGT base 950 K,
  0.002 K/Pa, 0.05 K/rpm; M-11 CHT base 398.15 K, load slope 65 K; expected
  power reference MAP 140 kPa; fuel keyed on expected power at SFC 0.312
  kg/kWh). They are NOT calibrated to the simulator, so the residual layer
  remains an independent check of it.
- **Measured disagreement** (nominal simulator, sensor noise on, residual =
  observed − expected):

  | Operating point | CHT | Fuel | EGT cyl 1 / 2 / 3 / 4 |
  |---|---|---|---|
  | 2500 rpm, 90 kPa  | −35.7 K (−8.6 %)  | −32.7 % | −37.4 / −42.0 / −49.7 / −53.4 K |
  | 4000 rpm, 110 kPa | −47.5 K (−11.0 %) | −30.9 % | −134.9 / −139.6 / −147.3 / −151.0 K |
  | 5500 rpm, 135 kPa | −65.5 K (−14.3 %) | −29.0 % | −239.9 / −244.6 / −252.3 / −256.0 K |

  Worst case over 2000–5800 rpm × 70–135 kPa: EGT 261 K, CHT 14.7 %,
  fuel 33.4 %, expected brake power 42.9 %
  (`scripts/fit_healthy_baseline_to_simulator.py`, diagnostic only).
- **Effect:** on a healthy engine the CHT, fuel and EGT residuals carry a
  large standing offset, so absolute residual thresholds are not usable yet.
  Fault detection still works on the change from nominal: a full-severity
  COOLING_FAULT moves the CHT residual by +35.0 K (−47.5 K → −12.5 K at
  4000 rpm / 110 kPa). The nominal-residual tests in
  `tests/unit/test_residual_expectations.py` are strict xfails recording
  these numbers.
- **Also:** the expected-power form `rated × MAP/ref × rpm/5800` has no
  friction term. The simulator gives all four cylinders the same EGT, so the
  per-cylinder offsets (−8, −3, +4, +7 K) cannot be verified against it.
- **Needs:** healthy-flight data from the actual installation to fit the EGT,
  CHT, power and fuel coefficients and the per-cylinder offsets. Do not fit
  them to the simulator; the diagnostic script's least-squares numbers are
  for information only.

## OI-6 — No airspeed channel for the CHT cooling credit

- **Where:** `HealthyExpectationModel.expected_cht_k`
  (`CHT_AIRSPEED_FRACTION = 0.0`).
- **Effect:** M-11's CHT expectation has a ram-air cooling credit, but no
  airspeed is acquired (`RawSignalRecord` / `OperatingPoint` have none), so the
  credit is never applied. In flight at cruise airspeed the expectation is up
  to 18 K (`k_cht_airspeed`) too hot, which masks CHT rises of that size;
  a low-power, high-airspeed descent is unaffected in the alarm direction.
- **Needs:** an airspeed (or cooling-air pressure) channel through L1, then
  pass the airspeed fraction into the operating point.

## OI-7 — Crank-CoV misfire threshold (partly resolved)

- **Fixed (Prompt 8b):** channel A limit is now
  `max(gate_crank_cov_floor_pct, gate_baseline_multiplier × gate_healthy_crank_cov_pct)`
  = max(0.30 %, 3.0 × healthy CoV) (M-06 baseline branch). The M-06 2.0 %
  fixed threshold is kept in `MisfireConfig` as a documented reference only.
  Simulator MISFIRE at 4000 rpm: CoV 0.306 / 0.508 / 0.808 / 1.008 % at
  severity 0.3 / 0.5 / 0.8 / 1.0 → all CONFIRMED (half-order 0.41–0.66);
  BEARING_WEAR and nominal (CoV 0.015 %) stay NORMAL.
- **Remains:** the 0.30 % floor is marked VERIFY (engineering estimate from
  a ~1.6 % per-revolution dip for one missed power stroke with ≈0.1 kg·m²
  crank-referred inertia). `gate_healthy_crank_cov_pct` is None until
  healthy flight data exists, so the floor always applies. At severity 0.3
  the misfire is CONFIRMED but no cylinder is localised (the per-cylinder EGT
  evidence is below its thresholds), so `misfire_detected` stays all False.
- **Needs:** healthy-flight crank CoV for the baseline, and real or
  bench-induced misfire data to verify the floor.

## OI-8 — No measured oil viscosity; LHI viscosity term excluded

- **Where:** `src/l2_digital_twin/lubrication_model.py`,
  `LubricationModel.lubrication_health_index`.
- **Effect:** viscosity is not measured; `dynamic_viscosity_pa_s` is derived
  from oil temperature with the repo Vogel curve, so a measured/expected
  viscosity ratio carries no information. The LHI viscosity component is
  therefore `valid=False` ("no measured viscosity channel; derived from oil
  temp"), reported with its derived value and zero weight; weights are
  re-normalised in log space over the valid components (pressure,
  over-temperature), and `lhi_coverage` = 2/3 records the missing share. LHI
  cannot detect fuel dilution or shear-down that changes viscosity without
  changing pressure.
- **Model conflict:** `M10.expected_viscosity_pa_s` (Vogel with c in degC,
  `T + 140`) disagrees with the repo curve (c in kelvin, `T − 140 K`): with
  the repo value as "measured", the viscosity factor would be 1.118 at 35 °C,
  1.000 at 85 °C and 0.963 at 115 °C. It is kept but not used for LHI.
- **Needs:** an oil-condition/viscosity sensor or oil-analysis data; then
  reconcile the two viscosity curves against the actual oil grade and restore
  the viscosity component.

## OI-9 — VHI references and envelope band (largely resolved)

- **Fixed at source (Prompt 8b):** M-04 envelope band moved from 300–1000 Hz
  to 650–1000 Hz (above 6X = 580 Hz at 5800 rpm, below Nyquist 1024 Hz) with
  raised-cosine band edges (`envelope_taper_hz` 50 Hz) and a Tukey time
  window (α 0.1); the envelope RMS uses the untapered centre of the record.
  A 2 m/s² 4X tone now leaks 0.00000 m/s² into the envelope at every rpm
  (was 0.03–0.07). The Nyquist guard accepts the new edges.
- **Simulator bearing model changed:** the 600 Hz AM carrier (±BPFO sidebands,
  which could not fit the band and exceeded Nyquist at high rpm) is replaced
  by periodic impacts at BPFO ringing an 800 Hz structural resonance
  (damping ratio ≈0.13). BEARING_WEAR 0.05: envelope ×17–33 (was ×6–15) while
  overall RMS ×1.17–1.26 (unchanged).
- **VHI references:** re-derived rpm table (still labelled "healthy
  installation baseline — provisional, simulator-derived"). Healthy VHI over
  2000–5800 rpm × 80–135 kPa: 0.995–1.003, 0 of 27 outside 0.8–1.2 (before:
  0.609–1.877, 6 of 27 outside). The envelope reference is now the sensor
  noise floor (~0.071 m/s²).
- **Remains:** references are simulator-derived and must be replaced by a
  baseline from the first healthy flights. The simulator generates no firing
  harmonics above 4X, so real 6X/8X content (8X = 773 Hz at 5800 rpm, inside
  the band) is untested. M-10's 2.0 / 2.5 / 0.05 references are kept as
  documentation; their unit (labelled m/s², magnitude typical of mm/s
  velocity) is unconfirmed and not converted. VHI has no half-order or
  kurtosis term; misfire is covered by the gate and CSI, not VHI.
- **Needs:** healthy-flight baseline; a check against real engine spectra for
  firing harmonics inside 650–1000 Hz.

## OI-10 — CSI thermal term (resolved for the simulator; verification pending)

- **Fixed (Prompt 8b):** the thermal term no longer uses the uncalibrated
  expected CHT (`HealthyExpectationModel.CALIBRATED = False`, OI-5). It is
  |CHT slope| / 0.5 K/s, computed only when |dP/dt| ≤ 0.5 kW/s and
  |dMAP/dt| ≤ 500 Pa/s over the window; during a transient it is excluded
  (invalid, coverage 2/3), not zero. The CHT slope was already a
  least-squares fit; the window is now 30 s (was 10 s). Noise-only slope std
  (0.72 K CHT noise at 1 Hz): 0.0709 → 0.0145 K/s, i.e. 0.142 → 0.029 of CSI.
  Simulator: steady CSI 0.011 (was 0.198); 20 s climb CSI 0.010 with the
  thermal term excluded (was 2.09); CHT runaway at constant power 1.000
  ALARM; MISFIRE 1.0 raises CSI 0.0107 → 0.0207 via the crank term.
- **Remains:** the transient limits (0.5 kW/s, 500 Pa/s) and the 30 s window
  are project choices, not verified against real noise and thermal time
  constants. The thermal term is blind during transients (by design). A
  misfire moves CSI only by CoV/100 (~0.01), so CSI alone does not flag it.
- **Needs:** flight data to set the window and transient limits against the
  real CHT sensor noise and engine thermal response.

## OI-11 — CSI crank term is CoV/100 (known limit, not retuned)

- **Where:** M-09 `combustion_stability_index` (speed term = crank CoV % / 100).
- **Effect:** a full misfire (crank-period CoV ≈1.0 %) moves CSI only
  0.011 → 0.021 (end-to-end: 0.0142 → 0.0242 at 4000 rpm), far below the
  0.15 WARNING line. CSI does not flag misfire; misfire detection is carried
  by the dual-channel gate (CONFIRMED at severity ≥ 0.3).
- **Decision:** recorded as a known limit; the M-09 weighting is not retuned.

## OI-12 — L1 validator dropped source-reported invalid channels (resolved, Prompt 9b)

- **Was:** `TelemetryValidator.validate_packet` rebuilt `signal_quality` from its
  own checks and discarded the source's `invalid_channels`; a MAP dropout
  (`map_counts` = 0, in range) reached L2 as `map_pressure` valid=True, 0 Pa,
  and the residual engine silently used 101 325 Pa. A lost EGT channel was
  visible only per cylinder.
- **Fixed:** the validator merges source flags and reasons (source reason wins)
  with its own findings; score = min(source, computed). `sensor_inverse` maps
  every raw field to the channels derived from it (`RAW_FIELD_DEPENDENTS`,
  including cold junctions, ADC reference and burst sample rate), so a channel
  is invalid if any dependency is flagged. `NormalizedSignalRecord` carries
  `invalid_channels` and `channel_coverage`; `EGTDiagnosticResult` carries
  `coverage` and `invalid_cylinders`; residuals are invalid (not built on a
  substituted default) when rpm, MAP or an ambient input is invalid.
  End-to-end: MAP dropout → map_pressure and boost_pressure invalid, record
  coverage 0.92; EGT1 dropout → egt_cyl_1 invalid, EGT coverage 0.75, record
  coverage 0.96. A parametrised test over every raw field of RawSignalRecord
  checks that a source flag survives to L2 (and fails for an unmapped new field).
- **Remains:** the diagnostic `overall_status` values keep their NORMAL /
  WARNING / CRITICAL meaning; lost channels are reported through coverage,
  not by changing a status band.

## OI-13 — CSI start-up with an under-filled window (resolved, Prompt 9b)

- **Was:** the thermal term was computed from 3 samples (~2 s of CHT); NOMINAL
  read CSI ALARM at records 2–4 and WARNING at 5–6 (max 1.11).
- **Fixed:** the thermal term is excluded (coverage 2/3, reason "CHT window
  filling") until the CHT window spans the full `csi_window_s` (30 s).
  NOMINAL end-to-end: no WARNING/ALARM at any of 120 records.

## OI-14 — Simulator inconsistencies (resolved, Prompt 9b; OI-1 remains)

- **a. Oil pressure vs oil temperature — fixed.** Simulator oil pressure is
  now `p_ref(rpm) × (μ(T)/μ(T_ref))^0.3` with the simulator's own Vogel
  constants (`simulator.physics.oil_vogel_b_k` 1100 K, `oil_vogel_c_k` 145 K,
  reference 363.15 K; VERIFY), applied after fault temperature changes.
  COOLING_FAULT 1.0 LHI: 1.309 ALARM → 1.007 NORMAL. NOMINAL LHI 1.034 →
  0.999; OIL_DEGRADATION 1.0 LHI 0.369 → 0.225.
- **b. Boost-leak fuel — fixed.** The INTAKE_BOOST_LEAK MAP reduction is
  applied before the air → fuel → IMEP chain (SRD-FUN-152), and the old
  arbitrary power multiplier is removed; power follows physically. Twin SFC:
  0.2368 → 0.2621 kg/kWh (+10.7 %). It rises even with OI-1's fixed 42 %
  indicated efficiency because friction is a larger share of the lower IMEP;
  the magnitude is still set by OI-1, not by a measured efficiency.

## OI-15 — NaN in NormalizedSignalRecord JSON (resolved, Prompt 9b)

- **Was:** invalid channels held NaN and `model_dump(mode="json")` kept it.
- **Fixed:** `ChannelValue.value` is `float | None`, and its JSON serializer
  emits `null` whenever the channel is invalid or the value is non-finite
  (in memory the value is unchanged). A record with every measured channel
  invalid encodes with `json.dumps(..., allow_nan=False)`.

## OI-16 — Replay pipeline and L3 coverage

- **Where:** `src/l1_data/simulator/replay_and_whatif.py`,
  `PipelineReplayAdapter.process_sequence`; L3 classifiers.
- **Effect:** the replay pipeline calls stateless per-record convenience
  functions, so windowed indicators (CSI, EGT/oil rate terms) never fill —
  CSI is always UNKNOWN there — and `PipelineStepResult` does not expose the
  misfire verdict, CSI, LHI or VHI. The health index does respond (NOMINAL
  0.905, MISFIRE 1.0 0.825, OIL_DEGRADATION 0.789), but the L3 fault
  classifier predicts NOMINAL with anomaly score 0.000 in every scenario.
- **Needs:** stateful L2 components per replay sequence; expose the L2
  indices in the step result; L3 model training/validation (out of the physics
  audit's scope).

## OI-17 — Configuration placeholders to verify (Prompt 10)

- **`engine.gearbox_ratio`** = 2.54 (set before Prompt 12; value supplied by
  the project for the Rotax 915 iS reduction unit), still marked `# VERIFY
  against Rotax 915 iS Operators Manual`. It was None through Prompts 10–11
  (propeller_speed valid=False then). The earlier unsourced 0.414 factor
  (≈2.42:1) was removed in Prompt 10.
- **`engine.engine_hours_at_install`** = None: must come from the engine
  logbook at installation. Until it is set, `engine_hours` is valid=False; the
  running time accumulated since monitoring started is reported in the
  channel's fault_flag.
- **Not instrumented:** intake-air temperature, wastegate position and
  throttle position have no raw signal. Coolant temperature is derived from
  Prompt 14 on (see OI-22).
  Fuel pressure, injector pulse width, SOI and ignition timing are derived
  from Prompt 13 on when their raw fields are present (see OI-21).
  `OperatingPoint.throttle_pct` is None when throttle is not instrumented
  (it was a substituted 50 %). Bus voltage, alternator current and battery
  current are derived from Prompt 12 on when their counts are present (see
  OI-19); without them they remain valid=False.

## OI-18 — Raw interface issues found while adding Prompt 11 signals (decided before Prompt 12)

- **Engineering units in RawSignalRecord — accepted exception.** `egt_cold_c`,
  `cht_cold_c`, `ambient_temp_c` (°C) and `ambient_press_pa` (Pa) are not
  refactored: they come from digital smart sensors (MAX31855-type
  cold-junction chip, MS5611-type barometer) whose raw output already is °C /
  Pa. Documented in RAW_SIGNAL_INTERFACE.md. **VERIFY against the actual
  hardware selection.**
- **Live CAN adapter placeholders — fixed.** Every field is None until the
  first frame of its group arrives ("No CAN frame 0x1xx received yet"); fields
  no frame carries are None ("Not carried on the CAN bus"). Core raw fields
  became required-but-nullable to allow this; L2 marks derived channels
  invalid. Consequence: from the live CAN adapter, EGT/CHT (no cold junction),
  MAP/oil pressure (no ADC reference), vibration and ambient-derived channels
  are valid=False until those signals are added to the bus.
- **Unsigned CAN encodings — fixed.** 0x101/0x102 thermocouple µV use a
  +10 000 µV offset (1 µV/count, −10 000 … 55 535 µV); 0x106 ignition delay is
  signed int16. Values at the encoding limits are flagged invalid.
- **Coolant NTC range — fixed.** Moved to 0x107 as u32 Ω×10 (100 Ω … 400 kΩ
  round-trips); 0x103 now carries fuel_press, bus_v, alt_i and a spare (the spare
  became batt_i_counts in Prompt 12).
- **Remains:** the DS-12 ECU parameter list is not available, so no ECU
  cross-check fields are defined. The live adapter's 0x103/0x104-0x106 fields
  are optional; the CSV adapter still fills missing core columns with
  defaults (30 000 µV, 135 Ω, 2048 counts…) — not addressed here.

## OI-19 — Electrical model placeholders to verify (Prompt 12)

No Rotax 915 iS charging-system or battery figure was available, so every
manufacturer-dependent value below is a placeholder marked `# VERIFY against
Rotax 915 iS Operators Manual` (or against the installed battery). None is a
manufacturer figure.

- **Generator pole count.** `engine.generator_poles` = **None** (L2). With
  None, L2 never reports `ripple_dominant_frequency_hz` as valid ("generator
  pole count unknown"); the ripple order check (ripple Hz / generator rev/s
  is an integer) still works without it. The simulator uses
  `simulator.electrical.generator_poles_placeholder` = **12**, a
  SIMULATION-ONLY placeholder that is NOT a Rotax figure. Tests that need a
  pole count set it explicitly as a test assumption.
- **L2 generator geometry (VERIFY):** `engine.generator_phases` = 3,
  `engine.generator_drive_ratio` = 1.0.
- **L2 expectations (VERIFY):** `electrical.regulator_setpoint_v` = 14.2 V,
  `electrical.cut_in_rpm` = 2400 rpm, `electrical.battery_ocv_nominal_v` =
  12.7 V, `electrical.battery_r_int_nominal_mohm` = 15 mΩ.
- **L2 bands (project choices, to confirm in flight):** charging residual
  warning / alarm 0.5 / 1.0 V; ripple healthy 0.5 % with warning / alarm at
  2× / 3×; R_int warning / alarm at 1.5× / 2.5× nominal; EHI weights charging
  0.5, ripple 0.25, R_int 0.25; EHI warning / alarm 0.75 / 0.5; R_int
  estimator gates (≥ 1 A step, ≤ 50 rpm change, ≤ 1 s apart, alternator
  current < 1 A, median of 7, ≥ 3 steps); ripple peak SNR 100; ripple band
  50–4000 Hz.
- **Sensor chain (calibration basis, must match the fitted parts):** bus
  voltage divider 20 V full scale; alternator Hall 0–50 A; battery Hall
  bidirectional ±50 A with zero at 0.5 × ADC reference (positive =
  discharge).
- **Simulator (`simulator.electrical`, VERIFY):** regulator setpoint 14.2 V,
  droop 0.005 Ω (finite stiffness), response time constant 0.3 s, cut-in
  2400 rpm, rated current 30 A, drive ratio 1.0, 3 phases; battery (generic
  12 V AGM basis) 17 Ah, R_int 15 mΩ, charge polarisation 0.3 Ω / 30 s, OCV
  table 11.8–12.8 V over SoC 0–1, initial SoC 0.9; loads 4–22 A with random
  3–8 A steps every 6 s on average (project choice); ripple filter gain 0.1.
  The simulator constants are independent of the L2 ones and are not
  calibrated to them.
- **Health weights rebalanced for the new electrical component:** thermal
  0.20 → 0.18, lubrication 0.20 → 0.18, vibration 0.20 → 0.18, combustion
  0.15 → 0.135, performance 0.10 → 0.09, anomaly/fault 0.15 → 0.135,
  electrical 0.10 (new). When electrical signals are absent the component is
  excluded and the remaining weights are re-normalised; coverage is reported.
- **Observability limit (by design, not a defect):** R_int is observable only
  while the battery supplies a load step (below cut-in, or alternator current
  < 1 A). In a 120 s NOMINAL flight at 4000 rpm the regulator holds the bus
  throughout, giving **0** usable R_int windows; R_int is valid=False with
  "regulator holding bus; not observable". BATTERY_DEGRADATION is therefore
  detected only at idle / below cut-in or during a charging failure.

## OI-20 — `/engine/health` constants and single-record evaluation (resolved)

Found while adding the electrical fields (Prompt 12). The endpoint returned
hard-coded `component_health = {"thermal": 0.95, "lubrication": 0.90,
"vibration": 0.95, "combustion": 0.92}` and `trend = "STABLE"`, and it
evaluated the latest record alone.

**Resolved:**
- **Stateful pipeline.** The replay pipeline is now a stateful `PipelineRun`.
  Every L2 model and L3 engine is built once per stream; before this, each
  record got fresh twin, EGT, lubrication, vibration, misfire and residual
  engines.
- **All L2 outputs reach L3.** The EGT diagnosis, vibration, combustion,
  derived and electrical states now go to HealthSupervision. The misfire
  detector now receives the EGT result instead of the `(state, result)` tuple.
- **Running pipeline.** `/engine/health` uses `RunningPipeline`: only new
  records are stepped, history persists across calls, and the run restarts
  when the stream changes.
- **Real L3 output.** Health index, degradation state, trend, rate, component
  health and data coverage all come from the L3 `HealthState`.
- **No default scores.** A component without valid evidence is `null`, with
  its reason in `component_health_reasons`; `HealthState.component_unavailable`
  carries the reasons. Such a component also no longer enters the Health
  Index average. It used to count as a 1.0 score.
- **Evidence validity unchanged.** The evidence-quality convention is kept:
  an absent combustion or anomaly input still counts 0.5 toward validity.
  `tests/unit/test_module14.py` depends on it and was not changed.
- **Tests.** `tests/integration/test_engine_health_running_pipeline.py`
  compares NOMINAL with OIL_DEGRADATION: lubrication 1.000 vs 0.141, and
  health index 0.899 vs 0.725. It checks that the API equals the pipeline and
  that null components come with a reason. It also checks history across
  calls, and that R_int is observable through the API at idle.

**Resolved after Prompt 14 (follow-up commit):**
- **Other endpoints.** `/engine/rul`, `/mission/risk`, `/diagnostics/anomaly`
  and `/diagnostics/fault` now run through `RunningPipeline` and return the
  L3 outputs. The following constants are gone:
  - RUL `trend="STABLE"` and ×0.8 / ×1.2 bounds;
  - `flight_phase="CRUISE"` and `risk_trend="STABLE"`;
  - fault confidence 0.90 / 1.0 with invented probabilities;
  - `status="SUCCESS"` while the models are unavailable.

  Each response carries `records_in_window`. When a model is unavailable,
  its values are `None` with status MODEL_UNAVAILABLE (OI-16).
- **Mission-risk flight phase.** Mission risk reports GROUND, because nothing
  in the pipeline infers a flight phase yet. The endpoint used to claim
  CRUISE.
- **Health trend.** The trend is no longer a one-step dHI/dt. It is the
  Theil–Sen slope over a 120 s window with Sen's CI, reusing the Prompt 14
  `TrendWindow` and `OperatingPointStepDetector`.
  - DEGRADATION or IMPROVING only when |slope| ≥ 0.006 HI/min and the CI
    excludes zero.
  - RAPID_DEGRADATION at ≥ 0.6 HI/min (the former thresholds).
  - STABLE otherwise, and INSUFFICIENT_DATA until the window is full.
  - The window restarts on operating-point steps.
  - `health_rate` is None while there is insufficient data.
  - NOMINAL now ends STABLE: +0.00138 HI/min, CI +0.00072..+0.00206.
  - OIL_DEGRADATION ends DEGRADATION: −0.0153 HI/min, CI −0.0214..−0.0114.

**Remaining:**
- **WebSocket.** It still processes `records[:5]` through `process_sequence`.
- **RUL at OIL_DEGRADATION.** RUL reports 0.113 h (about 7 min) with trend
  DEGRADATION. That is the RULEstimator's extrapolation of the HI slope,
  now fed a real trend. It has not been checked for plausibility.
- **Thermal on NOMINAL.** `thermal_health` is 0.60 on NOMINAL because the
  healthy expectations are uncalibrated (OI-5).
- **Anomaly/fault.** `anomaly_fault_health` is `null` because both L3 models
  are MODEL_UNAVAILABLE (OI-16).

## OI-21 — Injection, ignition and fuel-system placeholders and limits (Prompt 13)

No Rotax 915 iS ECU calibration, injector or fuel-pump data was available.
Every value below is a placeholder marked `# VERIFY against Rotax 915 iS
Operators Manual` (or the component datasheet). None is a manufacturer figure.

**Injector and rail (L2 `injection.*`, simulator `simulator.injection.*`):**
- Injector static flow K_inj = 0.004 kg/s at the reference rail pressure.
- Dead time 900 µs.
- Rail − manifold reference and setpoint 300 kPa; regulator droop 0.
- The regulator is assumed manifold-referenced, i.e. L2 computes
  rail dP = p_gauge + p_amb − MAP.
- The simulated ECU does not compensate pulse width for rail pressure.

L2 and the simulator use the same injector and regulator values, as a real
installation would take them from the datasheet.

**Fuel pump (simulator only):** Q_max 0.03 kg/s, stall at 600 kPa gauge.
FUEL_SYSTEM_FAULT removes 85 % of capacity at severity 1.

**Fuel-pressure sensor:** gauge, 1 MPa full scale.

**ECU maps (control schedules, not engine properties):**
- Start of injection: 300–370° after the TDC reference.
- Ignition advance: 13–33° BTDC over rpm × MAP.
- L2 holds a copy of the advance map as the ECU schedule.
- Knock retard at DETONATION_KNOCK severity 1: 8°.

**Exhaust energy balance** (EGT rise ∝ x(λ) η_c f/(1+f), peak EGT slightly lean
of stoichiometric). The constants are chosen independently in the simulator
and in L2:

| | x_st | rich slope | lean gain / scale | retard |
|---|---|---|---|---|
| Simulator | 0.30 | 0.20 | 0.08 / 0.08 | 0.012 per ° |
| L2 | 0.32 | 0.25 | 0.10 / 0.10 | — |

L2's model puts peak EGT at λ 1.111. The charge temperature is ambient + 30 K,
because there is no IAT channel.

**Bands (project choices):**
- Rail residual: ±30 / ±60 kPa.
- Delivery and flow ratio: ±0.05 / ±0.10.
- Duty limit: 85 %.
- Knock: retard ≥ 2°, supported by EGT ≥ +15 K vs the other cylinders or a
  CHT residual rise ≥ 10 K.
- Running medians over 10 records.

**Limits found (not tuned away):**
- **Rich of peak only.** The EGT-based injector flow ratio is defined only
  while the operating λ is rich of the model's peak-EGT λ, because EGT is
  not monotonic in λ around the peak. At cruise (λ ≈ 1.02–1.06) a lean
  cylinder past the peak yields only a bound. A 15 % clog on cylinder 3 at
  4000 rpm / 110 kPa gives:
  - flow ratio ≤ 0.951, just short of the 0.95 warning band;
  - fuel delivery ratio 0.962, inside ±0.05;
  - so it is not identified (strict xfail).

  At rated power (5800 rpm / 140 kPa, λ 0.83–0.94) the same clog gives a flow
  ratio of 0.863, true value 0.85. A leak there gives 1.143, true value 1.15.
- **Absolute EGT residual.** It carries the OI-5 bias (about −260 to −274 K
  per cylinder at rated power), so the diagnosis uses each cylinder's M-11
  residual relative to the median of the other three. The strict xfail on the
  absolute residual under a clog records −175.5 K. The common-mode
  (engine-wide) mixture is covered by the rail residual, the rail mixture
  factor and the fuel delivery ratio (SRD-FUN-084), not by EGT.
- **Operating λ.** It comes from the twin (`lambda_derived`), which reads
  0.83 at rated power where the simulator runs 0.937 (same class of bias as
  OI-5).
- **Lean-cylinder torque.** The simulator does not put a lean cylinder's
  torque loss into the crank-period burst: only MISFIRE shapes the burst.
- **SRD-FUN-044.** The requirement text is not in the repository. It is
  implemented as described in Prompt 13: hot means lean, and cold means rich
  unless the M-06 gate confirms a misfire on that cylinder (non-firing).
- **Knock.** It no longer adds accelerometer vibration in the simulator:
  knock energy lies above the 1024 Hz Nyquist limit and is anti-alias
  filtered. The existing +40 K CHT term for knock remains an additive
  ground-truth perturbation (pre-existing; to be replaced by a heat-flux
  model). The legacy `engine_model.py` / `FaultInjector` path, which
  ScenarioRunner does not use, still scales vibration for knock.

## OI-22 — Coolant loop, overheating trends and hot-weather profile (Prompt 14)

**Cooling layout.** `engine.has_coolant_circuit = True` was added and marked
VERIFY. Config was not silent: `CoolingConfig` (50/50 glycol, thermostat
85/95 °C, radiator effectiveness 0.65), the README and `simulator/cooling.py`
all describe a liquid circuit, but none cites a source. With the flag False,
every coolant parameter is unavailable with that reason (SRD-FUN-045
pattern).

**Placeholders (VERIFY; no Rotax cooling data available):**
- **Coolant NTC.** Steinhart-Hart A/B/C = 1.129148e-3 / 2.34125e-4 /
  8.76741e-8. These are the widely published example for a 10 kΩ (25 °C)
  NTC, not a datasheet for the fitted sensor.
- **Coolant loop (`simulator.cooling`):**
  - head heat 10 % of fuel power
  - coolant flow 1.0 kg/s at 5800 rpm, proportional to rpm
  - 2.5 kg coolant plus 9 kJ/K of head metal
  - radiator face 0.06 m², duct fraction 0.3, UA 1500 W/K at 45 m/s
  - thermostat 85–95 °C with 2 % leak
  - head-to-coolant hA 1450 W/K
  - oil–coolant coupling 0.5
  - oil-cooler ambient gain 1.0
- **Fault magnitudes:** blockage 80 %, pump loss 80 %, coolant loss 70 %
  (head hA −60 %) at severity 1.
- **Thermal lags:** head 90 s, oil 300 s.
- **L2 coolant expectation:** 90 °C + 6 K × load + 0.1 × (T_amb − 288.15);
  bands 8 / 15 K on the positive side, and 15 / 30 K on CHT − coolant.
- **Trends:**
  - window 180 s, at least 20 samples, 95 % confidence
  - minimum slope 0.2 K/min
  - alert horizon 30 min
  - expectation lags: CHT 60 s, coolant 90 s, oil 180 s, EGT 5 s
  - the window restarts after a step of more than 500 rpm, 20 kPa or 5 K
- **Alarm limits.** The Appendix B text is not in the repository. The
  limits used are:
  - CHT 135 °C (`cooling.max_cht_c`; `rotax_915is_params.py` cites "OM 4.3",
    not checked)
  - EGT 900 °C (`egt_critical_temp_k`)
  - oil 130 °C (`lhi_over_temp_limit_c`)
  - coolant 120 °C (new placeholder)

**Model changes:**
- **COOLING_FAULT is physical.** It is now a physical perturbation of the
  loop: `radiator_blockage` (default), `pump_degradation` or `coolant_loss`.
  It used to be a +35 K CHT and +25 K oil offset.
- **Thermal lags.** Simulated CHT and oil temperature follow their
  steady-state maps with first-order lags. The maps used to respond
  instantly, which made a power step look like an overheating trend. Steady
  runs are unchanged.
- **Ambient.** `ScenarioRunner` now uses ISA(altitude) + deviation. It used to
  ignore altitude and always use 288.15 K / 101 325 Pa.
- **What-if overrides.** `WhatIfEngine` now applies `ambient_temp_k` and
  `ambient_pressure_pa`, which it accepted but silently ignored before. It
  also has `isa_deviation_k` and `hot_weather_mission_profile()`
  (SRD-FUN-154).

**Trend gates added after measurement (principled, not tuned to a case):**
- The measured temperature must itself be rising by at least the minimum
  slope. A falling temperature after a power reduction produced positive
  residual slopes.
- An abrupt operating-point change restarts the window: the residual steps
  there, which is an offset, not a rate (cf. M-09).

**Results:**
- **Radiator blockage.** The fault ramps over 600 s at 5000 rpm / 125 kPa.
  - The coolant residual goes from −7.4 to +39.3 K.
  - Coolant time-to-limit is valid from 180 s and falls strictly
    (8485 → 0 s at 30 s sampling).
  - CHT time-to-limit falls overall but has two upticks: 3808 → 4663 s and
    1384 → 1517 s.
- **COOLING_FAULT magnitude (strict xfails).** At cruise the physical
  blockage raises CHT less than the old offset did: +14.4 K at steady state
  and +10.4 K 89 s after onset, against the >20 K criterion. The thermostat
  absorbs part of the loss.
- **Rated power.** At 5800 rpm / 140 kPa, 80 % blockage runs the coolant to
  185 °C. There is no boiling or coolant-loss model.
- **Healthy full-power climb, 0 → 2400 m (strict xfail).** It gives a
  WARNING at 446–485 s.
  - The simulated CHT is still settling (+0.42 K/min).
  - The M-11 expectation falls with ambient (0.85 K/K), and its lag is
    shorter.
  - Residual slope +0.43 K/min gives 27 min to the limit.

  At constant altitude the same power step gives no alert. Closing this
  needs calibrated expectations (ambient or density sensitivity, head time
  constant) from flight data, as in OI-5.
- **ISA+30 at rated power.** Simulated oil reaches 131 °C, above the 130 °C
  limit (oil-cooler ambient gain 1.0). The default hot-weather mission
  (climb power to 1500 m, then cruise) raises no alarm at ISA+0 or ISA+30.
- **Far-horizon trends.** In healthy runs some channels still report a valid
  time-to-limit far beyond the horizon (EGT 6–11 h, coolant ≥ 39 min). These
  do not alert.

## OI-23 — Fault taxonomy (Prompt 15) and items deferred to Prompt 18

**Taxonomy (Prompt 15):**
- **Placeholders (VERIFY).**
  - IMBALANCE simulator constants: severity 1 = 1.0 in/s peak lateral
    velocity at 2300 propeller rpm; lateral/vertical mount compliance 2.0;
    axial fraction 0.1. The basis is general-aviation propeller-balance
    guidance.
  - Accelerometer axes: y lateral, z vertical.
  - Imbalance dominance threshold: 0.5 of the lateral velocity energy, the
    definition of "dominant".
- **Not identified at the audit point** (docs/FAULT_TAXONOMY.md):
  - EXHAUST_VALVE_LEAK → INJECTOR_FAULT (EGT cannot separate them);
  - INTAKE_BOOST_LEAK → not evaluable (no boost reference);
  - COOLING_FAULT → NOMINAL (the thermostat masks it; the flag fires);
  - INJECTOR_FAULT at cruise (OI-21);
  - BATTERY_DEGRADATION above cut-in.
- **Misfire flag.** `combustion_instability` does not fire on the misfire run
  (OI-11).
- **Missing references.** SRD-FUN-112 and the DS-20/21 datasets are not in
  the repository. They are not mapped or run.
- **ML model.** The pipeline's ML result stays MODEL_UNAVAILABLE (OI-16). The
  rule-based diagnosis is reported beside it as `rule_based`, and is not
  substituted for it.

**Deferred to Prompt 18: RESOLVED in Prompt 18** (tests:
`tests/integration/test_oi23_resolution.py`):
- **Flight phase (resolved).** `PipelineRun` now derives the phase for every
  record with `MissionPhaseClassifier`.
  - Inputs: rpm, MAP, throttle, pressure altitude, and a vertical rate from
    the least-squares slope over `mission.vertical_rate_window_s` = 10 s
    (VERIFY).
  - Default mission: TAKEOFF → CLIMB → CRUISE, never GROUND in flight.
  - When the phase is not derivable (rpm invalid), the last derived phase is
    kept with `phase_quality` 0 instead of GROUND.
  - VERIFY: the phase thresholds in `MissionConfig` are unchanged project
    values. At 1800 rpm idle in flight the phase reads CRUISE with quality
    0.7, because no DESCENT/LANDING vertical rate is present.
- **WebSocket (resolved).** Every record is streamed through one stateful run
  with its own validator, then a `STREAM_COMPLETE` status (12 of 12 records
  in the test).
- **RUL plausibility (checked, algorithm unchanged).**
  - **Baseline RUL.** It extrapolates the HI slope of a short run into hours
    (0.113 h on the 120 s OIL_DEGRADATION run), which is not a plausible
    engine RUL. The test asserts it is never validated, its interval is
    labelled not calibrated, the API shows UNVALIDATED, and it stays within
    [0, horizon].
  - **Lifetime RUL.** Along a synthetic degradation history it is plausible:
    in [0, 400] h, inside its interval, and falling as severity rises.
  - **Does not transfer to engine variation (new finding).** On the life2
    test engines (engine variation), 78 % of healthy-engine windows get an
    ESTIMATED RUL. Band MAE is 35.9 / 38.4 / 103.4 h and 90 % coverage 0.543
    (`reports/headline_metrics.md`). It needs retraining on life2 (OI-24).

## OI-24 — Trained L3 models (Prompt 16)

See docs/ML_MODELS.md and reports/ml_metrics.*. Every metric was measured on
simulator data.

- **Targets missed (v2, test split).**
  - Anomaly PR-AUC 0.841 against 0.93 (−0.089).
  - Classifier macro-F1 0.767 against 0.81 (−0.043).
  - FPR 0.046 meets ≤ 0.05.
- **Weak classes.** FUEL_SYSTEM_FAULT (F1 0.13), BATTERY_DEGRADATION (0.12)
  and INTAKE_BOOST_LEAK (0.26, flagged unreliable: no boost reference).
- **RUL (updated in Prompt 16b).** A lifetime-scale RUL model (method ported
  from aerotwin_ml) is validated on held-out simulated engines and adopted.
  - MAE 19.0 / 25.6 / 84.1 h in the 0–50 / 50–150 / 150–400 h bands, against
    the baseline's 363.8 / 304.6 / 136.1 h.
  - Open: 90 % interval coverage is 0.755 overall and 0.52 in the 150–400 h
    band on test (0.93 on validation), below target.
  - Open: 5.4 % of healthy-engine windows get an ESTIMATED RUL.
  - Open: live use needs engine hours (`engine_hours_at_install`) and
    `ml.rul_model_version`. Without them RUL is reported UNVALIDATED.
  - Open: the live `RULEstimator` baseline resets its history on gaps longer
    than 300 s, so it cannot estimate across flights.
- **CHT-group dropout.** It still answers without reducing confidence (9.8 %
  confident-wrong vs 7.1 %). Reason: the group was fully missing in 1.06 % of
  training rows (almost all of them MAP-dropout records), just above the 1 %
  out-of-distribution threshold, and the CHT features carry 2 % of the
  importance. Fix: judge missingness patterns, not single groups.
- **Models are opt-in.** `ml.model_version` must be set to use them, and the
  artifacts are git-ignored (regenerate with the scripts). The repository
  default is still None (rules only); v2 is the version to enable.
- **v3 (Prompt 17b) not adopted.** It was trained with the lifetime-fleet
  train engines added (life2: engine variation and benign ageing).
  - Lifetime healthy false alarms: 0.857 → 0.158. The anomaly detector part
    went 0.825 → 0.001.
  - Prompt 16 test anomaly FPR: 0.046 → 0.201. This fails the ≤ 0.05
    target.
  - Lifetime detection 1.000 → 0.917 and median lead time 297 → 225 h. v2's
    detection is inflated because it alarms all the time.
  - Cause: one IsolationForest and one pooled 5 % threshold cannot serve both
    populations. On validation, v3 flags 21 % of Prompt 16 NOMINAL records
    and 0.1 % of lifetime healthy records; v2 flags 5 % and 85 %.
  - Needs: decided on validation BEFORE testing. Either balance the two
    healthy sources in anomaly training, or set the threshold per source
    (or per operating regime). Then evaluate once as v4.
- **No real data.** All training and test data come from the forward
  simulator. The model inherits the simulator's assumptions (OI-5, OI-19 to
  OI-23), and the DS-20/21 data were not used.

## OI-25 — Drift monitoring and adaptation (Prompt 17, 17b)

### Prompt 17b: new target, still rejected

- **Target changed.** Adaptation now targets the fleet's healthy residual at
  the operating point (`src/l3_ml/fleet_reference.py`, fitted on healthy life2
  TRAIN records). Commissioning corrections are now small and plausible (EGT
  a few K, gains about 0).
- **Still rejected.** Healthy false alarms do not drop.
  - With v3: 0.158 → 0.161. Adaptation does not change the rule-based part
    (0.050), and the ML part rises slightly (0.114 → 0.118).
  - With v2: 0.857 either way, because the gate refuses almost every flight.
- **Drift freezes adaptation.** 5676 of 6517 v3 flights were frozen by
  SIGNIFICANT drift.
  - Channels with a very narrow healthy spread (coolant 0.26 K, oil
    temperature 0.2 K, vibration) reach PSI 2–7 from small flight-to-flight
    differences.
  - Later, benign EGT ageing (5 K/100 h) reads as SIGNIFICANT too.
- **Needs:**
  - a minimum engineering-unit effect size per channel before a PSI band
    counts, VERIFY;
  - drift judged per operating regime;
  - a rule that lets slow, monotonic, gate-eligible change adapt instead of
    freezing.

  All three must be decided before the next evaluation.
- **Fleet reference placeholders.** `commissioning_max_offset` is now a
  bound on the deviation from the fleet reference (EGT 60, CHT 30,
  coolant 15, oil temperature 20 K, …), VERIFY. The life2 variation ranges
  are project assumptions, VERIFY.

### Prompt 17: adaptation is rejected

- **Where:** `src/l3_ml/adaptation.py`, `scripts/evaluate_adaptation.py`,
  `reports/adaptation_eval.md`.
- **Result.** On 60 held-out lifetime test engines, false alarms on healthy
  engines went from 0.499 to 0.740 with adaptation. Detection stayed at 100 %
  and the median lead time was 280 h without and 284 h with it.
- **Cause.** Commissioning drives residuals to zero. The frozen v2 models were
  trained on fleet-baseline residuals, which carry the operating-point
  dependent OI-5 bias (EGT about −80 to −228 K on a healthy engine), so zeroed
  residuals are out of distribution for them.
- **Current state.** No live path applies an adapted baseline.
- **Needs:** fix OI-5 first so healthy residuals are near zero, then retrain
  the models. Alternatively, fit the correction to the fleet's healthy
  residual at the same operating point instead of to zero.

### The v2 models do not generalise to the lifetime fleet

- **Result.** Without any adaptation, 50 % of healthy lifetime windows raise
  an alarm: anomaly 39 %, ML 18 %.
- **Effect.** The eligibility gate refuses most healthy flights, so only 37
  of 60 engines ever commissioned.
- **Needs:** training data covering the lifetime fleet's ISA, altitude and
  phase range, and an FPR check on it.

### Placeholders to verify

- **Adaptation settings.** Every value in `AdaptationConfig` is a project
  choice:
  - commissioning flights;
  - offset bounds;
  - linear gain;
  - rate limits and caps;
  - confidence threshold;
  - challenger trigger;
  - promotion margins.
- **Drift bands.** The PSI bands 0.10 / 0.25 are industry convention, not a
  Rotax figure.
- **Benign ageing rates.** `ageing_egt_k_per_100h` and
  `ageing_cht_k_per_100h`, and the 3–6 and 1–2.5 K/100 h used in the
  evaluation, are marked VERIFY.

### Drift measurement limits

- **Vibration quantization.** Healthy `vibration_rms` PSI is about 0.10,
  right at the moderate line, because the ADC quantization leaves few
  distinct values. Needs: a finer vibration count scale, or a
  quantization-aware binning.
- **Running-median channels.** `fuel_delivery_ratio` and
  `injector_flow_ratio_cyl*` are excluded from the overall PSI because they
  are autocorrelated.

### Not yet built

- **Live drift reference.** The live API commissions its drift reference
  itself from the first 200 healthy records. There is no stored per-engine
  reference in the API yet.
- **Challenger retrain.** The trigger is counted, and training, gating,
  registry and rollback exist and are tested. Nothing runs the retrain end to
  end automatically.
- **Bearer token.** It is a single shared token from an environment variable,
  with no rotation and no per-user identity.

## OI-26 — Verification findings (Prompt 18)

- **Rule false alarms from installation offsets.** On the life2 held-out
  healthy engines, 83 of 84 rule-based false-alarm windows are
  INJECTOR_FAULT (5.0 % of healthy windows).
  - Cause: per-cylinder EGT installation offsets (N(0, 8 K), VERIFY) look
    like an injector fault to the cylinder-differential rule.
  - Needs: per-engine commissioning of the cylinder EGT differentials
    (SRD FUN-107), decided on validation data.
- **Cooling fault at cruise.** COOLING_FAULT still reads NOMINAL at cruise;
  only `overheating_trend` fires. INTAKE_BOOST_LEAK and EXHAUST_VALVE_LEAK
  remain not separable (strict xfails in the full-taxonomy e2e tests).
- **Engine rating corrected.** `engine.rated_power_kw` is now 104.0 (was
  105.0, unsourced). Also added: `max_continuous_power_kw` 99.0 at
  `max_continuous_rpm` 5500 and `tbo_hours` 1200. `gearbox_ratio` 2.54 is
  verified.
  - Source: BRP-Rotax 915 iS A / iSc A product page,
    https://www.flyrotax.com/products/915-is-a-isc-a (read 2026-09-28).
  - The page gives no rpm for the 104 kW peak, so `rated_rpm` 5800 stays a
    project value, VERIFY against the Operators Manual.
  - v2 and v3 were trained with 105 kW (about 1 % shift in the
    expected-power residual features).
- **SIH "may utilize" list missing.** The list is not in the repository or
  in PRD-002/SRD-004. docs/SIH_COVERAGE.md omits it rather than guessing.
  Needs: the problem-statement text.
- **Not covered in this backend.** Dashboard UI, post-flight mission report
  (FUN-144), FMEA, federated learning (out of scope), edge-hardware
  resource measurements.

