# Physics audit — end-to-end results

Branch `fix/physics-audit`. Updated after Prompt 9b (fixes for OI-12…OI-15).
Produced by `tests/scientific/test_physics_audit_e2e.py`
with `tests/scientific/physics_audit_harness.py`. No threshold was tuned to make a
test pass; indicators that did not behave as intended are listed in section 4 and
kept as strict xfails.

## 1. Method

- **Real pipeline, stateful per scenario:** simulator (`ScenarioRunner`, seed 42,
  1 Hz, 120 s, 4000 rpm, 110 kPa, 60 % throttle; fault onset at 30 s) →
  `RawSignalRecord` (with 2048 Hz accelerometer and crank bursts) → HMAC
  `PacketSigner.sign_record` → `TelemetryValidator.validate_packet` (signature
  verify, replay protection, range/stale checks) → `sensor_inverse` →
  `ThermodynamicMechanicalTwin` → `EGTDiagnosticsEngine` → `LubricationModel` →
  `VibrationProcessor` → `MisfireDetector` (dual-channel gate + CSI) →
  `ResidualEngine`. Indicators are read at t = 119 s and compared with NOMINAL
  under the same seed (same sensor noise).
- **Replay pipeline:** `PipelineReplayAdapter.process_sequence` over the same records.
- **API:** the FastAPI app with its replay engine loaded with the scenario;
  `/diagnostics`, `/diagnostics/{egt,lubrication,vibration,combustion}`,
  `/engine/health` checked for HTTP 200, NaN/Infinity, and `valid:false` values
  that are not `null`.
- **Missing burst:** NOMINAL records with all burst fields removed. **CSV replay
  (OI-3):** NOMINAL records written to CSV (the format has no burst columns) and
  parsed back by `CSVReplayAdapter`.

## 2. Scenario matrix (real pipeline, t = 119 s)

Overall = worst of combustion status, CSI band, LHI band, VHI band, EGT status
and lubrication status; INVALID/UNKNOWN anywhere → "invalid/reduced"; otherwise
record channel coverage or EGT coverage below 1 → "reduced coverage". Coverage
column: CSI terms / LHI components (always 0.67, viscosity not measured, OI-8) /
record measured-channel coverage.

| Scenario | Misfire verdict | CSI | LHI | VHI | Coverage CSI / LHI / record | Overall | Expected | Met? |
|---|---|---|---|---|---|---|---|---|
| NOMINAL | NORMAL | 0.0142 NORMAL | 0.999 NORMAL | 1.05 NORMAL | 1.00 / 0.67 / 1.00 | NORMAL | no WARNING/ALARM anywhere | Yes — none at any of 120 records |
| MISFIRE 0.3 | CONFIRMED (no cylinder localised) | 0.0172 NORMAL | 0.999 | 1.86 ALARM | 1.00 / 0.67 / 1.00 | ALARM | gate CONFIRMED, CSI up | Yes |
| MISFIRE 1.0 | CONFIRMED, cyl 1 | 0.0242 NORMAL | 0.999 | 2.90 ALARM | 1.00 / 0.67 / 1.00 | ALARM (combustion CRITICAL) | gate CONFIRMED, CSI up | Yes (CSI rise small, OI-11) |
| BEARING_WEAR 0.05 | NORMAL | 0.0142 NORMAL | 0.999 | 30.66 ALARM | 1.00 / 0.67 / 1.00 | ALARM (VHI) | envelope & VHI up, not confirmed | Yes |
| OIL_DEGRADATION 1.0 | NORMAL | 0.0142 NORMAL | 0.225 ALARM | 1.05 | 1.00 / 0.67 / 1.00 | ALARM (LHI; lubrication CRITICAL) | LHI down | Yes |
| COOLING_FAULT 1.0 | NORMAL | 0.0145 NORMAL | 1.007 NORMAL | 1.05 | 1.00 / 0.67 / 1.00 | NORMAL at t = 119 s | CHT residual up, no LHI false alarm | Yes (CHT +35 K; CSI ALARM for one window after the 35 K CHT step at onset) |
| INTAKE_BOOST_LEAK 1.0 | NORMAL | 0.0142 NORMAL | 0.999 | 1.05 | 1.00 / 0.67 / 1.00 | NORMAL at t = 119 s | MAP & air down, SFC up | Yes (EGT WARNING at the onset step) |
| SENSOR dropout EGT1 | NORMAL | 0.0144 NORMAL | 0.999 | 1.05 | 1.00 / 0.67 / 0.96 | reduced coverage | EGT1 invalid and visible | Yes (egt_cyl_1 invalid, EGT coverage 0.75) |
| SENSOR dropout MAP | NORMAL | 0.0008 NORMAL | 0.999 | 1.05 | 0.67 / 0.67 / 0.92 | reduced coverage | MAP invalid and visible | Yes (map/boost invalid; residuals invalid; air flow invalid) |
| NOMINAL, missing burst | INVALID | UNKNOWN | 0.999 | UNKNOWN | — / 0.67 / 0.84 | invalid/reduced | INVALID or reduced coverage | Yes |
| NOMINAL, CSV replay | INVALID | UNKNOWN | 0.999 | UNKNOWN | — / 0.67 / 0.84 | invalid/reduced | INVALID or reduced coverage | Yes |

All 120 records of every scenario were signed, verified and accepted; a record
tampered after signing is rejected.

## 3. Fault × intended indicator (Prompt 9)

| Fault | Indicator | NOMINAL | Fault | Moved as intended |
|---|---|---|---|---|
| MISFIRE 1.0 | misfire gate | NORMAL | CONFIRMED (crank CoV 1.01 %, half-order 0.66) | Yes |
| MISFIRE 1.0 | CSI | 0.0142 | 0.0242 | Yes (+0.010; stays NORMAL, OI-11) |
| BEARING_WEAR 0.05 | envelope RMS | 0.075 m/s² | 1.215 m/s² (×16) | Yes |
| BEARING_WEAR 0.05 | VHI | 1.05 | 30.66 | Yes |
| BEARING_WEAR 0.05 | misfire gate | NORMAL | NORMAL (not confirmed) | Yes |
| OIL_DEGRADATION 1.0 | LHI | 0.999 | 0.225 | Yes |
| COOLING_FAULT 1.0 | CHT residual | −45.6 K | −10.5 K (+35.0 K) | Yes (absolute offset: OI-5) |
| INTAKE_BOOST_LEAK 1.0 | MAP | 109.7 kPa | 74.7 kPa | Yes |
| INTAKE_BOOST_LEAK 1.0 | air mass flow | 0.0510 kg/s | 0.0342 kg/s | Yes |
| INTAKE_BOOST_LEAK 1.0 | SFC | 0.2368 kg/kWh | 0.2621 kg/kWh (+10.7 %) | Yes (Prompt 9b; magnitude still set by OI-1) |
| NOMINAL | all bands | — | NORMAL at every record | Yes (Prompt 9b) |
| COOLING_FAULT 1.0 | LHI (must not false-alarm) | 0.999 | 1.007 NORMAL | Yes (Prompt 9b; was 1.309 ALARM) |

## 4. Indicators that did NOT behave as intended

Resolved in Prompt 9b: NOMINAL start-up CSI alarm (OI-13), COOLING_FAULT LHI
false alarm (OI-14a), flat boost-leak SFC (OI-14b), MAP dropout accepted as
valid and lost channels invisible (OI-12), NaN in JSON mode (OI-15).

Still open:

| Finding | Measured | Cause | OI |
|---|---|---|---|
| CSI hardly moves under misfire | 0.0142 → 0.0242 | M-09 speed term = CoV/100 (not retuned) | OI-11 |
| Replay pipeline: CSI never available; L3 names no fault | CSI UNKNOWN; predicted NOMINAL, anomaly 0.000 in all scenarios | stateless per-record calls; L3 untrained | OI-16 |
| Healthy residual offsets | CHT −8.6…−14.3 %, fuel −29…−33 %, EGT −37…−256 K | expectations not calibrated | OI-5 |
| Boost-leak SFC magnitude | +10.7 % | twin brake power from a fixed 42 % indicated efficiency | OI-1 |

## 5. Replay pipeline and API

| Scenario | Replay: accepted | Health index | Predicted class / anomaly | API (6 endpoints) |
|---|---|---|---|---|
| NOMINAL | 120/120 | 0.905 | NOMINAL / 0.000 | 200, no NaN, invalid → null |
| MISFIRE 0.3 | 120/120 | 0.863 | NOMINAL / 0.000 | same |
| MISFIRE 1.0 | 120/120 | 0.825 | NOMINAL / 0.000 | same |
| BEARING_WEAR 0.05 | 120/120 | 0.905 | NOMINAL / 0.000 | same |
| OIL_DEGRADATION 1.0 | 120/120 | 0.789 | NOMINAL / 0.000 | same |
| COOLING_FAULT 1.0 | 120/120 | 0.912 | NOMINAL / 0.000 | same |
| INTAKE_BOOST_LEAK 1.0 | 120/120 | 0.920 | NOMINAL / 0.000 | same |
| SENSOR dropout EGT1 / MAP | 120/120 | 0.905 / 0.920 | NOMINAL / 0.000 | same |
| Missing burst / CSV replay | 120/120 | 0.905 | NOMINAL / 0.000 | same |

Every API response is HTTP 200 with no NaN/Infinity, and every `valid:false`
tagged value serialises as `null`.

## 6. Before / after for each fix

| Fix | Before | After |
|---|---|---|
| D-01/D-02 thermocouple & RTD (NIST ITS-90, IEC 60751) | 800 °C hot / 25 °C cold junction read 807.05 °C; 135 Ω → 364.06 K | 800.00 °C; 363.92 K; NIST/IEC points within 0.05 / 0.01 °C; sim→L2 round trip ≈1e-11 K |
| D-04 speed-density air flow | MAP 70→140 kPa: air ×1.00 (air = fuel × 14.7) | ×2.06 |
| D-03 λ derived | λ = 1.0 constant | derived; ring-wear case 1.049 → 0.839 |
| Bursts (B-01 root cause) | one sample/record: crest 1.0, no spectrum | 2048 Hz burst: crest 1.56, dominant order 1.995 |
| M-01…M-04 | not computed | kurtosis Gaussian ≈0 / impulsive >3; 0.5X misfire/nominal ×21 000–42 000; 2X 0.939–0.940 |
| M-05 Nyquist guard | none | 2–10 kHz band at fs 2048 rejected at load |
| Simulator fuel (5b) | fuel independent of rpm; IMEP up to 6.0 MPa | fuel 5800/2000 rpm ×2.97; IMEP ≤ 1.90 MPa; rated 100.2 kW |
| FMEP (5b, Barnes-Moss) | 1.15 bar at 5800 rpm (project quadratic) | 3.52 bar (1.47 at 2000) |
| η_mech on nominal simulator | 0.92–0.99 | 0.757–0.888 (110 kPa) |
| M-06 misfire gate (8b) | MISFIRE 1.0 UNCONFIRMED (2.0 % CoV limit) | CONFIRMED at severity 0.3–1.0 (0.30 % floor); BEARING_WEAR not confirmed |
| CSI (8b) | steady 0.198 WARNING; 20 s climb 2.09 ALARM | 0.011; 0.010 (thermal excluded in transient) |
| CHT slope noise (8b) | 0.0709 K/s (10 s) | 0.0145 K/s (30 s) |
| M-04 envelope band (8b) | 300–1000 Hz brick-wall; 4X leakage 0.03–0.07 m/s² | 650–1000 Hz tapered; leakage 0.00000 |
| Healthy VHI grid (8b) | 0.305–1.246 (single ref), 0.609–1.877 (rpm table) | 0.995–1.003 (0/27 outside 0.8–1.2) |
| Bearing envelope rise (8b) | ×6–15 | ×17–33 (overall RMS ×1.17–1.26) |
| LHI | not computed | healthy 1.000 at 35/85/115 °C; OIL_DEGRADATION 1.0: 0.225–0.517 |
| L1 invalid-flag propagation (9b, OI-12) | MAP dropout → map_pressure valid, 0 Pa | invalid; record coverage 0.92; every raw field's source flag reaches L2 |
| CSI start-up (9b, OI-13) | NOMINAL CSI ALARM at records 2–4 (max 1.11) | no WARNING/ALARM at any record |
| Simulator oil pressure (9b, OI-14a) | COOLING_FAULT LHI 1.309 ALARM | 1.007 NORMAL |
| Boost-leak fuel (9b, OI-14b) | SFC 0.237 → 0.237 | 0.2368 → 0.2621 (+10.7 %) |
| JSON null for invalid (9b, OI-15) | NaN kept in `model_dump(mode="json")` | null; all-invalid record encodes with allow_nan=False |
| SFC / load | not computed | SFC 0.0055 kg/s at 64 kW = 0.309; load 0.5 at 52.5 kW, 1.2 at 140 kW |

## 7. Test status vs the baseline

`docs/BASELINE_TEST_RESULTS.txt`: 545 passed. Now (after 9b): **777 passed,
7 xfailed, 0 failed.** No baseline test changed status (all 545 still pass). Baseline tests
whose expected values were edited during the audit (status unchanged, PASSED):

| File | Change | Reason |
|---|---|---|
| tests/unit/test_module5.py | EGT 1025.07 → 1018.01 K; CHT 588.92 → 592.20 K; oil 364.06 → 363.92 K | NIST ITS-90 / IEC 60751 instead of linear |
| tests/scientific/test_module24_sensor_conversion.py | linear formula → NIST inverse | same |
| tests/unit/test_module6.py | air density 1.15–1.25 → 1.1165; AFR 14.7 → 9.383; rpm-invalid AFR valid → invalid | speed-density air; λ derived |
| tests/unit/test_module9.py | single-sample RMS 2.5 valid → invalid/None | spectral features need a full burst window |
| tests/integration/test_module9_boundary.py | record given a 2048-sample burst | keep testing the valid path |
| tests/unit/test_module10.py | NORMAL → INVALID; cyl 1 CRITICAL+detected → WARNING, not detected | dual-channel gate needs both channels |
| tests/unit/test_module17.py | power > 50 → > 35 kW, torque > 100 → > 60 N·m | power now from air → fuel → IMEP − FMEP at unboosted MAP |
| tests/integration/test_module19_boundary.py | `SignalQuality(quality=)` → `score=` | `SignalQuality` now forbids unknown fields |
| tests/unit/test_physics_fixes.py | M-05 import path; M-04 carrier 600 → 800 Hz | M-05 moved to core; M-04 band now 650–1000 Hz |

## 8. Remaining xfails

| Test | OI |
|---|---|
| test_residual_expectations: nominal CHT/fuel < 5 % (3 operating points) | OI-5 |
| test_residual_expectations: nominal per-cylinder EGT < 10 K (3 operating points) | OI-5 |
| test_physics_audit_e2e::test_replay_classifier_names_misfire | OI-16 |

Removed in 9b because they now pass: start-up CSI (OI-13), COOLING_FAULT LHI
(OI-14a), boost-leak SFC (OI-14b), MAP dropout (OI-12), JSON mode (OI-15).

## 9. Open items

| OI | Title | Status |
|---|---|---|
| OI-1 | Fixed 42 % indicated efficiency in the twin | Open (needs Rotax deck) |
| OI-2 | Edge HTTP transport omits the signed timestamp | Open |
| OI-3 | CSV replay cannot carry bursts | Open; verified to report INVALID/reduced, never NORMAL |
| OI-4 | ML builder maps invalid derived values to 0.0 | Open |
| OI-5 | Healthy expectations disagree with the simulator | Open (needs flight data) |
| OI-6 | No airspeed channel for the CHT cooling credit | Open |
| OI-7 | Crank-CoV misfire threshold | Partly resolved (0.30 % floor, VERIFY) |
| OI-8 | No measured oil viscosity; LHI viscosity excluded | Open |
| OI-9 | VHI references and envelope band | Largely resolved (references simulator-derived) |
| OI-10 | CSI thermal term | Resolved for the simulator; verification pending |
| OI-11 | CSI crank term is CoV/100 | Known limit, not retuned |
| OI-12 | Validator dropped source invalid channels | Resolved (9b) |
| OI-13 | CSI start-up under-filled window | Resolved (9b) |
| OI-14 | Simulator oil pressure / boost-leak inconsistencies | Resolved (9b) |
| OI-15 | NaN in NormalizedSignalRecord JSON mode | Resolved (9b) |
| OI-16 | Replay pipeline stateless; L3 names no fault | Open |

## 10. Prompt 18: full-taxonomy verification, new parameters, ML metrics

### 10.1 New and corrected parameters: before and after

Source for the engine figures: BRP-Rotax 915 iS A / iSc A product page,
https://www.flyrotax.com/products/915-is-a-isc-a (read 2026-09-28).

| Parameter | Before | After | Basis |
|---|---|---|---|
| `engine.rated_power_kw` | 105.0 (unsourced) | 104.0 | "104 kW / 141 hp", manufacturer page |
| `engine.max_continuous_power_kw` | – | 99.0 | "99.0 kW / 135 hp, max. continuous power at 5500 1/min" |
| `engine.max_continuous_rpm` | – | 5500 | same |
| `engine.tbo_hours` | – | 1200.0 | "TBO 1,200 hrs" |
| `engine.gearbox_ratio` | 2.54, VERIFY | 2.54, **verified** | "propeller speed reduction gearbox i = 2,54" |
| `engine.rated_rpm` | 5800 | 5800 (VERIFY) | the page gives no rpm for the 104 kW peak |
| `mission.vertical_rate_window_s` | – (no phase derived) | 10 s, at least 3 points (VERIFY) | project choice: 10 records at 1 Hz |
| M08 fallback `ROTAX_915IS_RATED_POWER_KW` | 105.0 | 104.0 | same source |

Measured effect (NOMINAL, seed 42, 60 s, mean of the last 10 records):

| Point | Quantity | rated 105 kW | rated 104 kW |
|---|---|---|---|
| 4000 rpm / 110 kPa | expected brake power | 56.989 kW | 56.447 kW |
| | brake-power residual | −4.804 kW | −4.262 kW |
| | normalised load | 0.4959 | 0.5006 |
| | health index | 0.8993 | 0.9022 |
| 5800 rpm / 140 kPa | expected brake power | 105.123 kW | 104.122 kW |
| | brake-power residual | −3.844 kW | −2.845 kW |
| | health index | 0.9664 | 0.9716 |

The fuel-flow residual moves by less than 1e-4 kg/s at both points.

### 10.2 OI-23 changes: before and after

| Item | Before | After |
|---|---|---|
| Flight phase in mission risk | always GROUND (never derived) | derived per record; default mission TAKEOFF → CLIMB → CRUISE, never GROUND; UNKNOWN → last phase with quality 0 |
| WebSocket stream | first 5 records | every record, then STREAM_COMPLETE (12 of 12 in the test) |
| WebSocket client leaving early | n/a (5 records) | stream stops about 3 s after the client leaves (600-record replay) |
| WebSocket slot after the stream ends | never released on normal completion (the socket stayed in `active_connections`) | released in `finally` |
| RUL plausibility | not checked | baseline RUL never validated and its interval never calibrated (0.113 h on the 120 s OIL_DEGRADATION run); lifetime RUL in [0, 400] h and falling along a degradation history |

### 10.3 Full taxonomy end to end (tests/scientific/test_physics_audit_e2e.py)

- **Setup.** 120 s, 1 Hz, onset 30 s, seed 42, through L1 (HMAC and
  validator) → L2 → L3.
- **Operating point.** Cruise 4000 rpm / 110 kPa, except BATTERY at idle
  (1800 rpm / 60 kPa) and INJECTOR at rated power (5800 rpm / 140 kPa).
- **Evidence.** Mean of the last 10 records, fault vs NOMINAL at the same
  point.
- **Class.** The rule-based class at the last record.

| Class | Evidence | NOMINAL → fault | Class produced | Flags |
|---|---|---|---|---|
| NOMINAL (cruise, idle, rated) | – | – | NOMINAL from 30 s on; no flag, no health alert | none |
| MISFIRE | crank CoV | 0.013 → 1.007 | MISFIRE | none (OI-11) |
| DETONATION_KNOCK | ignition residual cyl 2 | 0.0 → −8.0° | DETONATION_KNOCK | combustion_instability, overheating_trend |
| EXHAUST_VALVE_LEAK | EGT cyl 1 − mean | +12.8 → −167.2 K | **INJECTOR_FAULT** (strict xfail) | none |
| INTAKE_BOOST_LEAK | normalised load | 0.499 → 0.305 | **NOMINAL** (strict xfail: no boost reference) | none |
| OIL_DEGRADATION | LHI | 1.002 → 0.274 | OIL_DEGRADATION | lubrication_degraded, overheating_trend |
| COOLING_FAULT | coolant residual | −6.6 → +3.6 K | **NOMINAL** (strict xfail: thermostat) | overheating_trend |
| BEARING_WEAR | envelope RMS | 0.072 → 1.218 | BEARING_WEAR | none |
| SENSOR_FAULT (EGT1 dropout) | EGT1 residual | −134 K → not derivable | SENSOR_FAULT | none |
| INJECTOR_FAULT (rated) | flow ratio cyl 3 | +0.013 → −0.136 | INJECTOR_FAULT | overheating_trend |
| FUEL_SYSTEM_FAULT | rail residual | −326 → −90,436 Pa | FUEL_SYSTEM_FAULT | none |
| IMBALANCE | propeller-1X lateral fraction | 0.0002 → 0.775 | IMBALANCE | none |
| CHARGING_FAULT | charging residual | −0.05 → −1.50 V | CHARGING_FAULT | none |
| BATTERY_DEGRADATION (idle) | R_int | 15.3 → 30.1 mΩ | BATTERY_DEGRADATION | none |

- **Classes.** 11 of 14 are produced at their observable point.
- **Evidence.** All 13 fault classes move their intended evidence.
- **Flags.** All 3 condition flags fire on their intended run and stay off
  on NOMINAL.
- **Phase.** Every record of every run is accepted, and none reads GROUND.

### 10.4 reports/ml_metrics.json (v2, re-evaluated in Prompt 18)

`scripts/evaluate_models.py --dataset v2 --version v2` was rerun after the
rated-power correction.

- **Test and holdout_op metrics: identical to Prompt 16.** They are scored on
  the frozen v2 dataset, whose features were computed with 105 kW.
  - Test: PR-AUC 0.841, FPR 0.046, recall 0.489, macro-F1 0.767, rules
    macro-F1 0.524.
  - holdout_op: 0.842 / 0.022 / 0.809.
- **Cruise-point section: moves slightly.** It re-simulates, so it uses the
  new value. All 14 classes are unchanged, for both rule and ML. Confidences
  and anomaly scores move in the third or fourth decimal (for example
  NOMINAL ML confidence 0.812 → 0.830).
- **Unchanged.** Channel-dropout and leakage-audit results.
- **v3 was not re-evaluated.** It is not adopted.

Headline numbers on the life2 held-out test engines are in
`reports/headline_metrics.md` and docs/SIH_COVERAGE.md (all simulated).

### 10.5 Test status vs the baseline (Prompt 18)

Full suite: **1068 passed, 21 xfailed, 0 failed** (1089 tests).

Compared with `docs/BASELINE_TEST_RESULTS.txt` (545 passed):
- **543 baseline tests** still pass under the same name.
- **2 were renamed** in Prompt 15 (`732b218`, 9 → 14 classes), and their
  successors pass:
  - `test_config.py::TestMLConfig::test_nine_fault_classes_defined` →
    `test_fault_class_count_defined`;
  - `test_provenance.py::TestFaultClass::test_nine_classes` →
    `test_fault_class_and_fault_mode_match_exactly`.
- **No baseline test fails or is xfailed.**

**The 21 strict xfails.** These are the 18 from before plus 3 new ones, the
full-taxonomy class checks for EXHAUST_VALVE_LEAK, INTAKE_BOOST_LEAK and
COOLING_FAULT. Each records its measured value.

**Tests changed in Prompt 18** (expectation edits caused by the rated-power
correction to 104 kW):

| Test | Change |
|---|---|
| `tests/unit/test_config.py` (2 assertions) | 105.0 → 104.0; also asserts max continuous 99 kW at 5500 rpm, TBO 1200 h and gearbox 2.54 |
| `tests/unit/test_physics_fixes.py::test_m08_load_and_bands` | load 1.0 at 105 → 104 kW |
| `tests/unit/test_sfc_load_eta_mech.py` | `RATED_KW` 105 → 104; load point 52.5 → 52.0 kW (half of rated) |

**Tests added:**
- 47 full-taxonomy tests in `tests/scientific/test_physics_audit_e2e.py`;
- 7 in `tests/integration/test_oi23_resolution.py`.
