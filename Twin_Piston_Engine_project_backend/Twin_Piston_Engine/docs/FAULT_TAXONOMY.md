# Fault taxonomy (Prompt 15)

The system uses one taxonomy, `FaultClass` in `src/core/provenance.py`, with
`FAULT_CLASS_COUNT = 14` classes. The simulator's `FaultMode` is built from it,
so both enums have identical members and IDs.

- **IDs 0–8 are unchanged**, so stored data, tests and any nine-class model
  stay valid. A legacy model's 9-probability output is still accepted.
- **Classes 9–13 are appended.** Their simulator-only IDs from Prompts 12 and
  13 were renumbered: INJECTOR_FAULT 11 → 9, FUEL_SYSTEM_FAULT 12 → 10,
  CHARGING_FAULT 9 → 12, BATTERY_DEGRADATION 10 → 13.

Classes are **root causes**. **Condition flags** are **symptoms** that can
co-occur with any class, and they are reported as multi-label
`condition_flags` on every classification. A flag is `None` when it can't be
evaluated, never a substituted `False`.

`NineClassFaultClassifier` is kept as a deprecated alias of `FaultClassifier`.

## Classes

The "L2 evidence" column is also the rule used by the fallback (`FaultClassifier.classify_fault_rule_fallback`).

| ID | Class | Physical perturbation in the simulator | L2 evidence (rule-fallback rule) |
|---|---|---|---|
| 0 | NOMINAL | — | no rule fires |
| 1 | MISFIRE | cylinder heat release lost; 0.5X vibration; crank slow-down | M-06 dual-channel gate CONFIRMED (`misfire_detected`) |
| 2 | DETONATION_KNOCK | ECU knock retard on the cylinder; head heat flux (CHT) | consistent ignition retard vs the ECU schedule, supported by EGT/CHT (Prompt 13); not from vibration |
| 3 | EXHAUST_VALVE_LEAK | cylinder EGT drop, power loss | EGT diagnosis WARNING+ with a cold cylinder (≤ −35 K vs the others) |
| 4 | INTAKE_BOOST_LEAK | MAP lowered before the air/fuel chain | MAP residual < −10 kPa. The pipeline has no MAP expectation (no boost reference), so this is not evaluable there. |
| 5 | OIL_DEGRADATION | oil pressure drop, oil temperature rise | LHI band WARNING/ALARM (M-10) |
| 6 | COOLING_FAULT | coolant loop: radiator blockage, pump degradation or coolant loss (Prompt 14) | coolant status WARNING+ (coolant residual, CHT − coolant) |
| 7 | BEARING_WEAR | BPFO impacts ringing an 800 Hz resonance | VHI band WARNING+ with the envelope term dominant (M-04, M-10) |
| 8 | SENSOR_FAULT | channel dropout or saturation flagged by the source | an instrumented channel is invalid in the record |
| 9 | INJECTOR_FAULT | one cylinder's injector flow coefficient: clog (−) or leak (+) | injector flow ratio identifies a cylinder (EGT vs command, Prompt 13) |
| 10 | FUEL_SYSTEM_FAULT | pump capacity loss, so rail pressure sags under demand | fuel-system status WARNING+ (rail residual, delivery ratio; SRD-FUN-084) |
| 11 | IMBALANCE | rotating propeller imbalance: 1X of the propeller shaft, lateral-dominant | M-12: propeller 1X holds > 50 % of lateral velocity energy and lateral/vertical 1X ratio > 1 (SRD-FUN-070) |
| 12 | CHARGING_FAULT | regulator setpoint drift, output collapse or open diode (Prompt 12) | ripple status WARNING+, or charging status WARNING+ while the regulator holds the bus |
| 13 | BATTERY_DEGRADATION | R_int growth, capacity fade | R_int status WARNING+, or charging status WARNING+ below cut-in (battery regime) |

**Order and legacy rules.**
- The rules are evaluated in the order SENSOR, MISFIRE, INJECTOR, FUEL_SYSTEM,
  KNOCK, CHARGING, BATTERY, IMBALANCE, BEARING, OIL, COOLING, EXHAUST_VALVE,
  BOOST_LEAK.
- The first rule that fires is the class. The others that fire are listed in
  `also_consistent_with`, and the rules that couldn't run are in
  `not_evaluable`.
- A rule gives no calibrated probability, so the fallback's `confidence` is
  None.
- Four legacy rules (vibration RMS, oil pressure, oil-temperature residual,
  MAP residual) apply only when the modern evidence for their class is
  absent. With the pipeline's states, the oil-temperature rule fired on a
  healthy engine: its expectation is biased by +30 K (OI-5).

## Condition flags

| Flag | Fires when | Evidence |
|---|---|---|
| overheating_trend | a Prompt 14 time-to-limit is valid and within `overheat.alert_horizon_s` (30 min) | `time_to_limit_s` per channel, horizon |
| combustion_instability | CSI band WARNING or ALARM (M-09) | CSI, band |
| lubrication_degraded | LHI band WARNING or ALARM (M-10) | LHI, band |

## SIH section C mapping

| SIH C line | Class / flag | Evidence parameters | Validation data |
|---|---|---|---|
| Misfire | 1 MISFIRE | crank CoV, half-order fraction (M-06), per-cylinder EGT | forward model only |
| Injector abnormalities | 9 INJECTOR_FAULT | injector flow ratio, EGT differential, commanded fuel, delivery ratio | forward model only |
| Cooling degradation | 6 COOLING_FAULT | coolant residual, CHT − coolant, coolant temperature | forward model only |
| Lubrication issues | 5 OIL_DEGRADATION + `lubrication_degraded` | LHI (pressure, viscosity, over-temperature), oil-pressure residual | forward model only |
| Sensor drift / failure | 8 SENSOR_FAULT | invalid channels and source flags; physical-range checks | forward model only (dropout/saturation; drift not simulated) |
| Combustion instability | `combustion_instability` (+ 1 MISFIRE, 2 DETONATION_KNOCK) | CSI (EGT dispersion, crank CoV, CHT slope); ignition retard | forward model only |
| Overheating trends | `overheating_trend` (+ 6 COOLING_FAULT) | Theil–Sen slope residual, time-to-limit and range per channel | forward model only |
| Abnormal vibration | 7 BEARING_WEAR, 11 IMBALANCE | envelope RMS, VHI (bearing); propeller-1X velocity fraction, lateral/vertical ratio (imbalance) | external data referenced as **DS-20/21** (bearing, imbalance). The datasets are not in this repository and have not been run. |

SRD-FUN-112 (nine different classes) is referenced in the prompt, but its text
is not in the repository, so its classes are not mapped here (open item).

## Measured confusion (rule fallback)

Test: `tests/scientific/test_fault_taxonomy.py`. Each fault mode runs for
120 s at 4000 rpm / 110 kPa, 1 Hz, onset 30 s, seed 42. The class is taken at
the last record. The rules were not tuned to force the diagonal.

| Injected | Predicted | Note |
|---|---|---|
| NOMINAL | NOMINAL | |
| MISFIRE | MISFIRE | EXHAUST_VALVE_LEAK also fires (cold cylinder) |
| DETONATION_KNOCK | DETONATION_KNOCK | flags: overheating_trend, combustion_instability |
| EXHAUST_VALVE_LEAK | **INJECTOR_FAULT** | A cold cylinder without a confirmed misfire reads as a leaking injector. EGT alone cannot separate the two; EXHAUST_VALVE_LEAK is in `also_consistent_with`. |
| INTAKE_BOOST_LEAK | **NOMINAL** | Not evaluable: no boost reference is instrumented. |
| OIL_DEGRADATION | OIL_DEGRADATION | flags: lubrication_degraded, overheating_trend |
| COOLING_FAULT | **NOMINAL** | The thermostat absorbs the blockage at cruise: coolant residual +7.5 K at steady state, band 8 K (OI-22). The `overheating_trend` flag fires. |
| BEARING_WEAR | BEARING_WEAR | |
| SENSOR_FAULT | SENSOR_FAULT | |
| INJECTOR_FAULT (clog 15 %) | **NOMINAL** | EGT is near its peak at cruise, so the flow ratio is only bounded (OI-21). Identified at 5800 rpm / 140 kPa. |
| FUEL_SYSTEM_FAULT | FUEL_SYSTEM_FAULT | |
| IMBALANCE | IMBALANCE | lateral 1X velocity fraction 0.775, lateral/vertical ratio 2.01 |
| CHARGING_FAULT | CHARGING_FAULT | |
| BATTERY_DEGRADATION | **NOMINAL** | R_int is not observable above cut-in (Prompt 12). Identified at idle (1800 rpm). |

9 of 14 are on the diagonal at the audit point. The two observability cases
are identified where they are observable (idle, rated power).

**Flags:**
- **Misfire run.** The misfire run does **not** raise
  `combustion_instability`: CSI is 0.0242, against a WARNING band starting at
  0.15 (OI-11). This is a strict xfail.
- **Independence.** The flags are independent of the class. COOLING_FAULT
  gives class NOMINAL with `overheating_trend` set, and an injected CSI
  WARNING sets the flag without changing the MISFIRE class.

**Cross-effect.** At severity 1 a propeller imbalance (0.394X of crank speed)
leaks into the 0.4–0.6X half-order band (0.036, above M-02's 0.03 WARNING).
This is misfire gate channel B. Channel A (crank CoV) does not confirm it, so
no misfire is declared.

## Where things are

| Item | Location |
|---|---|
| Taxonomy, flags | `src/core/provenance.py` (`FaultClass`, `FAULT_CLASS_COUNT`, `CONDITION_FLAGS`) |
| Classifier, rules, flags | `src/l3_ml/anomaly_fault.py` (`FaultClassifier`, `evaluate_condition_flags`) |
| Imbalance evidence | `src/l2_digital_twin/physics/M12_rotating_imbalance.py`, `VibrationState.prop_1x_*` |
| Advisory templates (US-801) | `src/l3_ml/advisory_explainability.py` (`FAULT_ADVISORY_TEMPLATES`, `CONDITION_FLAG_TEMPLATES`) |
| API | `/api/v1/diagnostics/fault`: `condition_flags`, `condition_evidence`, `rule_based` |
