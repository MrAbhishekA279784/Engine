# SIH coverage matrix (PS SIH26054, DRDO)

**Source of the item list.** The verbatim SIH problem statement is not in this
repository or in any file available to this work. The items below follow its
structure as restated in **SIH2026-DT-PRD-002 v2.0 §11**, "Coverage of the SIH
problem statement": sections A–F, deliverables, innovation areas and
technical expectations. Each grouped PRD line is split into one row per named
item. Section C lines are the eight listed in docs/FAULT_TAXONOMY.md.
SRD IDs are from SIH2026-DT-SRD-004 v1.0.

**The "may utilize" list is missing.** It is not reproduced in PRD-002, in
SRD-004 or anywhere in the repository. Its rows cannot be written without the
original text, so they are left out rather than guessed. Paste the list to
add them.

**Status rules (strict).**
- **COVERED:** an automated test in this repository exercises the item.
- **PARTIAL:** says what is missing.
- **NOT COVERED:** says why.

Unless marked otherwise, all measured evidence is **simulated**: forward
simulator, seeds recorded.

Test paths are abbreviated: `u/` = tests/unit, `i/` = tests/integration,
`s/` = tests/scientific. `e2e` = `s/test_physics_audit_e2e.py`.

## A. Virtual engine synchronised with live data; architecture

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Virtual engine model (physics twin) | FUN-020..036 | `src/l2_digital_twin/thermo_mechanical_twin.py`, `physics/` | `u/test_physics_fixes.py`, `u/test_sfc_load_eta_mech.py`, `s/test_module24_physics.py` | η_mech plausible across rpm; SFC hand-computed; brake power bounded | PARTIAL: the twin is not reconciled to the manufacturer power/fuel deck (OI-1; no deck in the repository) |
| Synchronised with live data | INT-001, INT-005 | `src/l1_data/adapters/` (simulator, CSV replay, CAN live), `src/api/v1/telemetry.py` | `u/test_module2.py::test_live_telemetry_adapter_with_mock_can`, `u/test_can_encoding_and_live_adapter.py::test_l2_handles_partial_live_record`, `i/test_module20_api.py::test_ingest_telemetry_valid` | Partial live CAN records are accepted; missing groups are invalid, not defaulted | PARTIAL: the live path is exercised with a mock CAN bus only; no real engine or ECU was connected |
| Modular layered architecture | CON-001, CON-010 | L1 → L2 → L3 → API; `src/audit/` boundary auditor | `i/test_l1_to_l2_boundary.py`, `i/test_module*_boundary.py` (AST import checks) | L1 never imports L2; L2/L3 have no simulator imports | COVERED |
| Real-time ingestion and streaming | INT-007, INT-008, PER-001 | `src/api/v1/telemetry.py`, `websocket.py`, `RunningPipeline` | `i/test_oi23_resolution.py::test_websocket_streams_every_record`, `i/test_module20_websocket.py`, `i/test_module23_performance.py` | WebSocket streams 12 of 12 records, then STREAM_COMPLETE (was the first 5; OI-23) | COVERED (simulated and replayed streams) |

## B. Parameters to be monitored

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| RPM | FUN-006 | `src/l2_digital_twin/sensor_inverse.py` | `test_crank_period_to_rpm`, `test_crank_kinematics` | rpm from the crank period; invalid when the crank period is absent (`i/test_oi23_resolution.py::test_phase_not_derivable_keeps_last_phase_with_quality_zero`) | COVERED |
| CHT | FUN-001, FUN-045 | `sensor_inverse.py`, `residual_engine.py` | `u/test_sensor_standards.py`, `test_invalid_cht_channel_gives_invalid_residual_not_zero` | NIST Type K check points; an invalid CHT gives an invalid residual, not 0 | COVERED |
| EGT (per cylinder) | FUN-001, FUN-040..044 | `sensor_inverse.py`, `egt_diagnostics` | `u/test_sensor_standards.py::test_type_k_nist_check_points`, `test_egt_spread_imbalance`, e2e `test_egt_dropout_channel_invalid_and_in_coverage` | EXHAUST_VALVE_LEAK: cyl 1 EGT − mean +12.8 → −167.2 K | COVERED |
| Oil pressure | FUN-004, FUN-052..054 | `lubrication` (M-10 LHI) | `test_m10_lhi_is_one_for_healthy_oil_at_any_temperature`, e2e `test_oil_degradation_lhi_down` | OIL_DEGRADATION: LHI 1.002 → 0.274 | COVERED |
| Oil temperature | FUN-003, FUN-050, FUN-051 | `sensor_inverse.py` (Pt100 CVD), `lubrication` (viscosity) | `u/test_sensor_standards.py::test_pt100_iec_check_points`, `test_known_value_viscosity_calculation` | IEC 60751 check points; viscosity vs reference | COVERED |
| Fuel flow | FUN-007, FUN-084, FUN-103 | `sensor_inverse.py`, `residual_engine.py` | `test_fuel_pulse_conversion`, `test_air_mass_flow_independent_of_fuel_flow`, `test_healthy_flow_and_delivery_ratios_within_3_percent` | Healthy delivery ratio within 3 % | PARTIAL: the healthy fuel expectation is biased by about −30 % vs the simulator (OI-5; strict xfails in `u/test_residual_expectations.py`) |
| Vibration signatures | FUN-060..072 | `src/l2_digital_twin/vibration*` | `u/test_vibration_burst.py`, `u/test_vibration_condition_features.py`, e2e `test_taxonomy_intended_evidence_moves[BEARING_WEAR, IMBALANCE]` | Envelope RMS 0.072 → 1.218; propeller-1X lateral fraction 0.0002 → 0.775 | PARTIAL: tested on the forward model only; the DS-20/21 bearing and imbalance datasets are not in the repository |
| Battery / alternator health | FUN-010, FUN-090..093 | `src/l2_digital_twin/electrical*` | `s/test_electrical_health.py` (19 tests), e2e `[CHARGING_FAULT, BATTERY_DEGRADATION]` | Charging residual −0.05 → −1.50 V; R_int 15.3 → 30.1 mΩ at idle | COVERED (R_int is observable only below cut-in: documented) |
| Injection timing parameters | FUN-012, FUN-095..099 | `sensor_inverse.py`, `injection*` | `s/test_injection_fuel_system.py::test_interval_to_angle_1000_us_at_5000_rpm_is_30_deg`, `test_knock_is_ignition_retard_plus_egt_pattern`, e2e `[DETONATION_KNOCK, INJECTOR_FAULT]` | Ignition residual 0.0 → −8.0°; injector 3 flow ratio +0.013 → −0.136 at rated power | COVERED (a single clogged injector is not identified at cruise: OI-21) |

## C. Fault types and predictions (14-class taxonomy plus 3 condition flags)

Every row is run for 120 s through L1 → L2 → L3 in the e2e tests
(Prompt 18). The class shown is the **rule-based** class, which is the demo
default.

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Misfire | FUN-080..083 | `misfire` gate, rule classifier | e2e `test_taxonomy_intended_class_is_produced[MISFIRE]`, `test_misfire_confirmed_and_csi_up` | Crank CoV 0.013 → 1.007; class MISFIRE | COVERED |
| Injector abnormalities | FUN-084, FUN-095..098 | `injection*`, rule classifier | e2e `[INJECTOR_FAULT]` (rated power), `[FUEL_SYSTEM_FAULT]` | Flow ratio −0.136; rail residual −326 → −90,436 Pa; classes INJECTOR_FAULT and FUEL_SYSTEM_FAULT | COVERED (not at cruise: OI-21) |
| Cooling system degradation | FUN-122, FUN-151 | `coolant*`, `overheat_trend` | e2e `[COOLING_FAULT]` evidence, `test_condition_flag_fires_on_its_run[overheating_trend]` | Coolant residual +10.3 K; overheating_trend fires | PARTIAL: at cruise the class stays NOMINAL (thermostat absorbs the blockage, strict xfail); only the flag fires |
| Lubrication issues | FUN-054 | `lubrication` (LHI) | e2e `[OIL_DEGRADATION]`, `test_condition_flag_fires_on_its_run[lubrication_degraded]` | LHI 0.274; class OIL_DEGRADATION; flag fires | COVERED |
| Sensor drift / failure | FUN-117, INT-003/004 | `telemetry_validator`, `sensor_inverse` validity | e2e `test_taxonomy_sensor_fault_evidence_is_the_invalid_channel`, `[SENSOR_FAULT]` | EGT1 dropout: residual invalid; class SENSOR_FAULT | PARTIAL: dropout and saturation are covered; slow sensor drift is not simulated, and the drift monitor (PSI) only advises |
| Combustion instability | FUN-085, FUN-122 | `misfire` (CSI), condition flags | `test_condition_flag_fires_on_its_run[combustion_instability]` (knock run) | CSI 0.009 → 0.408 | PARTIAL: it does not fire on the misfire run (OI-11, strict xfail) |
| Overheating trends | FUN-122 | `overheat_trend` (Theil–Sen on the residual) | `s/test_cooling_overheat.py::test_radiator_blockage_time_to_limit_valid_and_decreasing`, flag test | Time to limit valid and decreasing | COVERED |
| Abnormal vibration | FUN-067..071 | vibration features, rule classifier | e2e `[BEARING_WEAR]`, `[IMBALANCE]` | Classes BEARING_WEAR and IMBALANCE | PARTIAL: forward model only (no DS-20/21) |
| Other classes: detonation, exhaust valve leak, intake/boost leak | FUN-112 | rule classifier | e2e `[DETONATION_KNOCK]`, `[EXHAUST_VALVE_LEAK]`, `[INTAKE_BOOST_LEAK]` | Knock identified. Valve leak reads as INJECTOR_FAULT and boost leak as NOMINAL (strict xfails); their evidence moves (EGT −167 K; load 0.50 → 0.31) | PARTIAL: 2 of 3 not separable with the instrumented signals (no boost reference) |
| NOMINAL raises nothing | PER-017 | whole pipeline | e2e `test_taxonomy_nominal_raises_nothing[cruise, idle, rated]` | No class, flag, health alert or anomaly from 30 s on | COVERED |

## D. AI/ML layer

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Anomaly detection | FUN-111, FUN-120, PER-016/017 | `src/l3_ml/anomaly_fault.py`, `training.py` | `u/test_ml_models.py` | v2 Prompt 16 test: PR-AUC 0.841, FPR 0.046. Lifetime (life2) healthy windows: v3 0.114 (ML + anomaly) | PARTIAL: below the PR-AUC target of 0.90–0.93. v3 FPR on the Prompt 16 test is 0.201. Models are opt-in (ML off in the demo) |
| Fault classification (14 classes) | FUN-112, PER-018 | same | `u/test_ml_models.py`, `s/test_fault_taxonomy.py` | v2 macro-F1 0.767 (target 0.80–0.81); rules 9/14 on the diagonal at cruise | PARTIAL: below target; weak classes (fuel system, battery, boost leak) |
| Remaining useful life | FUN-130..137, PER-020/023 | `lifetime_rul.py`, `rul_estimation.py` | `u/test_lifetime_rul.py`, `i/test_oi23_resolution.py::test_lifetime_rul_is_plausible_along_a_degradation_history` | life2 test engines: MAE 35.9 / 38.4 / 103.4 h (0–50 / 50–150 / 150–400 h), 90 % coverage 0.543 | PARTIAL: the RUL model does not transfer to engine-to-engine variation. 78 % of healthy-engine windows get an ESTIMATED RUL; retraining on life2 is needed |
| Trend analysis | FUN-116 | `health_supervision` trend | `i/test_health_trend_and_endpoints.py` | NOMINAL 120 s ends STABLE; OIL_DEGRADATION ends DEGRADATION | COVERED |
| Recommendations / advisory | FUN-140..143 | `advisory_explainability.py` | `u/test_module19.py`, `s/test_fault_taxonomy.py::test_class_advisory_gives_cause_evidence_and_action` | Cause, evidence and action; probabilistic wording | COVERED |
| Adaptive learning | FUN-170..176 | `adaptation.py`, `drift_monitor.py`, `fleet_reference.py` | `u/test_adaptation.py` (11 tests) | Gate, bounded baselines, rollback and promotion gate tested. Lifetime evaluation: false alarms 0.158 → 0.161 | PARTIAL: built and tested, but REJECTED by its own acceptance rule (false alarms not reduced; PSI freezes it, OI-25); not applied live |
| Explainable AI | FUN-123 | `pred_contrib` evidence, rule evidence | `u/test_ml_models.py`, rule evidence asserts in `s/test_fault_taxonomy.py` | Top-5 feature contributions per ML prediction | COVERED |

## E. Simulation and replay

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Mission replay | FUN-153 | `replay_and_whatif.py` (ReplayEngine), `api/v1/replay.py` | `s/test_module24_simulation_replay.py::test_simulation_replay_engine_controls`, e2e `test_replay_pipeline_accepts_and_carries_derived_state` | Replay reproduces derived state | COVERED |
| What-if analysis | FUN-154 | `WhatIfEngine`, `api/v1/what_if.py` | `test_what_if_baseline_non_mutation`, `s/test_cooling_overheat.py::test_what_if_ambient_overrides_are_applied` | Same stack as live; baseline not mutated | COVERED |
| Environmental conditions (ISA deviation) | FUN-032, FUN-036, FUN-151 | `forward_simulator.py`, `environment` | `u/test_environment_and_defaults.py::test_hot_day_at_84_kpa`, `test_isa_plus_30_raises_cht_and_oil_expectations_at_the_same_power` | ISA +30 K raises the CHT and oil expectations | COVERED |
| High altitude | FUN-151 | simulator, ISA | `test_healthy_full_power_climb_to_2400_m_gives_no_time_to_limit_alert` (strict xfail), `test_pressure_altitude_outside_troposphere_is_none` | Climb to 2400 m; lifetime windows at 1500–7600 m (scripts only) | PARTIAL: no test above 2400 m, and the 2400 m climb test is a strict xfail (OI-22 CHT expectation) |
| Endurance missions | FUN-151 | lifetime fleet generator | none (scripts `generate_lifetime_dataset.py`, `headline_metrics.py`) | 700 engine-hours per healthy engine simulated in 20 s windows | PARTIAL: exercised by scripts, not by a test; no continuous multi-hour flight simulation |
| Hot weather | FUN-151, FUN-154 | `hot_weather_mission_profile` | `s/test_cooling_overheat.py::test_hot_weather_mission_healthy_engine_nothing_alarms` | Healthy engine, hot mission: no alarm | COVERED |
| Rapid throttle changes | FUN-151 | simulator thermal lag, trend window restart | `test_healthy_full_power_increase_gives_no_time_to_limit_alert`, `i/test_health_trend_and_endpoints.py::test_operating_point_step_restarts_the_window` | A step restarts the trend window; no false time-to-limit | COVERED |
| Flight phase in mission risk | FUN-136 | `MissionPhaseClassifier` in the pipeline | `i/test_oi23_resolution.py::test_flight_phase_is_derived_and_never_ground_in_flight` | TAKEOFF → CLIMB → CRUISE; never GROUND in flight (OI-23) | COVERED |

## F. Dashboard

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Engine health status | INT-010, INT-011 | `api/v1/engine_health.py` | `i/test_engine_health_running_pipeline.py` | API only | NOT COVERED: no dashboard UI in this repository (backend scope); the data API is tested |
| Alerts | INT-013, INT-014 | `api/v1/fault_anomaly.py`, `advisories.py` | `i/test_module20_api.py` | API only | NOT COVERED: UI not built; alert data served and tested |
| Efficiency trends | INT-012 | SFC, η in the derived state; `api/v1/history.py` | `u/test_sfc_load_eta_mech.py` | SFC derived; no trend view | NOT COVERED: no UI; SFC trend over one hour is not computed as a series |
| Maintenance advisory | FUN-141 | `advisory_explainability.py` | `u/test_module19.py` | Advisory text served | NOT COVERED as a dashboard (UI not built); advisory content is COVERED in D |
| Mission reports | FUN-144 | none | none | none | NOT COVERED: no post-flight mission report generator exists yet |

## Deliverables

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Working prototype | — | whole backend | full suite: 1068 passed, 21 strict xfails, 0 failed | See docs/PHYSICS_AUDIT_RESULTS.md §10 | PARTIAL: backend only; no UI |
| System architecture | CON-001 | docs, AST boundary tests | `i/test_*_boundary.py` | Boundaries enforced by test | COVERED |
| Simulation model | FUN-150..152 | `forward_simulator.py` | `s/test_module24_simulation_replay.py::test_forward_simulator_fault_injection_all_9_classes`, e2e full taxonomy | 14 classes as physical perturbations (SRD-FUN-152) | COVERED |
| ML module | FUN-110..123 | `src/l3_ml/` | `u/test_ml_models.py`, `u/test_adaptation.py` | See D | PARTIAL: targets not met; ML off in the demo |
| Dataset demonstration | DAT-004, DAT-006 | `scripts/generate_dataset.py`, `generate_lifetime_dataset.py` | `u/test_ml_models.py` (split and leakage tests) | Manifests with seed and SHA; whole-engine splits | COVERED (datasets are git-ignored; regeneration steps in ML_MODELS.md) |
| Documentation | QUA-006 | docs/ | none | PRD/SRD traceability, OPEN_ITEMS, this matrix | COVERED (documents; not testable) |

## Innovation areas

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| Physics-informed AI | FUN-100..109 | residual features into the ML | `u/test_residual_expectations.py`, `u/test_ml_models.py` | ML consumes L2 residuals only | PARTIAL: healthy residual bias (OI-5) remains |
| Explainable AI | FUN-123, FUN-140 | see D | see D | see D | COVERED |
| Edge analytics | FUN-160, FUN-161 | `src/edge_ground/` | `i/test_module21_partition.py`, `i/test_module21_failure_recovery.py::test_offline_buffering_when_link_disconnected` | Edge runs L1 and buffering; recovers after an outage | PARTIAL: not measured on edge hardware (PER-004/005 pending) |
| Secure telemetry | SEC-001..005 | `telemetry_security.py`, `telemetry_validator.py`, `ml_ops.require_bearer_token` | `u/test_module3.py`, e2e `test_tampered_record_is_rejected`, `u/test_adaptation.py::test_ml_endpoints_and_guarded_rollback` | Tampered and replayed packets rejected; rollback 401/503 without a token | COVERED |
| Autonomous advisory | FUN-143 | offline advisory templates | `u/test_module19.py` | No network needed | COVERED |
| Federated learning | — | none | none | none | NOT COVERED: declared not in scope in PRD-002 v2.0; the architecture permits it |

## Technical expectations

| SIH item | SRD requirement | Code module | Test | Measured evidence | Status |
|---|---|---|---|---|---|
| IC engine modelling | FUN-020..035 | L2 twin, simulator | `u/test_physics_fixes.py` (65 tests) | See A | PARTIAL: no manufacturer deck reconciliation (OI-1) |
| UAV propulsion | FUN-151 | propeller speed (gearbox 2.54, verified), mission phases | `test_propeller_speed_uses_configured_gearbox_ratio`, OI-23 tests | Rated 104 kW / max continuous 99 kW (manufacturer page) | COVERED |
| Sensor fusion | FUN-117, FUN-102 | cross-channel consistency, residuals | `test_cylinder_reference_excludes_the_cylinder_it_judges`, `test_rail_expectation_does_not_read_the_rail_pressure` | No sibling leakage | PARTIAL: the mutual-consistency sensor-drift test (FUN-117) covers dropout only |
| Embedded systems | CON-009, PER-004/005 | edge node | `i/test_module23_resource_limits.py` | Resource limits on the development PC | PARTIAL: no target hardware |
| CAN bus | INT-005 | `can_encoding`, live adapter | `u/test_can_encoding_and_live_adapter.py`, `u/test_raw_signals_prompt11.py::test_can_round_trip_0x103_to_0x107` | Frames 0x100–0x107 round trip | COVERED (mock bus) |
| AI/ML | FUN-110..123 | see D | see D | see D | PARTIAL (see D) |
| Visualisation | INT-010..018 | none | none | none | NOT COVERED: no UI in this repository |
| Simulation | FUN-150..156 | simulator | see E | see E | COVERED |
| Reliability engineering (FMEA) | — | none | none | none | NOT COVERED: FMEA not written (PRD-002: "to be added to the SRD") |

## Headline numbers (all simulated)

From `reports/headline_metrics.md`: the 60 held-out life2 test engines, with
engine-to-engine variation and benign ageing.

| (simulated) | rule-based alone | v3 alone | rules + v3 |
|---|---|---|---|
| Healthy-window false-alarm rate | 0.050 | 0.114 | 0.158 |
| Detection rate | 0.771 | 0.917 | 0.917 |
| Median lead time before failure | 174.7 h | 207.7 h | 224.8 h |

- **Rule false alarms.** 83 of the 84 rule false-alarm windows on healthy
  engines are INJECTOR_FAULT: per-cylinder EGT installation offsets look like
  an injector fault to the rules. The other 1 is COOLING_FAULT.
- **RUL band errors.** 35.9 / 38.4 / 103.4 h (simulated) with the life1 RUL
  model, which does not transfer to engine variation (see D).
- **v2 lead time is not a headline.** v2 alarms on 93 % of pre-fault windows.
