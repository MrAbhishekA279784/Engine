# L3 models (Prompt 16)

All numbers here come from **simulator data** (provenance SIMULATED, PRD
4.4). Measured metrics are in `reports/ml_metrics.json` and
`reports/ml_metrics.md`. The PRD's "achieved" PR-AUC 0.93 and F1 0.81 were
not reproducible from anything in the repository; they are treated as
targets below.

## Pipeline

1. **Dataset** (`scripts/generate_dataset.py`).
   - **Trajectories.** 960 seeded trajectories of 240 s at 1 Hz, run through
     the real L1 → L2 → L3 pipeline: 60 per fault class and 180 NOMINAL. Each
     trajectory is one engine instance with a random mission: altitude 0–5 km,
     ISA −10 to +35 K, cruise, climbs, abrupt throttle steps and gradual
     throttle ramps.
   - **Faults.** Each fault has a random onset (30–150 s) and progression rate.
     Onset and rate are fields of `FaultScenarioConfig` (`severity_at`), so the
     fault stays a physical perturbation in the forward model.
   - **Features.** Schema 2.0.0 (`src/l3_ml/ml_features.py`): 72 L2/L3
     outputs, NaN when not derivable. There are no raw channels, no ground
     truth, no fault configuration and no timestamp, sequence or onset fields
     (test: `tests/unit/test_ml_models.py`).
   - **Output.** Parquet under `data/datasets/<version>/` (git-ignored) with a
     manifest recording the seed, code SHA, config hash, feature schema,
     dataset SHA-256 and the trajectory → split map.
2. **Split** by whole trajectory (SRD-DAT-004). Trajectories whose cruise rpm
   is 4400–4600 are held out entirely (`holdout_op`); the rest are split
   60/20/20 per class into train/val/test. No trajectory is in two splits
   (tested).
3. **Training** (`scripts/train_models.py`) uses train and val only.
   - **Anomaly detector.** IsolationForest (median imputer plus missing
     indicators) trained on NOMINAL trajectories. The threshold is the 95th
     percentile of validation NOMINAL scores (FPR ≤ 5 %).
   - **Classifier.** LightGBM over 14 classes, balanced class weights, early
     stopping on validation (stopped at iteration 120).
   - **Explanations.** LightGBM `pred_contrib` (TreeSHAP) gives the top five
     feature contributions per prediction in
     `FaultClassificationResult.evidence`. The `shap` package was not
     installed (user choice).
   - **Bundles.** Each is saved with joblib as a dict carrying its
     `ModelMetadata`, feature list, dataset hash, seeds, class count 14 and
     version. `MLModelLoader` recognises the bundles.
4. **Evaluation** (`scripts/evaluate_models.py`) uses the test split and,
   separately, `holdout_op`.

## Enabling the models

**Demo default: ML off (rules and physics).** `ml.model_version`,
`ml.rul_model_version` and `adaptation.fleet_reference_version` are all
None/unset in `config/default.yaml`. In that state:
- L3 reports MODEL_UNAVAILABLE for the ML result, and the rule-based
  diagnosis, condition flags, health index and physics run;
- RUL is the baseline estimator, labelled UNVALIDATED.

The rule-based result is always reported beside the ML result
(`rule_fault_result`, API `rule_based`). ML never replaces it.

**v3 (Prompt 17b) exists but is NOT adopted.** It failed the pre-registered
comparison with v2 (see the Prompt 17b section below), so v2 stays the version
to enable.

### ML demo: exact steps

Datasets and `*.joblib` bundles are git-ignored, so they are rebuilt from
seeds. Run from the repository root.

1. **Install the training extras** (scikit-learn, lightgbm, pandas, pyarrow):

   ```bash
   uv sync --extra train
   ```

2. **Short-trajectory dataset v2.** 960 trajectories, seed 20260928, about
   15 min on 16 workers:

   ```bash
   uv run python scripts/generate_dataset.py --version v2 --seed 20260928
   ```

3. **v2 models and Prompt 16 test report:**

   ```bash
   uv run python scripts/train_models.py --dataset v2 --version v2
   uv run python scripts/evaluate_models.py --dataset v2 --version v2
   ```

4. **Optional lifetime RUL.** life1 (seed 20260929), then the RUL bundle:

   ```bash
   uv run python scripts/generate_lifetime_dataset.py --version life1
   uv run python scripts/train_rul.py --dataset life1 --version life1
   ```

5. **Optional v3** (needs step 2). life2 is the train and val engines only,
   with engine variation; about 15 min:

   ```bash
   uv run python scripts/generate_lifetime_dataset.py --version life2 --variation --per-record --skip-test
   uv run python scripts/train_models.py --dataset v2 --extra-dataset life2 --version v3 --no-rul
   uv run python scripts/evaluate_models.py --dataset v2 --version v3 --report-dir reports/v3
   ```

6. **Enable the models** by editing `config/default.yaml` under `ml:`. Use v2
   (the adopted version) unless you are deliberately showing v3:

   ```yaml
   ml:
     model_version: v2          # or v3 (not adopted: Prompt 16 test FPR 0.201)
     rul_model_version: life1   # optional; also needs engine.engine_hours_at_install
   ```

   Then restart the API. `GET /api/v1/ml/models` shows the active version.

7. **Back to the demo default.** Remove those keys (or set them to null).

**Note (Prompt 18).** `engine.rated_power_kw` changed from 105 to 104 kW, per
the manufacturer page. The expected brake power, and therefore the
fuel-flow expectation and the ML features, shift by about 1 %.
- **Existing bundles.** v2 and v3 were trained with 105 kW. Regenerating
  from the steps above gives datasets whose SHA-256 differs from the
  committed manifests.
- **Effect on v2.** `reports/ml_metrics.json` was re-evaluated with the new
  value; before/after is in docs/PHYSICS_AUDIT_RESULTS.md §10.

## Results (v2, test split)

| Metric | Measured | Target | |
|---|---|---|---|
| Anomaly PR-AUC | 0.841 | 0.93 | **missed by 0.089** |
| Anomaly FPR | 0.046 | ≤ 0.05 | met |
| Classifier macro-F1 | 0.767 | 0.81 | **missed by 0.043** |
| Held-out operating point: PR-AUC / FPR / macro-F1 | 0.842 / 0.022 / 0.809 | | macro-F1 missed by 0.001 |
| Rules on the same test records (macro-F1) | 0.524 | | |
| Median warning lead time | 65 s (62 trajectories warned before full severity, 36 not) | | |
| RUL baseline MAE, ≤ 50 h band | 803 h | | see below |

- **Weak classes.** Per-class F1 is lowest for FUEL_SYSTEM_FAULT (0.13),
  BATTERY_DEGRADATION (0.12) and INTAKE_BOOST_LEAK (0.26); COOLING_FAULT is
  0.66. All four are weakly or not observable in much of the mission space.
- **RUL.** On the Prompt 16 dataset the faults develop over minutes, so RUL
  could not be evaluated there. It is now evaluated at lifetime scale; see the
  next section.

## v1 rejected: dataset confound found by the leakage audit

The v1 dataset had abrupt throttle steps (which restart the trend windows)
and climbs, but no gradual power changes. At the cruise point the v1
classifier identified INTAKE_BOOST_LEAK (confidence 0.86), which the rules
cannot see because no boost reference is instrumented. Permutation importance
showed it relied on the CHT, coolant and EGT trend slopes and on BMEP. In v1,
a gradual MAP decline at constant rpm appeared only in boost-leak
trajectories, so the model had learned "gradual power reduction = boost
leak".

v2 adds gradual throttle ramps to every trajectory's mission. The v2 model
says NOMINAL for boost leak at cruise, like the rules, and its boost-leak F1
fell from 0.37 to 0.26. The remaining validation recall (0.35) rests on
diffuse, weak features (largest permutation drop 0.066). The class is flagged
unreliable. The v1 metrics are kept in `reports/ml_metrics_v1.json`.

## Cruise point: ML vs rules (4000 rpm / 110 kPa, 120 s)

Both agree on 10 of 14 classes. They differ on:

| Injected | Rules | ML (v2) | Explanation (permutation importance, validation split) |
|---|---|---|---|
| EXHAUST_VALVE_LEAK | INJECTOR_FAULT | EXHAUST_VALVE_LEAK | Separates a valve leak from a leaking injector with evidence beyond the cold cylinder. |
| INTAKE_BOOST_LEAK | NOMINAL | NOMINAL | Unobservable without a boost reference: both say NOMINAL (after the v1 fix). |
| COOLING_FAULT | NOMINAL | COOLING_FAULT | CHT − coolant (drop 0.46) and the coolant residual. The healthy coolant residual is −6.6 K (OI-5 bias), so the +7.5 K fault is a 14 K shift, which the rules' absolute 8 K band misses. Physical signal; accepted. |
| INJECTOR_FAULT (clog, λ ≈ 1) | NOMINAL | INJECTOR_FAULT | EGT spread (0.67) and fuel delivery ratio (0.42). A 15 % clog leaves the measured fuel 3.8 % below the command and one cylinder 32 K hot. Both are physical, below the rule bands. Accepted. |
| BATTERY_DEGRADATION (above cut-in) | NOMINAL | NOMINAL | R_int is unobservable while the regulator holds the bus: both say NOMINAL; test recall 0.14. |

## Channel dropout (v2, test split, one sensor group removed at a time)

The classifier scales its confidence by the importance-weighted feature
coverage. When a sensor group is fully missing and that happened in fewer
than 1 % of training rows, the input is outside the training distribution: the
models abstain (MODEL_UNAVAILABLE, reason in the result) and the rule result
applies.

This rule was added after the v1 measurement. There, removing the EGT group
gave accuracy 0.04 at mean confidence 0.75, with 78 % of records wrong at
≥ 0.7 confidence; the oil group gave 77 % and the electrical group 69 %.

| Group removed | v2 result |
|---|---|
| egt, oil, vibration, fuel, electrical, coolant, crank | abstain (100 % of records) |
| cht | answers: 9.8 % confident-wrong vs 7.1 % with all groups; confidence **not** reduced (the CHT features carry little importance) |
| ignition | answers: 6.3 % confident-wrong |

## Lifetime RUL (Prompt 16b)

### Method

The method is ported from `aerotwin_ml` and driven by this repository's
simulator and pipeline. `aerotwin_bundle.joblib` was not loaded and its
EVALUATION.md figures were not used. Code: `src/l3_ml/lifetime_rul.py`,
`scripts/generate_lifetime_dataset.py`, `scripts/train_rul.py`.

- **Fleet.** 300 engines: 60 healthy (700 h each) and 20 for each of 12
  progressive fault classes. SENSOR_FAULT is excluded because it is a step
  dropout here.
- **Fault life.** Onset 60–300 h, 150–450 h from onset to failure, severity
  following a power law with exponent 1.4–2.6.
- **Monitoring windows.** One 20-s window (1 Hz) every 5 engine hours, at a
  sampled flight condition (cruise, climb or descent; altitude; ISA −20 to
  +40 K). Each window runs through the real simulator and L1 → L2 → L3
  pipeline with the fault at its current severity. The fleet has 31 573
  windows.
- **Split.** By whole engine, stratified by class: 180 train, 60 validation,
  60 test.
- **Severity model.** LightGBM regression from the 72 ML features to
  severity. MAE on test engines: 0.066.
- **RUL model.** LightGBM (Huber loss) on the severity-estimate history only.
  The history is indexed by monitoring window. Engine hours are only the time
  axis of the targets and the sampling cadence. Throttle, airspeed,
  electrical load, flight phase and `t_h` are not inputs (tested).
- **Training data.** The RUL model is trained on out-of-fold severity
  estimates (3 folds by engine). Its 90 % intervals come from per-band
  5–95 % error quantiles on validation engines.
- **Status.** NO_DEGRADATION, INDETERMINATE (fewer than 12 windows since
  degradation was first seen) or ESTIMATED.

### Results on held-out test engines (windows with status ESTIMATED)

| True RUL | n | Lifetime model MAE | 90 % coverage | Baseline MAE (same windows) |
|---|---|---|---|---|
| 0–50 h | 423 | 19.0 h | 0.87 | 363.8 h |
| 50–150 h | 542 | 25.6 h | 0.88 | 304.6 h |
| 150–400 h | 494 | 84.1 h | **0.52** | 136.1 h |

- **Coverage below target.** Overall interval coverage is **0.755** against
  the 0.90 target. The 150–400 h band under-covers: 0.52 on test against 0.93
  on validation, so the far-band calibration does not transfer between
  engine sets.
- **Where the model reports a number.** On faulty test windows the status is
  NO_DEGRADATION 59 %, INDETERMINATE 11 % and ESTIMATED 30 %. On healthy test
  engines, 5.4 % of windows still get an ESTIMATED RUL (false estimates).
- **Adopted.** It beats the baseline in every band with data. That was
  decided on validation engines (15.7 / 27.5 / 37.7 h against
  359.6 / 300.7 / 163.2 h) and holds on test.
- **Baseline method.** The baseline numbers apply the existing estimator's
  method: a least-squares health-index trend projected to HI 0.2, capped at
  2000 h. The live `RULEstimator` itself cannot run across flights, because
  its history resets on any gap longer than 300 s. Across 5-h windows it only
  returns INSUFFICIENT_HISTORY.
- **Report.** `reports/rul_metrics.json` and `reports/rul_metrics.md`. All
  data are SIMULATED.

### Live use

The lifetime model is used only when `ml.rul_model_version` is set, the
bundle exists, and the engine-hours channel is valid
(`engine.engine_hours_at_install` configured). `RULState.validated` is True
only for ESTIMATED.

In every other case the baseline is reported with `validated=False`. The API
then returns `status: "UNVALIDATED"` with a note (`/engine/rul`), and
`rul_validated: false` (`/mission/risk`, WebSocket). With the default config,
live RUL is therefore UNVALIDATED.

### aerotwin_ml modules

| Module | Use |
|---|---|
| `sim/faults.py` (FaultProcess) | **Adapted.** Onset, life and power-law severity became `FaultLife`. The class-to-effect table was dropped: this backend's forward model applies each fault physically through `FaultScenarioConfig`. The sensor-drift and regulator/rectifier sub-modes were dropped; this backend's own sub-modes are used. |
| `sim/fleet.py` | **Adapted.** Kept: engine lives, one window per flight period, a seed sequence per engine, and a whole-engine stratified split. Dropped: its analytic `steady_state`, `measure`, burst synthesis, lag model and `EngineParams` manufacturing variation (this simulator has none), and the throttle, airspeed, electrical-load, phase and `t_h` columns. Windows run this backend's `ScenarioRunner` and pipeline at a fixed 5-h cadence. |
| `models/rul.py` | **Adapted.** Kept: history features, Huber LightGBM, the 400 h cap, per-band 5–95 % calibration and the status logic. Changed: the history is indexed by window, so the slopes are per window and `hours_since_degradation` was removed. |
| `models/health.py` | **Adapted** as `SeverityModel`: a LightGBM severity regressor on this backend's 72 features. Subsystem attribution was dropped. |
| `scripts/train.py` (out-of-fold severity, 3 folds by engine), `scripts/evaluate.py` (MAE per true band, coverage) | **Adapted.** |
| `sim/engine.py`, `sim/sensors.py`, `sim/vibration.py`, `dsp.py`, `features.py`, `models/expectation.py`, `models/anomaly.py`, `models/classifier.py`, `pipeline.py`, `registry.py`, `service.py`, `backend_adapter.py`, `taxonomy.py`, the tests | **Dropped.** This backend has its own simulator, L2 features, models and taxonomy. |
| `models/aerotwin_bundle.joblib`, `reports/EVALUATION.md` | **Not used.** |

## Why removing the CHT group still answers confidently

The abstention rule treats a sensor group as out of distribution when it was
fully missing in fewer than 1 % of training rows. The CHT group (`res_cht`,
`trend_cht`) was fully missing in **1.06 %** of training rows, just above that
threshold. 97 % of those rows are MAP-dropout SENSOR_FAULT records, where the
CHT residual is NaN because the operating point is invalid, not start-up
windows. So "CHT missing" was seen in training only together with a MAP
failure, which is not the situation of a lone CHT sensor failing.

In addition, the two CHT features carry only **2 %** of the classifier's
importance (`res_cht` 0.5 %, `trend_cht` 1.5 %). The importance-weighted
coverage therefore scales the confidence down by at most about 2 %.

The fix would be to judge the missingness pattern (which groups are missing
together) rather than each group on its own (OI-24). It was not tuned here.

## Drift monitoring and gated adaptation (Prompt 17)

**Status: built, tested, and REJECTED by the lifetime guardrail, both in
Prompt 17 (target zero, v2) and in Prompt 17b (target = fleet healthy
residual, v2 and v3).** No live path applies an adapted baseline. The running pipeline always uses the fleet
baseline `fleet-0` unless a caller passes a baseline in explicitly.

### Drift (`src/l3_ml/drift_monitor.py`)

- PSI per residual channel (17 channels) against a reference built from
  healthy records. The bands come from config (`drift.psi_moderate` 0.10,
  `drift.psi_significant` 0.25).
- `DriftState` is on every pipeline step and served by `GET /ml/drift`.
- **Noise floor.** With no real change, PSI still averages about
  (bins − 1)·(1/n_ref + 1/n_win). With 10 bins and 20-record windows a steady
  healthy run read PSI 3.8. The config therefore uses 5 bins and at least 200
  records for both the reference and the window (floor about 0.04).
- **Excluded from the overall score.** The running-median channels
  (`fuel_delivery_ratio`, `injector_flow_ratio_cyl*`) are autocorrelated, so
  consecutive records are not independent samples. They are still reported
  per channel.
- **Healthy vibration.** On a healthy steady 500 s run most channels read
  0.01–0.06, but `vibration_rms` reads 0.104. This is ADC quantization (few
  distinct values), and it sits just over the moderate line (OI-25).
- **Live API.** The API has no stored reference. It builds one itself from
  the first 200 healthy records, and until then `reference_status` is
  `BUILDING`.

### Eligibility gate (`record_eligibility`, `flight_eligibility`)

A flight may be used for adaptation only if **every** record is healthy. Any
one of these makes a record unhealthy, and each reason is logged:

- rejected by L1 validation;
- anomaly alarm;
- health index invalid or below `health.healthy_threshold`;
- health status not NORMAL, or degradation above WATCH;
- rule-based class not NOMINAL;
- ML class not NOMINAL with confidence above
  `adaptation.fault_confidence_threshold` (0.70);
- any condition flag set;
- any core residual channel invalid (a sensor group missing).

The gate has no bypass.

### Per-engine baseline (`AdaptationManager`)

- **Correction.** `expected' = expected + a + b·(expected − ref)` on EGT
  cyl 1–4, CHT, coolant, fuel flow, oil pressure, oil temperature, vibration
  and brake power.
- **Target (Prompt 17b).** The fleet's healthy residual at the record's
  operating point (`src/l3_ml/fleet_reference.py`), not zero. In Prompt 17
  the target was zero.
- **Commissioning.** Needs 3 eligible flights and at least 200 records, and
  a fleet reference (`adaptation.fleet_reference_version`; without one,
  nothing is commissioned).
  - `a` is bounded by `commissioning_max_offset`. Since 17b this bound is on
    the deviation from the fleet reference (EGT 60 K), no longer on the
    expectation bias (350 K).
  - `|b|` is at most `max_linear_gain` (0.5).
- **Later updates.** Offset only, and only on an eligible flight whose pooled
  drift is MODERATE.
  - Each update is limited to `rate_limit_per_flight`.
  - The total change from the commissioning offset is capped by
    `max_total_from_commissioning`.
- **SIGNIFICANT drift.** Adaptation freezes, and an advisory is added:
  "data distribution has shifted; possible sensor drift or unmodelled
  condition". It is cross-checked against a SENSOR_FAULT class.
- **Versions.** Every baseline is a new, immutable version (`<engine>-vN`)
  that records its parent. `rollback` restores the previous version in one
  call.
- **Log.** Every decision (commissioning, refused, frozen, no change, adapted,
  rollback) goes to an append-only, hash-chained JSONL log
  (`AdaptationLog.verify`, SRD-QUA-003).
- **Versioning fields.** `HealthState`, `FaultClassificationResult` and their
  API responses carry `baseline_version`, `adapted_baseline`,
  `model_version` and `hours_since_baseline`.

### Challenger and promotion (`ModelRegistry`, `train_challenger`, `promotion_gate`)

- **Trigger.** A challenger is retrained after `challenger_after_flights`
  eligible flights. It adds the eligible rows to the Prompt 16 train split,
  and `train_challenger` asserts that no test-split row is used.
- **Promotion.** Only if, on the FROZEN Prompt 16 test split, macro-F1 and
  anomaly PR-AUC drop by at most 0.02 each and FPR is at most 0.05.
- **Registry.** Kept in `models/registry.json`. Model rollback is one call.
  Bundle directories are immutable, and the tests check the frozen bundle
  hash.
- **Not automated.** The retrain trigger is counted, but nothing yet runs the
  retrain end to end on a schedule (OI-25).

### API

- `GET /ml/models`: versions, model cards and active flags.
- `GET /ml/drift`: drift state for the replayed stream.
- `POST /ml/rollback` (`{"target": "model" | "baseline", "engine_id"}`):
  - requires `Authorization: Bearer <token>`;
  - the token is read from the environment variable named by
    `security.api_bearer_token_env_var` (`TWIN_API_BEARER_TOKEN`);
  - if that variable is unset the endpoint returns 503 (fail closed); a
    wrong token returns 401.
- **Bearer token was not pre-existing.** The prompt called it "existing", but
  the repo had no bearer-token option; this minimal one was added.
- **RUL interval labelling.** Beyond 150 h the RUL interval is labelled
  `interval_calibrated: false`, with a note that measured 90 % coverage there
  is 0.52. The baseline RUL estimator's ±2 SE interval is always labelled not
  calibrated.

### Lifetime guardrail, Prompt 17 (life1, v2, target zero; superseded report)

**Setup.**
- Engines: the 60 held-out test engines of the lifetime fleet life1 (12
  healthy, 48 with a slowly developing fault).
- Runs: each engine flown without and with adaptation, using the frozen v2
  models.
- Benign ageing: a per-engine EGT rise of 3–6 K/100 h and CHT rise of
  1–2.5 K/100 h (VERIFY; project assumption, not a Rotax figure).
- Window alarm: anomaly on at least half of the window's records, or a
  non-NOMINAL rule class, or an ML class with confidence above 0.7, at the
  last record. Condition flags block adaptation but are not counted as an
  alarm here.
- Detection: two consecutive alarm windows after onset.

| | without | with |
|---|---|---|
| (a) healthy windows with an alarm | 0.499 | **0.740** |
| – anomaly / rule / ML | 0.394 / 0.001 / 0.183 | 0.705 / 0.001 / 0.231 |
| (b) detection rate | 1.000 | 1.000 |
| (b) median lead time | 280.1 h | 284.3 h |
| pre-onset windows with an alarm | 0.386 | 0.447 |

Adaptation decisions over all flights:

| Decision | Flights |
|---|---|
| Refused by the gate | 3432 |
| Commissioning pending | 483 |
| Commissioned | 37 |
| Frozen | 2565 |

**Verdict.** (b) is not worse, but (a) is worse: adaptation raises false
alarms instead of reducing them. **Adaptation is rejected.** Nothing was tuned
after seeing these numbers.

**Why (a) fails.** Traced on one healthy test engine:
1. With the fleet baseline, the healthy EGT residual is not near zero. It
   depends on the operating point: about −163 K in cruise, −80 K in descent
   and −228 K in climb. This is the known OI-5 bias.
2. The frozen v2 models were trained on these fleet-baseline residuals, so
   for them "healthy" means those offsets.
3. Commissioning, as specified, drives the engine's residuals to about zero.
   A linear-in-expected correction cannot represent an operating-point bias:
   `b` hits the −0.5 limit on every channel and `a` is about −176 K for EGT.
4. The corrected residuals (+10 to +45 K) are out of distribution for the
   frozen models. From the first adapted flight on, every window raises an
   anomaly alarm and a SENSOR_FAULT prediction. Drift is then SIGNIFICANT, so
   adaptation freezes, as designed.

**The detector is not usable either.** Without adaptation, 50 % of healthy
lifetime windows already raise an alarm. The v2 anomaly model was trained on
the Prompt 16 scenario set, and it does not generalise to the lifetime
fleet's ISA and altitude range. That is its own problem, separate from
adaptation (OI-25).

**Few engines ever commission.** Because of those false alarms the gate
refuses most flights, so only 37 of 60 engines ever commissioned.

**What a working design needs.** The correction target should be the fleet's
healthy residual at the same operating point, not zero. Alternatively, fix
OI-5 first so that healthy residuals are near zero, then retrain the models.

## v3 and the fleet-referenced adaptation target (Prompt 17b)

Reports: `reports/v3/ml_metrics.md` (Prompt 16 test, v3) and
`reports/adaptation_eval.md` (lifetime, v2 and v3, without and with
adaptation). All data are SIMULATED.

### Data: life2

- **Same plans and split.** `scripts/generate_lifetime_dataset.py --version
  life2 --variation --per-record --skip-test` uses the same engines, window
  plans, seeds and whole-engine split as life1.
- **Engine-to-engine variation and benign ageing.** These are physical
  simulator parameters drawn per engine (`engine_variation`), all VERIFY
  (project assumptions, not Rotax figures):
  - per-cylinder EGT installation offset, N(0, 8 K) clipped at ±20 K;
  - indicated efficiency ×(1 ± 2 %);
  - peak volumetric efficiency ×(1 ± 1.5 %);
  - friction constant ×(1 ± 5 %);
  - EGT ageing 3–6 K/100 h and CHT ageing 1–2.5 K/100 h.
- **One row per record.** The models score single records, and the
  operating point is stored as metadata (not a feature).
- **Test engines never simulated.** Only train and val engines are in the
  dataset (240 engines, 501,120 rows). The 60 test engines are simulated only
  inside the evaluation.

### Training: v3

- **Data.** `scripts/train_models.py --dataset v2 --extra-dataset life2
  --version v3 --no-rul`.
  - TRAIN is the v2 train trajectories plus the life2 train engines (501,480
    rows).
  - VAL is the v2 val trajectories plus the life2 val engines (162,600 rows).
- **Labels.** The engine's fault class once its severity is above zero,
  NOMINAL before. This is the same rule as the Prompt 16 trajectories.
- **Unchanged.** Same estimators, same hyper-parameters, and the same
  threshold rule (5 % FPR on validation NOMINAL). The threshold was not
  changed after the test results.
- **No RUL regressor.** v3 trains none; the lifetime RUL model is separate.

### Fleet healthy-residual reference

- **Model.** A quadratic polynomial (with interactions) in the operating
  point: rpm, MAP, altitude, ambient temperature and ambient pressure.
- **Fit.** Ridge least squares on the healthy life2 TRAIN records; stored as
  `models/v3/fleet_residual_reference.json`.
- **Spread removed.** Healthy residual spread before and after subtracting
  the reference:

| Channel | Healthy spread | After reference |
|---|---|---|
| EGT (per cylinder) | 44 K | 11–12 K |
| CHT | 16.0 K | 3.7 K |
| Oil temperature | 11.2 K | 0.2 K |
| Coolant | 1.46 K | 0.26 K |

- **What remains.** The rest of the EGT spread is the engine-to-engine
  variation and ageing that adaptation is meant to absorb, plus model misfit.

### Prompt 16 test split (frozen)

| | v2 | v3 |
|---|---|---|
| Anomaly PR-AUC | 0.841 | 0.853 |
| Anomaly FPR | 0.046 | **0.201** |
| Anomaly recall | 0.489 | 0.745 |
| Classifier macro-F1 | 0.767 | 0.766 |
| holdout_op PR-AUC / FPR / macro-F1 | 0.842 / 0.022 / 0.809 | 0.847 / 0.180 / 0.768 |

**Why v3's FPR regressed** (diagnosed on VALIDATION only, after the test
result; nothing was changed because of it):

| Share of healthy validation records flagged | v2 | v3 |
|---|---|---|
| Prompt 16 NOMINAL | 5.0 % | 21.0 % |
| life2 healthy | 84.7 % | 0.1 % |

- **v3's healthy training data is mostly lifetime.** About 80 % of v3's
  NOMINAL training records are life2, so its 5 % threshold on the pooled
  validation set fell where almost all false alarms are Prompt 16 records.
- **v2 fails the other way.** It flags almost every lifetime healthy record.
- **Neither threshold fits both.** One IsolationForest with one pooled
  threshold does not serve two populations this different.
- **Possible fix, not done.** Balancing the sources, or choosing the
  threshold per population on validation, would be a design change. It must
  be decided before looking at test results (OI-24).

### Lifetime test engines (12 healthy, 48 faulty)

| | v2 without | v2 with | v3 without | v3 with |
|---|---|---|---|---|
| Healthy windows with an alarm | 0.857 | 0.857 | 0.158 | 0.161 |
| – anomaly detector | 0.825 | 0.825 | 0.001 | 0.001 |
| – ML classifier | 0.340 | 0.340 | 0.114 | 0.118 |
| – rule-based classifier | 0.050 | 0.050 | 0.050 | 0.050 |
| Detection rate | 1.000 | 1.000 | 0.917 | 0.917 |
| Median lead time | 296.8 h | 296.8 h | 224.8 h | 224.8 h |
| Pre-onset windows with an alarm | 0.931 | 0.931 | 0.068 | 0.075 |
| Healthy windows on an adapted baseline | – | 0.000 | – | 0.914 |

- **v2's detection numbers mean little.** v2 also alarms in 93 % of
  pre-onset windows, so its 100 % detection and 297 h lead reflect alarming
  all the time.
- **v3 misses 4 of 48 faulty engines.** It detects 44: it misses 2
  BATTERY_DEGRADATION and 2 COOLING_FAULT engines. Its lead time is shorter
  mainly for FUEL_SYSTEM_FAULT (35 h) and INTAKE_BOOST_LEAK (60 h).
- **Every row counts for the decision.** The pre-registered rule compares
  the rows as they are, so these two points do not change the outcome.

### Adaptation verdict: REJECTED for both versions

- **v2.** Nothing is learned. The gate refuses 6500 of 6517 flights because
  of v2's false alarms, and no engine commissions.
- **v3.**
  - **Commissioning works with the new target.** 59 of 60 engines
    commissioned, with small, plausible corrections. On a traced
    healthy engine: EGT −6 to +1 K, CHT −2.1 K, gains about 0. In Prompt 17
    it was −176 K with the gain at its limit.
  - **But false alarms do not drop** (0.158 → 0.161), so (a) fails.
    - The anomaly detector already raises almost nothing (0.1 %).
    - The remaining false alarms come from the ML classifier and the rules,
      and adaptation does not change them.
    - Adaptation even raises ML alarms slightly (0.114 → 0.118).
- **Drift freezes adaptation almost immediately.** 5676 of 6517 v3 flights
  were frozen.
  - **Narrow channels.** Coolant (0.26 K healthy spread after the
    reference), oil temperature (0.2 K) and vibration are so narrow that the
    small differences between flights give PSI 2–7 right after
    commissioning.
  - **Ageing.** Later, the benign EGT ageing (5 K/100 h) also reads as
    SIGNIFICANT drift.
  - **Why.** PSI is scale-free: it does not ask whether a shift matters in
    engineering units. As designed, SIGNIFICANT drift freezes adaptation.
    The result is that the slow ageing adaptation is meant to follow is
    exactly what stops it (OI-25).

### Model version decision

The rule was fixed in `scripts/evaluate_adaptation.py` (`compare_versions`)
before any v3 result. v3 is adopted only if all of these hold:

- lifetime false alarms are lower;
- lifetime detection rate and median lead time are not worse;
- on the Prompt 16 test, PR-AUC and macro-F1 are within 0.02 of v2 and FPR
  is at most 0.05.

v3 passes the false-alarm, PR-AUC and macro-F1 checks. It fails detection
(0.917 < 1.000), lead time (224.8 < 296.8 h) and FPR (0.201 > 0.05).
**v2 is kept.**

- **What v2 is still better at.** v2's FPR on the Prompt 16 test is within
  target.
- **What v3 shows.** Lifetime-fleet data fixes the anomaly detector's
  lifetime false alarms: 0.825 → 0.001.
- **What a candidate needs.** A version that keeps both requires the
  threshold and source-balancing change above, decided on validation.

