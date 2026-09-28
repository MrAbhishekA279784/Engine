# ML metrics (Prompt 16)

**Every metric below was measured on SIMULATOR data (provenance SIMULATED, PRD 4.4):**
the models were trained, thresholded and evaluated on trajectories from this repository's
forward simulator. None is a flight-data result.

Dataset v2 (sha256 891745ec552a, seed 20260928,
code 732b21886f8e2daa1d9d95f818de07c3c7bc0140-dirty), 960 trajectories, splits {'holdout_op': 112, 'test': 169, 'train': 508, 'val': 171};
models v3. Test split only; the held-out operating point (cruise 4400-4600 rpm) separately.

| Metric | Test | Held-out op | Target |
|---|---|---|---|
| Anomaly PR-AUC (SRD-PER-016) | 0.853 | 0.847 | 0.93 |
| Anomaly FPR (SRD-PER-017) | 0.201 | 0.180 | <= 0.05 |
| Classifier macro-F1, ML (SRD-PER-018) | 0.766 | 0.768 | 0.81 |
| Classifier macro-F1, rules (same records) | 0.524 | 0.521 | - |
| Median warning lead time [s] (SRD-PER-019) | 74.2 | 60.9 | - |
| RUL baseline MAE <=50 h [h] (SRD-PER-020) | 803.0 | 851.8 | - |

RUL bands 50-150 h and >150 h have no samples: the simulated faults develop over minutes.
RUL regressor: not trained (lifetime RUL model is separate, scripts/train_rul.py).

Per-class F1 (test): NOMINAL 0.83, MISFIRE 1.00, DETONATION_KNOCK 0.98, EXHAUST_VALVE_LEAK 0.94, INTAKE_BOOST_LEAK 0.25, OIL_DEGRADATION 0.98, COOLING_FAULT 0.68, BEARING_WEAR 1.00, SENSOR_FAULT 0.96, INJECTOR_FAULT 0.94, FUEL_SYSTEM_FAULT 0.18, IMBALANCE 0.99, CHARGING_FAULT 0.89, BATTERY_DEGRADATION 0.12

Cruise point (4000 rpm / 110 kPa, 120 s): injected -> rule / ML

- NOMINAL: NOMINAL / NOMINAL (SUCCESS)
- MISFIRE: MISFIRE / MISFIRE (SUCCESS)
- DETONATION_KNOCK: DETONATION_KNOCK / DETONATION_KNOCK (SUCCESS)
- EXHAUST_VALVE_LEAK: INJECTOR_FAULT / EXHAUST_VALVE_LEAK (SUCCESS)
- INTAKE_BOOST_LEAK: NOMINAL / NOMINAL (SUCCESS)
- OIL_DEGRADATION: OIL_DEGRADATION / OIL_DEGRADATION (SUCCESS)
- COOLING_FAULT: NOMINAL / COOLING_FAULT (SUCCESS)
- BEARING_WEAR: BEARING_WEAR / BEARING_WEAR (SUCCESS)
- SENSOR_FAULT: SENSOR_FAULT / SENSOR_FAULT (SUCCESS)
- INJECTOR_FAULT: NOMINAL / INJECTOR_FAULT (SUCCESS)
- FUEL_SYSTEM_FAULT: FUEL_SYSTEM_FAULT / FUEL_SYSTEM_FAULT (SUCCESS)
- IMBALANCE: IMBALANCE / IMBALANCE (SUCCESS)
- CHARGING_FAULT: CHARGING_FAULT / CHARGING_FAULT (SUCCESS)
- BATTERY_DEGRADATION: NOMINAL / NOMINAL (SUCCESS)

Channel dropout (test): abstain rate / confident-wrong rate per sensor group removed

- none: 0.003 / 0.044
- cht: 1.000 / 0.000
- coolant: 1.000 / 0.000
- crank: 1.000 / 0.000
- egt: 1.000 / 0.000
- electrical: 1.000 / 0.000
- fuel: 1.000 / 0.000
- ignition: 1.000 / 0.000
- oil: 1.000 / 0.000
- vibration: 1.000 / 0.000
