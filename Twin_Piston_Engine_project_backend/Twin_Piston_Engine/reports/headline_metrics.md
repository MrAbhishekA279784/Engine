# Headline numbers (ALL SIMULATED)

12 healthy and 48 faulty held-out test engines of the lifetime fleet (life2 engine split), engine-to-engine variation and benign ageing ({'ageing_egt_k_per_100h': [3.0, 6.0], 'ageing_cht_k_per_100h': [1.0, 2.5], 'egt_cylinder_offset_sd_k': 8.0, 'indicated_efficiency_rel': 0.02, 'eta_v_peak_rel': 0.015, 'fmep_a_rel': 0.05}). Every number below is from the forward simulator: **simulated**.

| (simulated) | rule-based alone | v3 alone | rules + v3 |
|---|---|---|---|
| Healthy-window false-alarm rate | 0.050 (simulated) | 0.114 (simulated) | 0.158 (simulated) |
| Detection rate | 0.771 (simulated) | 0.917 (simulated) | 0.917 (simulated) |
| Median lead time before failure (h) | 174.7 (simulated) | 207.7 (simulated) | 224.8 (simulated) |
| Pre-onset windows with an alarm | 0.052 (simulated) | 0.019 (simulated) | 0.069 (simulated) |
| Missed engines | BATTERY_DEGRADATION, BATTERY_DEGRADATION, BATTERY_DEGRADATION, BATTERY_DEGRADATION, COOLING_FAULT, COOLING_FAULT, COOLING_FAULT, INTAKE_BOOST_LEAK, INTAKE_BOOST_LEAK, INTAKE_BOOST_LEAK, INTAKE_BOOST_LEAK | BATTERY_DEGRADATION, BATTERY_DEGRADATION, COOLING_FAULT, COOLING_FAULT | BATTERY_DEGRADATION, BATTERY_DEGRADATION, COOLING_FAULT, COOLING_FAULT |

RUL (lifetime model `models/life1/rul_lifetime.joblib`, trained on life1 without engine variation), ESTIMATED windows of faulty engines, true RUL capped at 400 h:

| True-RUL band | n | MAE (h) | 90 % interval coverage |
|---|---|---|---|
| 0-50h | 473 | 35.9 (simulated) | 0.71 (simulated) |
| 50-150h | 925 | 38.4 (simulated) | 0.88 (simulated) |
| 150-400h | 2385 | 103.4 (simulated) | 0.38 (simulated) |

Overall 90 % coverage 0.543 (simulated); healthy-engine windows with an ESTIMATED RUL 0.778 (simulated).

v2 lead time is deliberately NOT a headline: v2 alarms on 93 % of pre-fault windows (reports/adaptation_eval.md), so its lead time measures constant alarming.
