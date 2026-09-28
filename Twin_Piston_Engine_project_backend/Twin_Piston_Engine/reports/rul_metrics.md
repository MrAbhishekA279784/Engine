# Lifetime RUL metrics

**Measured on SIMULATOR data only (provenance SIMULATED).** Held-out TEST engines.

Fleet life1: 300 engines, 31573 monitoring windows (one per 5.0 engine hours), sha256 cc56a79d1162.
Severity model MAE: val 0.063, test 0.066.

| True-RUL band | n | ML MAE [h] | ML 90 % coverage | Baseline MAE [h], same rows |
|---|---|---|---|---|
| 0-50h | 423 | 19.0 | 0.87 | 363.8 |
| 50-150h | 542 | 25.6 | 0.88 | 304.6 |
| 150-400h | 494 | 84.1 | 0.52 | 136.1 |

Overall 90 % interval coverage: 0.755.
Status share on faulty test windows: {'NO_DEGRADATION': 0.5869, 'ESTIMATED': 0.3009, 'INDETERMINATE': 0.1122}.
Healthy test engines: share of windows with an ESTIMATED RUL 0.054; baseline median 2000.0 h.

Decision: ADOPTED (decided on validation: beats baseline in every band with data = True; test: True).
