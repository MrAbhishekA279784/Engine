# Adaptation and model-version evaluation on the lifetime fleet (SIMULATED)

12 healthy and 48 faulty held-out TEST engines (engine split of life2), engine-to-engine variation and benign ageing as in life2 ({'ageing_egt_k_per_100h': (3.0, 6.0), 'ageing_cht_k_per_100h': (1.0, 2.5), 'egt_cylinder_offset_sd_k': 8.0, 'indicated_efficiency_rel': 0.02, 'eta_v_peak_rel': 0.015, 'fmep_a_rel': 0.05}); fleet healthy-residual reference v3.
Window alarm: anomaly on >= half the records, or rule / ML (confidence > 0.7) class not NOMINAL at the last record. Detection: two consecutive alarm windows after onset.

| | v2 without | v2 with | v3 without | v3 with |
|---|---|---|---|---|
| (a) healthy windows with an alarm | 0.857 | 0.857 | 0.158 | 0.161 |
| - anomaly | 0.825 | 0.825 | 0.001 | 0.001 |
| - ml | 0.340 | 0.340 | 0.114 | 0.118 |
| - rule | 0.050 | 0.050 | 0.050 | 0.050 |
| (b) detection rate | 1.000 | 1.000 | 0.917 | 0.917 |
| (b) median lead time (h) | 296.8 | 296.8 | 224.8 | 224.8 |
| pre-onset windows with an alarm | 0.931 | 0.931 | 0.068 | 0.075 |
| healthy windows on an adapted baseline | 0.000 | 0.000 | 0.000 | 0.914 |

**v2 adaptation: REJECTED: (a) false alarms on healthy engines are not reduced**  
decisions (flights): {'refused': 6500, 'commissioning_pending': 17}

**v3 adaptation: REJECTED: (a) false alarms on healthy engines are not reduced**  
decisions (flights): {'commissioning_pending': 539, 'commissioned': 59, 'adapted': 44, 'frozen': 5676, 'refused': 188, 'no_change': 11}

## Model version decision (pre-registered rule)

| Prompt 16 test | v2 | v3 |
|---|---|---|
| pr_auc | 0.841 | 0.853 |
| fpr | 0.046 | 0.201 |
| macro_f1 | 0.767 | 0.766 |

| check | pass |
|---|---|
| lifetime_fewer_false_alarms | True |
| lifetime_detection_not_worse | False |
| lifetime_lead_not_worse | False |
| p16_pr_auc_within_margin | True |
| p16_macro_f1_within_margin | True |
| p16_fpr_at_most_max | False |

Strictly better: {'p16_pr_auc': True, 'p16_macro_f1': False, 'p16_fpr': False, 'lifetime_false_alarms': True}

**Decision: v2 kept; failed: lifetime_detection_not_worse, lifetime_lead_not_worse, p16_fpr_at_most_max**
