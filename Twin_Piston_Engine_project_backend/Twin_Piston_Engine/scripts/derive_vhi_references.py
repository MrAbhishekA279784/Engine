"""
DIAGNOSTIC -- reports, never writes configuration.

Derives PROVISIONAL, SIMULATOR-DERIVED healthy references for the Vibration
Health Index (VibrationConfig.vhi_ref_*) and reports healthy VHI over
2000-5800 rpm x 80-135 kPa for a given reference set. The numbers printed
here were copied into src/core/config.py by hand and labelled as a
provisional installation baseline, to be replaced by a baseline captured from
the first healthy flights (docs/OPEN_ITEMS.md, OI-9).

Run from the repo root:
    uv run python scripts/derive_vhi_references.py
"""

from __future__ import annotations

import numpy as np

from src.core.config import get_settings
from src.l1_data.simulator.forward_simulator import ForwardPhysicsModel, SensorForwardModel
from src.l2_digital_twin.physics import M10_lubrication_vibration_indices as M10
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.vibration_processor import VibrationProcessor

REF_MAP_PA = 110000.0
TABLE_RPM = [2000.0, 3000.0, 4000.0, 5000.0, 5800.0]
GRID_RPM = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 5800.0]
GRID_MAP = [80000.0, 110000.0, 135000.0]


def nominal_features(rpm: float, map_pa: float, seed: int = 7) -> tuple[float, float, float]:
    """(overall RMS, crest, envelope RMS) of a nominal simulator burst."""
    gt = ForwardPhysicsModel(seed=seed).compute_ground_truth(time_s=1.0, rpm=rpm, map_pa=map_pa)
    norm = convert_raw_to_engineering_state(SensorForwardModel(seed=seed).convert_to_raw_record(gt))
    state, _ = VibrationProcessor().process_record(norm)
    return state.overall_rms_m_s2.value, state.crest_factor.value, state.envelope_rms_m_s2.value


def grid_vhi(ref_rpm, ref_rms, ref_crest, ref_env) -> dict[tuple[float, float], float]:
    out = {}
    for rpm in GRID_RPM:
        r_rms = float(np.interp(rpm, ref_rpm, ref_rms))
        r_crest = float(np.interp(rpm, ref_rpm, ref_crest))
        r_env = float(np.interp(rpm, ref_rpm, ref_env))
        for map_pa in GRID_MAP:
            rms, crest, env = nominal_features(rpm, map_pa)
            out[(rpm, map_pa)] = M10.vibration_health_index(rms, r_rms, crest, r_crest, env, r_env).value
    return out


def report(label: str, vhi: dict[tuple[float, float], float]) -> None:
    print(f"\n{label}: healthy VHI, rows rpm, columns MAP kPa {[int(m / 1000) for m in GRID_MAP]}")
    for rpm in GRID_RPM:
        print(f"  {rpm:6.0f}  " + "  ".join(f"{vhi[(rpm, m)]:.3f}" for m in GRID_MAP))
    vals = list(vhi.values())
    print(f"  range {min(vals):.3f} - {max(vals):.3f}; outside 0.8-1.2: "
          f"{sum(1 for v in vals if not 0.8 <= v <= 1.2)} of {len(vals)}")


if __name__ == "__main__":
    print("DIAGNOSTIC ONLY: nothing is written to configuration.")
    cfg = get_settings().vibration
    report("Configured references", grid_vhi(cfg.vhi_ref_rpm, cfg.vhi_ref_rms_m_s2,
                                              cfg.vhi_ref_crest, cfg.vhi_ref_envelope_m_s2))

    rows = [nominal_features(rpm, REF_MAP_PA) for rpm in TABLE_RPM]
    table = ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    print(f"\nrpm table derived at MAP {REF_MAP_PA / 1000:.0f} kPa (nominal simulator):")
    print(f"  vhi_ref_rpm:           {TABLE_RPM}")
    print(f"  vhi_ref_rms_m_s2:      {[round(v, 3) for v in table[0]]}")
    print(f"  vhi_ref_crest:         {[round(v, 3) for v in table[1]]}")
    print(f"  vhi_ref_envelope_m_s2: {[round(v, 4) for v in table[2]]}")
    report("rpm-table references", grid_vhi(TABLE_RPM, *table))
