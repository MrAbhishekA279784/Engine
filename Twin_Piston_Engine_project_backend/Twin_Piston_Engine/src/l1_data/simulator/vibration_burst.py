"""
Simulator accelerometer and crank-period bursts (L1, simulation only).

Each telemetry record (1 Hz) carries a short high-rate burst so that L2 can
compute spectral features over a real window at the vibration sample rate,
instead of over one sample per record.

Accelerometer burst (fs = 2048 Hz, N = 2048 samples = 1 s):
    - Firing order 2X = 2 * rpm/60 Hz (four-cylinder four-stroke: two firing
      events per crank revolution), plus its 4X harmonic.
    - Half order 0.5X (and 1.5X): one cylinder's combustion impulse once per
      720 deg cycle. Present only under a misfire; relative amplitude rises
      with severity.
    - Bearing wear: periodic impacts at the outer-race defect frequency
      BPFO = (Nb/2) * f_rot * (1 - (d/D) cos(alpha)), each ringing a
      structural resonance (800 Hz, damping ratio ~0.13; the standard
      impact-resonance bearing model). The resonance sits inside the
      650-1000 Hz envelope band, above the firing harmonics. (Was a 600 Hz
      carrier with +/-BPFO sidebands, whose sidebands could not fit the band
      and exceeded Nyquist at high rpm; OI-9.)
    - Gaussian sensor noise.
    Energy accounting: when the pre-fault (base) vibration level is known,
    the firing tones carry the base level and the fault component carries
    exactly the energy the fault added, RMS_fault = sqrt(total^2 - base^2).
    Without a base level, fault components use severity-relative amplitudes
    and the whole waveform is scaled to the total level.

Crank burst (K per-revolution periods, us):
    Firing order 1-3-2-4 puts cylinders 1 and 3 in one crank revolution and
    2 and 4 in the next. A misfiring cylinder loses its power stroke once per
    two revolutions, so every second revolution is slowed.

Fault identifiers are compared by integer value, so both
simulator.fault_injection.FaultMode and core.provenance.FaultClass work
(their numbering is identical: MISFIRE = 1, BEARING_WEAR = 7).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.l1_data.simulator.fault_injection import FaultMode

BURST_FS_HZ = 2048.0
BURST_SAMPLES = 2048
CRANK_BURST_REVS = 32

FOURTH_ORDER_REL = 0.25          # 4X relative to 2X firing amplitude
HALF_ORDER_REL_PER_SEVERITY = 1.5  # 0.5X relative to 2X at full misfire severity
BEARING_REL_PER_SEVERITY = 2.0   # bearing carrier relative to 2X at full severity
BEARING_RESONANCE_HZ = 800.0     # structural resonance excited by each impact
BEARING_RESONANCE_DECAY_S = 0.0015  # 1/e decay time (damping ratio ~0.13 at 800 Hz)
BEARING_CARRIER_HZ = BEARING_RESONANCE_HZ  # backwards-compatible name
BEARING_BALLS = 9                # rolling elements
BEARING_D_OVER_PITCH = 0.2       # ball diameter / pitch diameter
ACCEL_NOISE_STD_M_S2 = 0.2       # sensor noise, 1 sigma
CRANK_JITTER_STD_US = 2.0        # timing jitter, 1 sigma (matches single-period noise)
MISFIRE_REV_SLOWDOWN_PER_SEVERITY = 0.02  # fractional period increase per misfiring cylinder


def bearing_bpfo_hz(rpm: float) -> float:
    """Outer-race defect frequency for the modelled bearing (contact angle 0)."""
    f_rot = rpm / 60.0
    return (BEARING_BALLS / 2.0) * f_rot * (1.0 - BEARING_D_OVER_PITCH)


def bearing_impact_train(rpm: float, fs_hz: float, n_samples: int) -> np.ndarray:
    """Impacts at BPFO, each ringing the structural resonance (unit peak).

    Circular convolution of an impulse train with a decaying sinusoid, so the
    burst is stationary with no start-up transient.
    """
    bpfo = bearing_bpfo_hz(rpm)
    if bpfo <= 0.0:
        return np.zeros(n_samples)
    impulses = np.zeros(n_samples)
    period = fs_hz / bpfo
    positions = np.arange(0.0, n_samples, period)
    impulses[np.round(positions).astype(int) % n_samples] = 1.0
    k = np.arange(n_samples) / fs_hz
    ring = np.exp(-k / BEARING_RESONANCE_DECAY_S) * np.sin(2.0 * math.pi * BEARING_RESONANCE_HZ * k)
    return np.real(np.fft.ifft(np.fft.fft(impulses) * np.fft.fft(ring)))


def generate_accel_burst(
    rpm: float,
    axis_rms_m_s2: float | tuple[float, float, float],
    fault_mode: int = FaultMode.NOMINAL,
    severity: float = 0.0,
    rng: np.random.Generator | None = None,
    fs_hz: float = BURST_FS_HZ,
    n_samples: int = BURST_SAMPLES,
    enable_noise: bool = True,
    base_rms_m_s2: float | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tri-axial acceleration burst in m/s^2 (x, y, z), each n_samples long.

    axis_rms_m_s2 is the deterministic per-axis RMS level (fault included):
    one value for all three axes, or an (x, y, z) tuple. base_rms_m_s2, if
    given, is the pre-fault level in the same form; see module docstring.
    """
    t = np.arange(n_samples) / fs_hz
    f_rot = max(rpm, 0.0) / 60.0
    sev = min(max(severity, 0.0), 1.0)
    mode = int(fault_mode)
    rng = rng or np.random.default_rng(0)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    firing = np.sin(2.0 * math.pi * 2.0 * f_rot * t + phase)
    firing += FOURTH_ORDER_REL * np.sin(2.0 * math.pi * 4.0 * f_rot * t + 2.0 * phase)

    # Unit-amplitude fault shape (relative to the 2X firing tone).
    fault = np.zeros(n_samples)
    if mode == FaultMode.MISFIRE and sev > 0.0:
        fault += np.sin(2.0 * math.pi * 0.5 * f_rot * t + 0.5 * phase)
        fault += 0.5 * np.sin(2.0 * math.pi * 1.5 * f_rot * t + 1.5 * phase)
        rel = HALF_ORDER_REL_PER_SEVERITY * sev
    elif mode == FaultMode.BEARING_WEAR and sev > 0.0:
        fault += bearing_impact_train(rpm, fs_hz, n_samples)
        rel = BEARING_REL_PER_SEVERITY * sev
    else:
        rel = 0.0

    def _per_axis(v: float | tuple[float, float, float]) -> tuple[float, ...]:
        return tuple(v) if isinstance(v, (tuple, list)) else (v,) * 3

    def _rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(a ** 2)))

    totals = _per_axis(axis_rms_m_s2)
    bases = _per_axis(base_rms_m_s2) if base_rms_m_s2 is not None else None
    firing_rms, fault_rms = _rms(firing), _rms(fault)

    axes = []
    for i, total in enumerate(totals):
        total = max(total, 0.0)
        if bases is not None:
            base = min(max(bases[i], 0.0), total)
            fault_level = math.sqrt(total * total - base * base) if fault_rms > 0.0 else 0.0
            if fault_rms == 0.0:
                base = total  # no fault shape: all energy stays in the firing tones
            signal = firing * (base / firing_rms if firing_rms > 0.0 else 0.0)
            if fault_rms > 0.0:
                signal = signal + fault * (fault_level / fault_rms)
        else:
            wave = firing + rel * fault
            wave_rms = _rms(wave)
            signal = wave * (total / wave_rms if wave_rms > 0.0 else 0.0)
        noise = rng.normal(0.0, ACCEL_NOISE_STD_M_S2, n_samples) if enable_noise else 0.0
        axes.append(signal + noise)
    return axes[0], axes[1], axes[2]


def accel_to_counts(samples_m_s2: np.ndarray, counts_per_m_s2: float) -> tuple[int, ...]:
    """Quantise acceleration to ADC counts, as the single-sample channel is."""
    return tuple(int(v) for v in np.asarray(samples_m_s2) * counts_per_m_s2)


def generate_crank_burst(
    rpm: float,
    fault_mode: int = FaultMode.NOMINAL,
    severity: float = 0.0,
    affected_cylinders: Sequence[int] = (1,),
    rng: np.random.Generator | None = None,
    n_revs: int = CRANK_BURST_REVS,
    enable_noise: bool = True,
) -> tuple[float, ...]:
    """Last n_revs per-revolution crank periods in microseconds, oldest first."""
    period_us = 60.0e6 / max(rpm, 100.0)
    periods = np.full(n_revs, period_us)

    sev = min(max(severity, 0.0), 1.0)
    if int(fault_mode) == FaultMode.MISFIRE and sev > 0.0:
        # Firing order 1-3-2-4: cylinders 1,3 fire in even revolutions, 2,4 in odd.
        for cyl in affected_cylinders:
            if cyl in (1, 3):
                periods[0::2] *= 1.0 + MISFIRE_REV_SLOWDOWN_PER_SEVERITY * sev
            elif cyl in (2, 4):
                periods[1::2] *= 1.0 + MISFIRE_REV_SLOWDOWN_PER_SEVERITY * sev

    if enable_noise:
        rng = rng or np.random.default_rng(0)
        periods = periods + rng.normal(0.0, CRANK_JITTER_STD_US, n_revs)
    return tuple(float(p) for p in np.maximum(periods, 1.0))
