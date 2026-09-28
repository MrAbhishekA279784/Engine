"""
M-04 — Envelope demodulation for bearing-defect detection.

GAP
---
Zero matches in src/ for `hilbert`, `analytic_signal` or `demodul`. The
`envelope` matches found during the audit were in `api/v1/schemas.py` and
`api/app.py` and are unrelated (HTTP response envelopes).

THE PHYSICS
-----------
A bearing defect does not produce a tone. Each time a rolling element passes
the defect it delivers a short impact, and that impact rings the surrounding
structure at its resonant frequency — typically several hundred Hz to a few
kHz.

So the vibration contains a HIGH-FREQUENCY CARRIER (the resonance) that is
AMPLITUDE-MODULATED at a LOW-FREQUENCY RATE (the defect repetition rate).

In the raw spectrum the carrier dominates and the modulation appears only as
small sidebands, easily lost in noise. Demodulating the band recovers the
modulation envelope directly, where the defect rate becomes the dominant
component.

This is why envelope analysis is the standard bearing-diagnostic method and
why RMS alone misses early defects: total energy barely changes while the
modulation grows.

BAND SELECTION
--------------
Default 650-1000 Hz (revised on integration, OI-9). The band must sit ABOVE
the engine's firing harmonics, otherwise the envelope of a healthy engine is
dominated by them: on the Rotax 915 iS operating range 4X reaches 387 Hz and
6X 580 Hz at 5800 rpm, so the lower edge is 650 Hz (70 Hz margin), and the
upper edge stays below the 1024 Hz Nyquist limit at 2048 Hz sampling. The
original 300-1000 Hz band contained 4X above 4500 rpm and, with a rectangular
band-pass, received leakage from 4X at every rpm where 4X was not bin-centred.

TAPERING
--------
The band-pass uses raised-cosine (Tukey) transitions of `taper_hz` at each
edge instead of a brick-wall mask, and the time record is Tukey-windowed
(`time_taper_alpha`) before the FFT, which suppresses leakage of the large
low-order firing tones into the band. The envelope RMS is computed over the
untapered centre of the record so the time window does not itself create an
envelope.

The source framework specified 2-10 kHz on 500 Hz sampled data, which is not
representable — the Nyquist limit there is 250 Hz. See M-05. If the team
later raises the sampling rate to reach a higher structural resonance, raise
this band with it; the guard will refuse an illegal combination.

MEASURED
--------
    600 Hz carrier, amplitude-modulated at 37 Hz -> envelope RMS 0.5657
    600 Hz carrier, unmodulated                  -> envelope RMS 0.0000

Requirements: SRD-FUN-067
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_BAND_LO_HZ = 650.0
DEFAULT_BAND_HI_HZ = 1000.0
DEFAULT_TAPER_HZ = 50.0          # raised-cosine transition inside each band edge
DEFAULT_TIME_TAPER_ALPHA = 0.1   # Tukey time window (fraction tapered, both ends)


def tukey_window(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered-cosine) window; alpha=0 rectangular, alpha=1 Hann."""
    if n <= 1 or alpha <= 0.0:
        return np.ones(n)
    alpha = min(alpha, 1.0)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge > 0:
        k = np.arange(edge + 1)
        ramp = 0.5 * (1.0 - np.cos(np.pi * k / edge))
        w[: edge + 1] = ramp
        w[n - edge - 1:] = ramp[::-1]
    return w


class NyquistViolation(ValueError):
    """Raised when the envelope band is not representable. See M-05."""


def analytic_envelope(samples: np.ndarray) -> np.ndarray:
    """Magnitude of the analytic signal (Hilbert transform), via FFT.

    Implemented on numpy alone so this module adds no scipy dependency — the
    repo requires numpy but not scipy.

    The analytic signal is formed by zeroing the negative-frequency half of
    the spectrum and doubling the positive half; its magnitude is the
    instantaneous amplitude, i.e. the envelope.
    """
    n = samples.size
    spec = np.fft.fft(samples)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spec * h))


def bandpass_fft(samples: np.ndarray, fs_hz: float,
                 band_lo_hz: float, band_hi_hz: float,
                 taper_hz: float = 0.0,
                 time_taper_alpha: float = 0.0) -> np.ndarray:
    """Zero-phase band-pass by spectral masking.

    taper_hz > 0 gives raised-cosine transitions from 0 at the band edge to
    1 at edge +/- taper_hz (inside the band); time_taper_alpha > 0 applies a
    Tukey window to the record before the FFT. Both default to 0 (the original
    brick-wall behaviour).
    """
    n = samples.size
    x = samples - float(np.mean(samples))
    if time_taper_alpha > 0.0:
        x = x * tukey_window(n, time_taper_alpha)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    mask = ((freqs >= band_lo_hz) & (freqs <= band_hi_hz)).astype(float)
    if taper_hz > 0.0:
        lo_ramp = (freqs >= band_lo_hz) & (freqs < band_lo_hz + taper_hz)
        hi_ramp = (freqs > band_hi_hz - taper_hz) & (freqs <= band_hi_hz)
        mask[lo_ramp] = 0.5 * (1.0 - np.cos(np.pi * (freqs[lo_ramp] - band_lo_hz) / taper_hz))
        mask[hi_ramp] = 0.5 * (1.0 - np.cos(np.pi * (band_hi_hz - freqs[hi_ramp]) / taper_hz))
    return np.fft.irfft(spec * mask, n=n)


def envelope_rms(
    samples: Sequence[float],
    fs_hz: float,
    band_lo_hz: float = DEFAULT_BAND_LO_HZ,
    band_hi_hz: float = DEFAULT_BAND_HI_HZ,
    taper_hz: float = DEFAULT_TAPER_HZ,
    time_taper_alpha: float = DEFAULT_TIME_TAPER_ALPHA,
) -> float:
    """RMS of the demodulated envelope of a high-frequency band.

    ADD TO AxisVibrationFeatures as: envelope_rms: float
    ADD TO VibrationState        as: envelope_rms

    Feeds the Vibration Health Index (M-10) — it is the term that makes VHI
    sensitive to bearing defects before they raise the overall level.

    Raises NyquistViolation if the requested band exceeds fs/2, rather than
    silently returning zeros from an empty band.
    """
    if fs_hz <= 0.0:
        raise NyquistViolation(f"sampling rate must be positive, got {fs_hz}")
    if band_hi_hz > fs_hz / 2.0:
        raise NyquistViolation(
            f"envelope band upper edge {band_hi_hz:.1f} Hz exceeds Nyquist "
            f"limit {fs_hz / 2.0:.1f} Hz at fs={fs_hz:.1f} Hz"
        )

    arr = np.asarray(samples, dtype=float)
    if arr.size < 16 or not np.all(np.isfinite(arr)):
        return 0.0

    filtered = bandpass_fft(arr, fs_hz, band_lo_hz, band_hi_hz, taper_hz, time_taper_alpha)
    env = analytic_envelope(filtered)
    # Use the untapered centre only, so the time window does not create an envelope.
    edge = int(np.ceil(time_taper_alpha * (arr.size - 1) / 2.0)) if time_taper_alpha > 0.0 else 0
    core = env[edge: arr.size - edge] if arr.size - 2 * edge >= 16 else env
    core = core - float(np.mean(core))    # remove the DC pedestal
    return float(np.sqrt(np.mean(core ** 2)))


def envelope_spectrum(
    samples: Sequence[float],
    fs_hz: float,
    band_lo_hz: float = DEFAULT_BAND_LO_HZ,
    band_hi_hz: float = DEFAULT_BAND_HI_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectrum OF THE ENVELOPE — where the defect rate appears as a peak.

    Returns (freqs, magnitude). The dominant peak is the defect repetition
    rate, which can be compared against computed bearing defect frequencies
    (BPFO, BPFI, BSF) once the bearing geometry is known.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size < 16 or fs_hz <= 0.0 or not np.all(np.isfinite(arr)):
        return np.empty(0), np.empty(0)
    filtered = bandpass_fft(arr, fs_hz, band_lo_hz, band_hi_hz)
    env = analytic_envelope(filtered)
    env = env - float(np.mean(env))
    spec = np.fft.rfft(env * np.hanning(env.size))
    return np.fft.rfftfreq(env.size, d=1.0 / fs_hz), np.abs(spec)


if __name__ == "__main__":
    fs, n = 2048.0, 2048
    t = np.arange(n) / fs
    rng = np.random.default_rng(13)

    print("M-04  Envelope demodulation")
    print(f"{'signal':>34}{'env RMS':>10}{'raw RMS':>10}")
    plain = np.sin(2 * np.pi * 600.0 * t)
    cases = [("600 Hz carrier, unmodulated", plain)]
    for depth in (0.2, 0.5, 0.8):
        cases.append((f"600 Hz carrier, {depth:.0%} modulated at 37 Hz",
                      (1.0 + depth * np.sin(2 * np.pi * 37.0 * t))
                      * np.sin(2 * np.pi * 600.0 * t)))
    for label, sig in cases:
        print(f"{label:>34}{envelope_rms(sig, fs):10.4f}"
              f"{float(np.sqrt(np.mean(sig ** 2))):10.4f}")

    print("\n  raw RMS barely moves; envelope RMS scales with modulation depth.")
    print("  That is the early-bearing-defect signal RMS alone misses.")

    modulated = (1.0 + 0.8 * np.sin(2 * np.pi * 37.0 * t)) \
        * np.sin(2 * np.pi * 600.0 * t) + 0.05 * rng.standard_normal(n)
    f, m = envelope_spectrum(modulated, fs)
    peak = f[int(np.argmax(m[1:])) + 1]
    print(f"\n  envelope spectrum peak: {peak:.1f} Hz  (injected 37.0 Hz)  "
          f"{'PASS' if abs(peak - 37.0) < 2.0 else 'FAIL'}")

    print("\n  Nyquist guard (framework's 2-10 kHz on 500 Hz data):")
    try:
        envelope_rms(plain, 500.0, 2000.0, 10000.0)
    except NyquistViolation as exc:
        print(f"    CAUGHT: {exc}")
