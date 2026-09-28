"""
M-05 — Nyquist guard on every analysis band.

GAP
---
A repository-wide search for `nyquist`, `anti-alias` and `fs/2` returns zero
matches in src/. No code checks that a requested analysis band is
representable at the channel's sampling rate.

WHY THIS GUARD EXISTS
---------------------
The source framework (Physics-Based Parameter Derivation Framework, section
2.6) specified two things that cannot both be true:

    2.6C  "Apply Fast Fourier Transform to 500 Hz sampled vibration data"
    2.6D  "Bandpass filter (2-10 kHz) -> HF_acceleration"

The Nyquist limit at 500 Hz is 250 Hz. A 2-10 kHz band does not exist in that
signal — it was aliased away before digitisation. The specified envelope
analysis could not have run at all.

That defect reached a written specification and survived review. A guard turns
this class of error into an immediate, named failure instead of silently empty
output, which is how it survived: an FFT over a band containing no bins
returns zeros, and zeros look like a quiet engine.

The repo now samples at 2048 Hz (config.py:253), so the limit is 1024 Hz and
the configured bands (100/500/1000 Hz) are legal. The guard is still needed
because:

    - the sampling rate is configurable and can be lowered
    - order-referenced bands (M-02, M-03) move with RPM, so a band that is
      legal at 2000 rpm can exceed Nyquist at 5800 rpm
    - the envelope band (M-04) is a configuration value

Requirements: SRD-CON-005
"""

from __future__ import annotations


class NyquistViolation(ValueError):
    """Raised when an analysis band is not representable at the sample rate."""


def nyquist_limit_hz(fs_hz: float) -> float:
    """Half the sampling rate."""
    if fs_hz <= 0.0:
        return 0.0
    return fs_hz / 2.0


def assert_band_within_nyquist(f_high_hz: float, fs_hz: float,
                               label: str = "band") -> None:
    """Raise unless the band's upper edge sits below fs/2.

    Call this at every point a band is defined: the three configured frequency
    bands, the envelope band, and every order band after conversion to Hz.
    """
    if fs_hz <= 0.0:
        raise NyquistViolation(
            f"{label}: sampling rate must be positive, got {fs_hz}")
    limit = fs_hz / 2.0
    if f_high_hz > limit:
        raise NyquistViolation(
            f"{label}: upper edge {f_high_hz:.1f} Hz exceeds Nyquist limit "
            f"{limit:.1f} Hz at fs={fs_hz:.1f} Hz"
        )


def band_is_representable(f_high_hz: float, fs_hz: float) -> bool:
    """Non-raising form, for code paths that degrade rather than fail."""
    return fs_hz > 0.0 and f_high_hz <= fs_hz / 2.0


def max_representable_order(rpm: float, fs_hz: float) -> float:
    """Highest engine order representable at this speed and sample rate.

        order_max = (fs / 2) / (rpm / 60)

    Report this alongside order-band features so a consumer can tell whether
    a zero means "no energy there" or "not measurable at this speed".
    """
    if rpm <= 0.0 or fs_hz <= 0.0:
        return 0.0
    return (fs_hz / 2.0) / (rpm / 60.0)


def validate_vibration_config(
    sampling_frequency_hz: float,
    freq_band_low_max_hz: float,
    freq_band_mid_max_hz: float,
    freq_band_high_max_hz: float,
    envelope_band_high_hz: float | None = None,
) -> list[str]:
    """Validate the configured bands at startup.

    Returns a list of problems; empty means the configuration is legal. Call
    this from the settings loader so a bad configuration fails at boot rather
    than producing silent zeros in flight.
    """
    problems: list[str] = []
    checks = [
        ("freq_band_low_max_hz", freq_band_low_max_hz),
        ("freq_band_mid_max_hz", freq_band_mid_max_hz),
        ("freq_band_high_max_hz", freq_band_high_max_hz),
    ]
    if envelope_band_high_hz is not None:
        checks.append(("envelope_band_high_hz", envelope_band_high_hz))

    for name, edge in checks:
        try:
            assert_band_within_nyquist(edge, sampling_frequency_hz, name)
        except NyquistViolation as exc:
            problems.append(str(exc))

    if freq_band_low_max_hz >= freq_band_mid_max_hz:
        problems.append("freq_band_low_max_hz must be below freq_band_mid_max_hz")
    if freq_band_mid_max_hz >= freq_band_high_max_hz:
        problems.append("freq_band_mid_max_hz must be below freq_band_high_max_hz")
    return problems


if __name__ == "__main__":
    print("M-05  Nyquist guard")

    print("\n  the original framework specification:")
    try:
        assert_band_within_nyquist(10000.0, 500.0, "framework envelope band 2-10 kHz")
    except NyquistViolation as exc:
        print(f"    CAUGHT: {exc}")

    print("\n  current repo configuration (fs 2048 Hz):")
    problems = validate_vibration_config(2048.0, 100.0, 500.0, 1000.0, 1000.0)
    print(f"    problems: {problems or 'none — configuration is legal'}")

    print("\n  same bands if someone lowers fs to 500 Hz:")
    for p in validate_vibration_config(500.0, 100.0, 500.0, 1000.0, 1000.0):
        print(f"    {p}")

    print("\n  order-band representability at fs 2048 Hz:")
    print(f"{'rpm':>8}{'f_1X Hz':>10}{'max order':>12}{'4X legal':>10}")
    for rpm in (1500.0, 3000.0, 4500.0, 5800.0):
        mo = max_representable_order(rpm, 2048.0)
        print(f"{rpm:8.0f}{rpm / 60.0:10.1f}{mo:12.1f}"
              f"{str(mo >= 4.0):>10}")
