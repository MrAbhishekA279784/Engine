"""
B-01 — Vibration ring buffer: the live path never fills a window.

USAGE RESTRICTION (added on integration)
----------------------------------------
Must NEVER be fed at the telemetry record rate (1 Hz). 2048 record-rate
samples span ~34 minutes; analysing them as one 2048 Hz window is wrong.
The "Usage inside VibrationProcessor" example below is superseded: L2 now
analyses the per-record accelerometer burst (RawSignalRecord.accel_burst_*).
This buffer is reserved for the live CAN adapter, and only if the CAN stream
delivers accelerometer samples at the vibration sample rate.

BUG
---
File  : src/l2_digital_twin/vibration_processor.py
Lines : 283-297, `process_record`

    x_val = record.vibration_x.value if record.vibration_x.valid else 0.0
    ...
    return self.process_window(
        [x_val] if record.vibration_x.valid else [],
        [y_val] if record.vibration_y.valid else [],
        [z_val] if record.vibration_z.valid else [],
        fs_hz=self._cfg.sampling_frequency_hz, ...)

A SINGLE SAMPLE per axis is passed to a function that performs an FFT.

An FFT over one sample has no spectrum. `process_window` itself is written
correctly for a full window, and the config declares the right sizes
(config.py:253-254, sampling_frequency_hz 2048.0, window_size 2048), but the
live record path never accumulates anything.

OBSERVED CONSEQUENCE
--------------------
On the real-time path every spectral feature is degenerate:

    dominant_frequency_hz   0.0
    dominant_amplitude      0.0
    band_energy_low/mid/high 0.0
    dominant_order          None
    spectral_centroid_hz    0.0
    spectral_bandwidth_hz   0.0

The time-domain features (RMS, peak, crest) are computed over one sample, so
RMS equals peak and the crest factor is always 1.0.

WHY THIS IS FIRST IN THE FIX ORDER
----------------------------------
M-01 to M-04 all consume a window. None of them can do useful work until this
buffer exists, so it is a precondition for the entire vibration work package.

Requirements: SRD-FUN-063 (windowed transform of at least 1024 points)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VibrationRingBuffer:
    """Fixed-capacity per-axis sample buffer for the real-time path.

    Push one tri-axial sample per acquisition tick; read a full window once
    `ready()` is true. Capacity should match the configured FFT window size.

    Usage inside VibrationProcessor:

        def __init__(self, settings=None):
            ...
            self._buffer = VibrationRingBuffer(capacity=self._cfg.window_size)

        def process_record(self, record):
            self._buffer.push(record.vibration_x.value,
                              record.vibration_y.value,
                              record.vibration_z.value)
            if not self._buffer.ready():
                return self._pending_state(record)     # no fabricated spectrum
            x, y, z = self._buffer.window()
            return self.process_window(x, y, z,
                                       fs_hz=self._cfg.sampling_frequency_hz,
                                       rpm=..., timestamp=record.timestamp)
    """

    capacity: int = 2048
    _x: deque = field(init=False, repr=False)
    _y: deque = field(init=False, repr=False)
    _z: deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 8:
            raise ValueError("Ring buffer capacity must be at least 8 samples")
        self._x = deque(maxlen=self.capacity)
        self._y = deque(maxlen=self.capacity)
        self._z = deque(maxlen=self.capacity)

    def push(self, x: float, y: float, z: float) -> None:
        """Append one tri-axial sample."""
        self._x.append(float(x))
        self._y.append(float(y))
        self._z.append(float(z))

    def ready(self) -> bool:
        """True once a full window is available."""
        return len(self._x) >= self.capacity

    def fill_fraction(self) -> float:
        """0.0 to 1.0 — useful for reporting warm-up state to the dashboard."""
        return len(self._x) / float(self.capacity)

    def samples_until_ready(self) -> int:
        return max(0, self.capacity - len(self._x))

    def window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Current window as three arrays, oldest sample first."""
        return (
            np.fromiter(self._x, dtype=float, count=len(self._x)),
            np.fromiter(self._y, dtype=float, count=len(self._y)),
            np.fromiter(self._z, dtype=float, count=len(self._z)),
        )

    def clear(self) -> None:
        self._x.clear()
        self._y.clear()
        self._z.clear()


def warmup_seconds(capacity: int, sampling_frequency_hz: float) -> float:
    """How long before the first window is available.

    Worth surfacing on the dashboard: at 2048 samples and 2048 Hz the first
    spectrum is one second after start, and features should be reported as
    warming up rather than as zeros until then.
    """
    if sampling_frequency_hz <= 0.0:
        return float("inf")
    return capacity / sampling_frequency_hz


if __name__ == "__main__":
    fs, n = 2048.0, 2048
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 100.0 * t)

    print("B-01  Vibration ring buffer")
    print(f"\n  warm-up at {int(fs)} samples / {fs:.0f} Hz: "
          f"{warmup_seconds(n, fs):.2f} s")

    buf = VibrationRingBuffer(capacity=n)
    for k, s in enumerate(sig):
        buf.push(s, s, s)
        if k in (0, 511, 1023, 2047):
            print(f"  after {k + 1:5d} samples  fill {buf.fill_fraction():5.1%}  "
                  f"ready {str(buf.ready()):5s}  "
                  f"remaining {buf.samples_until_ready()}")

    x, y, z = buf.window()
    print(f"\n  window shapes: {x.shape}, {y.shape}, {z.shape}")

    spec = np.fft.rfft(x * np.hanning(x.size))
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    peak = freqs[int(np.argmax(np.abs(spec[1:]))) + 1]
    print(f"  spectrum peak: {peak:.1f} Hz   (injected 100.0 Hz)  "
          f"{'PASS' if abs(peak - 100.0) < 2.0 else 'FAIL'}")

    print("\n  shipped behaviour — one sample:")
    one = np.array([sig[0]])
    s1 = np.fft.rfft(one)
    print(f"    window size {one.size}, rfft bins {s1.size} -> "
          f"no spectrum, crest factor always 1.0")
