"""IMU extraction via telemetry-parser.

Produces per-sensor sample lists in SI units (m/s² for accelerometer,
rad/s for gyroscope) suitable for writing straight to CSV.

Plan §3: `present` is `True` ONLY when both the gyroscope and the
accelerometer streams parse successfully. Single-sensor clips report
`present=False` and the writer skips CSV output entirely — the
`imu_present: N` cell in the report then tells the reader no IMU file
was produced, rather than a half-populated pair of files.

Rates are mean rates in Hz across the whole video. Variable-rate
sources (some CAMM tracks) are smoothed to a single number; the
expectation per the locked spec is that real-world footage is
uniform-rate 99% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImuSamples:
    present: bool
    # (timestamp_s, x, y, z) per row. Empty when `present` is False.
    accel: list[tuple[float, float, float, float]] = field(default_factory=list)
    gyro: list[tuple[float, float, float, float]] = field(default_factory=list)
    accel_hz: float | None = None
    gyro_hz: float | None = None


def extract_imu(video_path: str | Path) -> ImuSamples:
    """Extract IMU samples + per-sensor mean rates.

    Steps:
      1. Parser(path) — fails cleanly for files with no telemetry.
      2. `telemetry()` — authoritative both-sensor presence check.
      3. `normalized_imu()` — SI-unit samples on the parser's common
         timeline. Accel / gyro tuples are extracted independently so
         a sample where one sensor is `None` only contributes to the
         other's list.
    """
    try:
        import telemetry_parser as tp
    except ImportError:
        return ImuSamples(present=False)

    try:
        parser = tp.Parser(str(video_path))
    except Exception:
        return ImuSamples(present=False)

    if not _has_both_sensors(parser):
        return ImuSamples(present=False)

    try:
        samples = parser.normalized_imu()
    except Exception:
        return ImuSamples(present=False)

    if not samples:
        return ImuSamples(present=False)

    accel: list[tuple[float, float, float, float]] = []
    gyro: list[tuple[float, float, float, float]] = []
    for s in samples:
        ts_ms = s.get("timestamp_ms")
        if ts_ms is None:
            continue
        t_s = float(ts_ms) / 1000.0
        a = s.get("accl")
        g = s.get("gyro")
        if a is not None and len(a) >= 3:
            accel.append((t_s, float(a[0]), float(a[1]), float(a[2])))
        if g is not None and len(g) >= 3:
            gyro.append((t_s, float(g[0]), float(g[1]), float(g[2])))

    # Plan §3 guardrail — even if telemetry() claimed both sensors,
    # the normalized stream may have been populated from only one.
    if not accel or not gyro:
        return ImuSamples(present=False)

    return ImuSamples(
        present=True,
        accel=accel,
        gyro=gyro,
        accel_hz=_mean_rate(accel),
        gyro_hz=_mean_rate(gyro),
    )


def _has_both_sensors(parser) -> bool:
    """Scan raw telemetry payload blocks for Gyroscope + Accelerometer keys.

    telemetry-parser labels streams consistently across vendors — GPMF
    (`Gyroscope`/`Accelerometer`), CAMM (same names), Sony / Insta360
    / DJI (same). Single-sensor containers are reported with only one
    key present in the payload dicts.
    """
    try:
        raw = parser.telemetry()
    except Exception:
        return False
    if not raw:
        return False

    has_gyro = False
    has_accel = False
    for block in raw:
        if not isinstance(block, dict):
            continue
        for key in block.keys():
            k = str(key).lower()
            if "gyro" in k:
                has_gyro = True
            if "accel" in k:
                has_accel = True
            if has_gyro and has_accel:
                return True
    return has_gyro and has_accel


def _mean_rate(rows: list[tuple[float, float, float, float]]) -> float | None:
    """Mean sampling rate in Hz, rounded to 1 decimal.

    Returns `None` when fewer than 2 samples or zero duration.
    """
    if len(rows) < 2:
        return None
    dur_s = rows[-1][0] - rows[0][0]
    if dur_s <= 0:
        return None
    return round((len(rows) - 1) / dur_s, 1)
