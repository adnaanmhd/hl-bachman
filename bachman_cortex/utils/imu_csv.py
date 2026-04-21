"""IMU CSV writer.

Plan Q5 choice (option c): one file per sensor, each at its native
cadence. Written as `{video_name}_accel.csv` and `{video_name}_gyro.csv`
alongside `report.md` inside the per-video output directory.

Only called when `ImuSamples.present` is True. A missing-sensor video
yields no CSV output and `imu_present: N` in the report — see
`imu_extraction.extract_imu`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from bachman_cortex.utils.imu_extraction import ImuSamples


_ACCEL_HEADER = ("timestamp_s", "ax", "ay", "az")
_GYRO_HEADER = ("timestamp_s", "gx", "gy", "gz")


def write_imu_csvs(
    samples: ImuSamples,
    out_dir: str | Path,
    video_name: str,
) -> dict[str, Path]:
    """Write `{stem}_accel.csv` and `{stem}_gyro.csv` into `out_dir`.

    `video_name` is the raw file name (e.g. `"GH011093.MP4"`); the
    stem (without extension) is used for the CSV filenames so they
    line up with `{stem}.json` / `{stem}.parquet` already produced by
    the reporting module.

    Returns `{"accel_csv": Path, "gyro_csv": Path}` on success, or an
    empty dict when `samples.present` is False.
    """
    if not samples.present:
        return {}

    stem = Path(video_name).stem
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    accel_csv = out_path / f"{stem}_accel.csv"
    gyro_csv = out_path / f"{stem}_gyro.csv"

    _write_rows(accel_csv, _ACCEL_HEADER, samples.accel)
    _write_rows(gyro_csv, _GYRO_HEADER, samples.gyro)

    return {"accel_csv": accel_csv, "gyro_csv": gyro_csv}


def _write_rows(
    path: Path,
    header: tuple[str, ...],
    rows: list[tuple[float, float, float, float]],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t, x, y, z in rows:
            writer.writerow((
                f"{t:.6f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
            ))
