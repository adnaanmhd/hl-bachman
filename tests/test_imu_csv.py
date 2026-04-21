"""Tests for utils.imu_csv — the per-sensor CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path

from bachman_cortex.utils.imu_csv import write_imu_csvs
from bachman_cortex.utils.imu_extraction import ImuSamples


def _samples(present=True):
    return ImuSamples(
        present=present,
        accel=[(0.0, 1.0, 2.0, 9.81), (0.005, 1.01, 2.01, 9.82)],
        gyro=[(0.0, 0.1, 0.2, -0.3), (0.005, 0.11, 0.21, -0.29)],
        accel_hz=200.0, gyro_hz=200.0,
    )


def test_write_imu_csvs_produces_both_files(tmp_path):
    paths = write_imu_csvs(_samples(), tmp_path, "video.mp4")
    assert "accel_csv" in paths and "gyro_csv" in paths
    assert paths["accel_csv"] == tmp_path / "video_accel.csv"
    assert paths["gyro_csv"] == tmp_path / "video_gyro.csv"
    assert paths["accel_csv"].exists()
    assert paths["gyro_csv"].exists()


def test_write_imu_csvs_accel_header_and_rows(tmp_path):
    write_imu_csvs(_samples(), tmp_path, "video.mp4")
    with (tmp_path / "video_accel.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp_s", "ax", "ay", "az"]
    assert rows[1] == ["0.000000", "1.000000", "2.000000", "9.810000"]
    assert rows[2] == ["0.005000", "1.010000", "2.010000", "9.820000"]


def test_write_imu_csvs_gyro_header_and_rows(tmp_path):
    write_imu_csvs(_samples(), tmp_path, "video.mp4")
    with (tmp_path / "video_gyro.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp_s", "gx", "gy", "gz"]
    assert len(rows) == 3


def test_write_imu_csvs_skips_when_absent(tmp_path):
    paths = write_imu_csvs(_samples(present=False), tmp_path, "video.mp4")
    assert paths == {}
    assert not (tmp_path / "video_accel.csv").exists()
    assert not (tmp_path / "video_gyro.csv").exists()


def test_write_imu_csvs_uses_video_stem(tmp_path):
    """Filename stem must match `{video_name}.json` / `{video_name}.parquet`."""
    paths = write_imu_csvs(_samples(), tmp_path, "GH011093.MP4")
    assert paths["accel_csv"].name == "GH011093_accel.csv"
    assert paths["gyro_csv"].name == "GH011093_gyro.csv"
