"""Tests for utils.imu_extraction — parser wrapper + rate math."""

from __future__ import annotations

import sys
import types

from bachman_cortex.utils import imu_extraction as imu
from bachman_cortex.utils.imu_extraction import ImuSamples


# ── _mean_rate ────────────────────────────────────────────────────────────

def test_mean_rate_uniform_spacing():
    rows = [(i * 0.005, 0.0, 0.0, 0.0) for i in range(201)]
    assert imu._mean_rate(rows) == 200.0


def test_mean_rate_zero_duration_returns_none():
    rows = [(0.0, 1.0, 2.0, 3.0), (0.0, 1.0, 2.0, 3.0)]
    assert imu._mean_rate(rows) is None


def test_mean_rate_single_row_returns_none():
    assert imu._mean_rate([(0.0, 1.0, 2.0, 3.0)]) is None


def test_mean_rate_empty_returns_none():
    assert imu._mean_rate([]) is None


# ── Fake telemetry-parser harness ──────────────────────────────────────────

def _install(monkeypatch, *, telemetry_payload, normalized_samples,
             raise_on_init=False):
    fake = types.ModuleType("telemetry_parser")

    def _factory(path):
        if raise_on_init:
            raise OSError("Unsupported file format")
        obj = types.SimpleNamespace(path=path)
        obj.telemetry = lambda: telemetry_payload
        obj.normalized_imu = lambda: normalized_samples
        return obj

    fake.Parser = _factory
    monkeypatch.setitem(sys.modules, "telemetry_parser", fake)


def _sample(t_ms, gyro=(0.0, 0.0, 0.0), accl=(0.0, 0.0, 9.81)):
    return {"timestamp_ms": t_ms, "gyro": gyro, "accl": accl, "magn": None}


def test_extract_imu_happy_path(monkeypatch):
    telemetry = [{"Gyroscope": {"Data": []}, "Accelerometer": {"Data": []}}]
    samples = [_sample(i * 5.0) for i in range(201)]
    _install(monkeypatch, telemetry_payload=telemetry, normalized_samples=samples)

    result = imu.extract_imu("/tmp/f.mp4")
    assert result.present
    assert result.accel_hz == 200.0
    assert result.gyro_hz == 200.0
    assert len(result.accel) == 201
    assert len(result.gyro) == 201
    assert result.accel[0] == (0.0, 0.0, 0.0, 9.81)


def test_extract_imu_missing_gyro_reports_absent(monkeypatch):
    telemetry = [{"Accelerometer": {"Data": []}}]  # gyro missing
    samples = [_sample(i * 5.0, gyro=None) for i in range(10)]
    _install(monkeypatch, telemetry_payload=telemetry, normalized_samples=samples)
    result = imu.extract_imu("/tmp/f.mp4")
    assert not result.present
    assert result.accel == []
    assert result.gyro == []


def test_extract_imu_missing_accel_reports_absent(monkeypatch):
    telemetry = [{"Gyroscope": {"Data": []}}]  # accel missing
    samples = [_sample(i * 5.0, accl=None) for i in range(10)]
    _install(monkeypatch, telemetry_payload=telemetry, normalized_samples=samples)
    result = imu.extract_imu("/tmp/f.mp4")
    assert not result.present


def test_extract_imu_unsupported_file_returns_absent(monkeypatch):
    _install(monkeypatch, telemetry_payload=None, normalized_samples=None,
             raise_on_init=True)
    result = imu.extract_imu("/tmp/f.mp4")
    assert not result.present
    assert result.accel_hz is None


def test_extract_imu_empty_normalized_returns_absent(monkeypatch):
    telemetry = [{"Gyroscope": {"Data": []}, "Accelerometer": {"Data": []}}]
    _install(monkeypatch, telemetry_payload=telemetry, normalized_samples=[])
    result = imu.extract_imu("/tmp/f.mp4")
    assert not result.present


def test_extract_imu_normalized_with_null_sensor_still_requires_both(monkeypatch):
    """telemetry() claims both sensors but normalized only gives accel."""
    telemetry = [{"Gyroscope": {"Data": []}, "Accelerometer": {"Data": []}}]
    samples = [_sample(i * 5.0, gyro=None) for i in range(10)]
    _install(monkeypatch, telemetry_payload=telemetry, normalized_samples=samples)
    result = imu.extract_imu("/tmp/f.mp4")
    # Accel list populated but gyro list empty → must report absent.
    assert not result.present


def test_extract_imu_library_missing_returns_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "telemetry_parser", None)
    result = imu.extract_imu("/tmp/f.mp4")
    assert not result.present
