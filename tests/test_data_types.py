"""Smoke tests for bachman_cortex.data_types dataclasses."""

import dataclasses

from bachman_cortex import data_types as dt


def test_metadata_check_result_roundtrips_as_dict():
    m = dt.MetadataCheckResult(check="duration", status="pass",
                               accepted=">= 59.0s", detected="120.0s")
    assert dataclasses.asdict(m) == {
        "check": "duration", "status": "pass",
        "accepted": ">= 59.0s", "detected": "120.0s",
    }


def test_quality_segment_accepts_mixed_value_types():
    dt.QualitySegment(start_s=0.0, end_s=1.0, duration_s=1.0,
                      value=0.92, value_label="confidence")
    dt.QualitySegment(start_s=0.0, end_s=1.0, duration_s=1.0,
                      value="P", value_label="contact_state")
    dt.QualitySegment(start_s=0.0, end_s=1.0, duration_s=1.0,
                      value=True, value_label="obstructed")


def test_batch_report_defaults_initialize_empty_containers():
    b = dt.BatchScoreReport(
        generated_at="2026-04-17T00:00:00Z",
        video_count=0,
        total_duration_s=0.0,
        total_wall_time_s=0.0,
    )
    assert b.videos == []
    assert b.errors == []
    assert b.metadata_check_stats == {}


def test_canonical_name_tuples_are_exhaustive():
    assert "duration" in dt.METADATA_CHECKS
    assert set(dt.TECHNICAL_CHECKS) == {
        "luminance", "stability", "frozen", "pixelation"
    }
    assert set(dt.QUALITY_METRICS) == set(dt.QUALITY_VALUE_LABELS.keys())


def test_capture_device_roundtrips_as_dict():
    cd = dt.CaptureDevice(device_type="ext_camera", device_model="GoPro HERO10 Black")
    assert dataclasses.asdict(cd) == {
        "device_type": "ext_camera", "device_model": "GoPro HERO10 Black",
    }


def test_imu_info_roundtrips_as_dict():
    imu = dt.ImuInfo(present=True, accel_hz=202.7, gyro_hz=202.7)
    assert dataclasses.asdict(imu) == {
        "present": True, "accel_hz": 202.7, "gyro_hz": 202.7,
    }


def test_imu_info_absent_keeps_rates_none():
    imu = dt.ImuInfo(present=False, accel_hz=None, gyro_hz=None)
    assert imu.accel_hz is None
    assert imu.gyro_hz is None


def test_new_field_tuples_match_dataclass_fields():
    assert set(dt.CAPTURE_DEVICE_FIELDS) == {
        f.name for f in dataclasses.fields(dt.CaptureDevice)
    }
    # IMU_FIELDS uses report-column naming (imu_* prefix) — the tuple
    # is the canonical CSV column order, not a direct dataclass mirror.
    assert dt.IMU_FIELDS == ("imu_present", "imu_accel_hz", "imu_gyro_hz")


def test_video_score_report_accepts_new_optional_sections():
    report = dt.VideoScoreReport(
        video_path="/tmp/v.mp4",
        video_name="v.mp4",
        generated_at="2026-04-21T00:00:00Z",
        processing_wall_time_s=0.0,
        duration_s=60.0,
    )
    assert report.capture_device is None
    assert report.imu is None
