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
