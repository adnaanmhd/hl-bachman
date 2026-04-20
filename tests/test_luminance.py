"""Tests for LuminanceAccumulator."""

import numpy as np

from bachman_cortex.checks.luminance import (
    LuminanceAccumulator,
    LuminanceThresholds,
)


def _frame(value: int, shape=(40, 40)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_classifies_into_four_buckets():
    acc = LuminanceAccumulator()
    # dead_black, too_dark, usable, blown_out
    for i, v in enumerate((5, 30, 128, 240)):
        acc.process_frame(_frame(v), i)
    r = acc.finalize()
    assert r.class_array == [0, 1, 2, 3]


def test_fails_when_good_fraction_below_threshold():
    # 70% usable, 30% blown out → fails 0.80 threshold.
    # Use a tiny flicker window with a high stddev threshold so the
    # usable→blown-out jump doesn't also get flagged as flicker.
    acc = LuminanceAccumulator(LuminanceThresholds(
        good_frame_ratio=0.80,
        flicker_window=3,
        flicker_stddev_threshold=200.0,
    ))
    for i in range(7):
        acc.process_frame(_frame(120), i)
    for i in range(3):
        acc.process_frame(_frame(245), i + 7)
    r = acc.finalize()
    assert r.pass_fail is False
    assert r.good_ratio == 0.7


def test_rolling_flicker_marks_whole_window():
    # Alternate bright/dim across a 10-frame window → stddev ≈ 60 > 30.
    acc = LuminanceAccumulator(LuminanceThresholds(
        flicker_window=10, flicker_stddev_threshold=30.0,
    ))
    for i in range(20):
        value = 200 if i % 2 == 0 else 80
        acc.process_frame(_frame(value), i)
    r = acc.finalize()
    assert all(r.flicker_array), "every frame should be inside a flicker window"


def test_sample_indices_preserve_cadence_positions():
    acc = LuminanceAccumulator()
    # Cadence: native_idx 0, 3, 6, 9 → every 3 native frames.
    for i, native_idx in enumerate([0, 3, 6, 9]):
        acc.process_frame(_frame(128), native_idx)
    r = acc.finalize()
    assert r.sample_indices == [0, 3, 6, 9]


def test_no_samples_returns_fail():
    acc = LuminanceAccumulator()
    r = acc.finalize()
    assert r.pass_fail is False
    assert r.class_array == []
    assert r.flicker_array == []
