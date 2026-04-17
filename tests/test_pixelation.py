"""Tests for PixelationAccumulator."""

import numpy as np

from bachman_cortex.checks.pixelation import (
    PixelationAccumulator,
    PixelationThresholds,
    compute_blockiness,
)


def _smooth_frame(h=64, w=64) -> np.ndarray:
    """Smooth gradient → blockiness ≈ 1.0."""
    gx = np.linspace(0, 255, w, dtype=np.uint8)
    gy = np.linspace(0, 255, h, dtype=np.uint8)
    gray = (gx[None, :] + gy[:, None]) // 2
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def _blocky_frame(block_size=8, h=64, w=64) -> np.ndarray:
    """Discrete 8x8 blocks of uniform value → strong boundary edges."""
    rng = np.random.default_rng(0)
    block_rows = h // block_size
    block_cols = w // block_size
    block_vals = rng.integers(0, 255, size=(block_rows, block_cols), dtype=np.uint8)
    gray = np.repeat(np.repeat(block_vals, block_size, axis=0),
                     block_size, axis=1)
    return np.stack([gray, gray, gray], axis=-1)


def test_blocky_frames_exceed_threshold():
    score = compute_blockiness(_blocky_frame(), block_size=8)
    assert score > 2.0


def test_smooth_frames_stay_near_one():
    score = compute_blockiness(_smooth_frame(), block_size=8)
    assert 0.8 <= score <= 1.5


def test_accumulator_pass_on_mostly_smooth():
    acc = PixelationAccumulator(
        PixelationThresholds(good_frame_ratio=0.80, max_blockiness_ratio=1.5)
    )
    for i in range(9):
        acc.process_frame(_smooth_frame(), i)
    acc.process_frame(_blocky_frame(), 9)
    r = acc.finalize()
    assert r.pass_fail is True
    assert r.good_ratio >= 0.9


def test_accumulator_fail_on_mostly_blocky():
    acc = PixelationAccumulator(
        PixelationThresholds(good_frame_ratio=0.80, max_blockiness_ratio=1.5)
    )
    for i in range(8):
        acc.process_frame(_blocky_frame(), i)
    for i in range(8, 10):
        acc.process_frame(_smooth_frame(), i)
    r = acc.finalize()
    assert r.pass_fail is False
    assert r.good_ratio <= 0.4


def test_empty_accumulator_fails():
    r = PixelationAccumulator().finalize()
    assert r.pass_fail is False
    assert r.ratio_array == []
