"""Tests for HandsAccumulator — four per-frame hand-derived metrics."""

import math

import numpy as np

from bachman_cortex.checks.hand_visibility import (
    HandsAccumulator,
    HandsThresholds,
)
from bachman_cortex.models.hand_detector import (
    ContactState,
    HandDetection,
    HandSide,
)


def _hand(side: HandSide, conf: float, bbox=(100, 100, 200, 200),
          contact=ContactState.NO_CONTACT, contact_conf=0.0) -> HandDetection:
    return HandDetection(
        bbox=np.array(bbox, dtype=np.float32),
        confidence=conf,
        side=side,
        contact_state=contact,
        contact_state_confidence=contact_conf,
    )


FRAME_WH = (640, 480)


def test_both_hands_pass_requires_left_and_right():
    acc = HandsAccumulator(HandsThresholds(conf_threshold=0.7))
    # Two right hands, no left → both-hands fails, single passes.
    acc.process_frame(
        [_hand(HandSide.RIGHT, 0.9), _hand(HandSide.RIGHT, 0.85)],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    # One left, one right → both passes.
    acc.process_frame(
        [_hand(HandSide.LEFT, 0.9), _hand(HandSide.RIGHT, 0.8)],
        frame_idx=3, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.both_hands_pass == [False, True]
    assert r.both_hands_conf == [0.0, 0.9]       # max(L, R) on the passing frame
    assert r.single_hand_pass == [True, True]
    assert r.single_hand_conf[0] == 0.9           # max of the two right-hand confs


def test_low_confidence_hands_are_filtered_out():
    acc = HandsAccumulator(HandsThresholds(conf_threshold=0.7))
    acc.process_frame(
        [_hand(HandSide.LEFT, 0.6), _hand(HandSide.RIGHT, 0.6)],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.both_hands_pass == [False]
    assert r.single_hand_pass == [False]
    assert r.single_hand_conf == [0.0]


def test_hand_obj_interaction_uses_all_detected_hands():
    """Plan §1: HOI fires on ANY detected hand (not just above conf_threshold)."""
    acc = HandsAccumulator(HandsThresholds(conf_threshold=0.7))
    # Low-conf hand (0.5) with PORTABLE_OBJ should still pass HOI.
    acc.process_frame(
        [_hand(HandSide.LEFT, 0.5,
               contact=ContactState.PORTABLE_OBJ, contact_conf=0.8)],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.hand_obj_pass == [True]
    assert r.hand_obj_contact == ["P"]


def test_hand_obj_fallback_contact_when_no_interaction():
    acc = HandsAccumulator()
    acc.process_frame(
        [_hand(HandSide.LEFT, 0.9,
               contact=ContactState.SELF_CONTACT, contact_conf=0.9)],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.hand_obj_pass == [False]
    assert r.hand_obj_contact == ["S"]


def test_hand_obj_contact_is_none_when_zero_hands():
    acc = HandsAccumulator()
    acc.process_frame([], frame_idx=0, frame_wh=FRAME_WH)
    r = acc.finalize()
    assert r.hand_obj_pass == [False]
    assert r.hand_obj_contact == [None]


def test_hand_angle_centre_bbox_passes_40_deg_threshold():
    """Bbox around the centre → angle ≈ 0."""
    w, h = 640, 480
    centre_bbox = (w // 2 - 10, h // 2 - 10, w // 2 + 10, h // 2 + 10)
    acc = HandsAccumulator()
    acc.process_frame(
        [_hand(HandSide.LEFT, 0.9, bbox=centre_bbox)],
        frame_idx=0, frame_wh=(w, h),
    )
    r = acc.finalize()
    assert r.hand_angle_pass == [True]
    assert r.hand_angle_mean_deg[0] < 5.0


def test_hand_angle_corner_bbox_fails():
    """Bbox in the corner → angle approaches FOV/2 = 45°."""
    w, h = 640, 480
    corner_bbox = (0, 0, 20, 20)
    acc = HandsAccumulator()
    acc.process_frame(
        [_hand(HandSide.LEFT, 0.9, bbox=corner_bbox)],
        frame_idx=0, frame_wh=(w, h),
    )
    r = acc.finalize()
    assert r.hand_angle_pass == [False]
    assert r.hand_angle_mean_deg[0] > 40.0


def test_hand_angle_no_hands_yields_nan_and_fail():
    acc = HandsAccumulator()
    acc.process_frame([], frame_idx=0, frame_wh=FRAME_WH)
    r = acc.finalize()
    assert r.hand_angle_pass == [False]
    assert math.isnan(r.hand_angle_mean_deg[0])


def test_sample_indices_are_preserved():
    acc = HandsAccumulator()
    for idx in (0, 15, 30, 45):
        acc.process_frame([], frame_idx=idx, frame_wh=FRAME_WH)
    r = acc.finalize()
    assert r.sample_indices == [0, 15, 30, 45]
