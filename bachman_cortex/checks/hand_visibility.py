"""Hand-based quality metrics derived from Hands23 per-frame detections.

A single `HandsAccumulator` consumes Hands23 detections at the quality
cadence and emits four per-sampled-frame signals:
  - both_hands_visibility (pass + max(L_conf, R_conf))
  - single_hand_visibility (pass + max confidence)
  - hand_obj_interaction (pass + contact char, or None if no hands)
  - hand_angle (pass + mean angle in degrees, NaN if no hands)

Pass/fail is per-frame only; aggregation into percent_frames and
segments happens at report time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bachman_cortex.models.hand_detector import (
    ContactState,
    HandDetection,
    HandSide,
)


# ── Per-frame result + thresholds ──────────────────────────────────────────

@dataclass(frozen=True)
class HandsThresholds:
    conf_threshold: float = 0.7
    max_angle_deg: float = 40.0
    diagonal_fov_degrees: float = 90.0


_CONTACT_CHAR: dict[ContactState, str] = {
    ContactState.NO_CONTACT: "N",
    ContactState.SELF_CONTACT: "S",
    ContactState.OTHER_PERSON: "O",
    ContactState.PORTABLE_OBJ: "P",
    ContactState.STATIONARY_OBJ: "F",   # fixed/stationary → "F"
}

_INTERACTION_STATES: frozenset[ContactState] = frozenset({
    ContactState.PORTABLE_OBJ,
    ContactState.STATIONARY_OBJ,
})


@dataclass
class HandsFinalizeResult:
    """Per-sampled-frame arrays for the four hand-based metrics."""
    both_hands_pass: list[bool]
    both_hands_conf: list[float]
    single_hand_pass: list[bool]
    single_hand_conf: list[float]
    hand_obj_pass: list[bool]
    hand_obj_contact: list[str | None]
    hand_angle_pass: list[bool]
    hand_angle_mean_deg: list[float]    # NaN when no hands detected
    sample_indices: list[int]


# ── Geometry ───────────────────────────────────────────────────────────────

def _hand_angle(bbox: np.ndarray, frame_w: int, frame_h: int,
                fov_deg: float) -> float:
    hx = (bbox[0] + bbox[2]) / 2
    hy = (bbox[1] + bbox[3]) / 2
    cx, cy = frame_w / 2, frame_h / 2
    pixel_dist = math.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
    half_diag = math.sqrt(cx ** 2 + cy ** 2)
    if half_diag <= 0:
        return 0.0
    return (pixel_dist / half_diag) * (fov_deg / 2.0)


# ── Accumulator ────────────────────────────────────────────────────────────

@dataclass
class HandsAccumulator:
    thresholds: HandsThresholds = field(default_factory=HandsThresholds)

    _both_pass: list[bool] = field(init=False, default_factory=list)
    _both_conf: list[float] = field(init=False, default_factory=list)
    _single_pass: list[bool] = field(init=False, default_factory=list)
    _single_conf: list[float] = field(init=False, default_factory=list)
    _hoi_pass: list[bool] = field(init=False, default_factory=list)
    _hoi_contact: list[str | None] = field(init=False, default_factory=list)
    _angle_pass: list[bool] = field(init=False, default_factory=list)
    _angle_mean: list[float] = field(init=False, default_factory=list)
    _sample_indices: list[int] = field(init=False, default_factory=list)

    def process_frame(
        self,
        hands: list[HandDetection],
        frame_idx: int,
        frame_wh: tuple[int, int],
    ) -> None:
        """Append per-frame results for one sampled frame."""
        th = self.thresholds
        frame_w, frame_h = frame_wh

        kept = [h for h in hands if h.confidence >= th.conf_threshold]

        has_left = any(h.side == HandSide.LEFT for h in kept)
        has_right = any(h.side == HandSide.RIGHT for h in kept)
        left_conf = max((h.confidence for h in kept
                         if h.side == HandSide.LEFT), default=0.0)
        right_conf = max((h.confidence for h in kept
                          if h.side == HandSide.RIGHT), default=0.0)

        both_pass = has_left and has_right
        self._both_pass.append(both_pass)
        self._both_conf.append(max(left_conf, right_conf) if both_pass else 0.0)

        single_pass = bool(kept)
        self._single_pass.append(single_pass)
        self._single_conf.append(
            max((h.confidence for h in kept), default=0.0) if single_pass else 0.0
        )

        # Hand-object interaction: use ALL detected hands (not just high-conf),
        # because HOI cares about contact state, not hand-detection confidence.
        interacting = [h for h in hands if h.contact_state in _INTERACTION_STATES]
        if interacting:
            best = max(interacting, key=lambda h: h.contact_state_confidence)
            self._hoi_pass.append(True)
            self._hoi_contact.append(_CONTACT_CHAR.get(best.contact_state))
        else:
            self._hoi_pass.append(False)
            if hands:
                fallback = max(hands, key=lambda h: h.confidence)
                self._hoi_contact.append(
                    _CONTACT_CHAR.get(fallback.contact_state, "N")
                )
            else:
                self._hoi_contact.append(None)

        # Hand angle: mean over detected hands. Zero hands → fail + NaN.
        if hands:
            angles = [
                _hand_angle(h.bbox, frame_w, frame_h, th.diagonal_fov_degrees)
                for h in hands
            ]
            mean_angle = float(np.mean(angles))
            all_within = all(a <= th.max_angle_deg for a in angles)
            self._angle_pass.append(all_within)
            self._angle_mean.append(mean_angle)
        else:
            self._angle_pass.append(False)
            self._angle_mean.append(float("nan"))

        self._sample_indices.append(frame_idx)

    def finalize(self) -> HandsFinalizeResult:
        return HandsFinalizeResult(
            both_hands_pass=list(self._both_pass),
            both_hands_conf=list(self._both_conf),
            single_hand_pass=list(self._single_pass),
            single_hand_conf=list(self._single_conf),
            hand_obj_pass=list(self._hoi_pass),
            hand_obj_contact=list(self._hoi_contact),
            hand_angle_pass=list(self._angle_pass),
            hand_angle_mean_deg=list(self._angle_mean),
            sample_indices=list(self._sample_indices),
        )
