"""Per-frame evaluation helpers for pipeline early-stop gates."""

from bachman_cortex.models.hand_detector import HandDetection
from bachman_cortex.models.scrfd_detector import FaceDetection
from bachman_cortex.models.yolo_detector import Detection
from bachman_cortex.checks.participants import _is_wearer_body_part


def eval_face_presence(
    faces: list[FaceDetection],
    confidence_threshold: float,
) -> bool:
    """True if NO face exceeds the confidence threshold (clean frame)."""
    return all(f.confidence < confidence_threshold for f in faces)


def eval_participants(
    persons: list[Detection],
    faces: list[FaceDetection],
    hands: list[HandDetection],
    frame_h: int,
    frame_w: int,
    person_conf_threshold: float,
    face_conf_threshold: float,
    min_person_height_ratio: float,
) -> bool:
    """True if no other person detected in this frame."""
    min_person_height = frame_h * min_person_height_ratio

    for p in persons:
        if p.confidence < person_conf_threshold:
            continue
        bbox_height = p.bbox[3] - p.bbox[1]
        if bbox_height < min_person_height:
            continue
        if _is_wearer_body_part(p.bbox, hands, frame_h, frame_w):
            continue
        return False  # other person detected

    for f in faces:
        if f.confidence >= face_conf_threshold:
            return False  # face = another person

    return True
