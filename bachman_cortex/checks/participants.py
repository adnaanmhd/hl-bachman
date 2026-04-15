"""Participants: wearer-vs-other heuristics shared by egocentric checks."""

import numpy as np
from bachman_cortex.models.hand_detector import HandDetection


def _is_wearer_body_part(
    person_bbox: np.ndarray,
    hand_detections: list[HandDetection],
    frame_height: int,
    frame_width: int,
    hand_overlap_threshold: float = 0.3,
    edge_margin: int = 10,
    edge_min_height_ratio: float = 0.50,
    edge_max_aspect_ratio: float = 0.50,
) -> bool:
    """Check if a person detection is likely the camera wearer's own body.

    Heuristics for wearer-origin detections in egocentric video:
      A. Bottom-center anchored (arm/torso entering from below).
      B. Edge-anchored AND tall-and-narrow (arm reaching in from the side).
         The narrow constraint (width/height <= edge_max_aspect_ratio, default
         0.5) prevents mis-classifying a wide full-body subject whose bbox
         happens to touch an edge.
      C. Overlaps significantly with a detected hand.
    """
    px1, py1, px2, py2 = person_bbox
    person_center_x = (px1 + px2) / 2
    person_bottom = py2
    person_height = py2 - py1
    person_width = px2 - px1

    # Rule A: bottom-center anchored
    if (person_bottom > frame_height * 0.85
            and frame_width * 0.2 < person_center_x < frame_width * 0.8):
        return True

    # Rule B: edge-anchored AND tall-and-narrow
    if (person_height >= frame_height * edge_min_height_ratio
            and (px1 < edge_margin or px2 > frame_width - edge_margin)
            and person_width <= person_height * edge_max_aspect_ratio):
        return True

    # Rule C: overlaps significantly with a detected hand
    person_area = max((px2 - px1) * (py2 - py1), 1)
    for hand in hand_detections:
        hx1, hy1, hx2, hy2 = hand.bbox
        overlap_x1 = max(px1, hx1)
        overlap_y1 = max(py1, hy1)
        overlap_x2 = min(px2, hx2)
        overlap_y2 = min(py2, hy2)
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            if overlap_area / person_area > hand_overlap_threshold:
                return True

    return False
