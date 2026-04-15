"""Segment operations for the validation & processing pipeline.

Handles conversion of per-frame labels to time segments, overlap merging,
good/bad segment computation, and minimum-duration filtering.
"""

from bachman_cortex.data_types import TimeSegment, FrameLabel, CheckableSegment


def per_frame_to_bad_segments(
    frame_labels: list[FrameLabel],
    frame_interval: float = 1.0,
) -> list[TimeSegment]:
    """Convert per-frame pass/fail labels into contiguous bad TimeSegments.

    Each frame at timestamp T covers [T, T+frame_interval) seconds.
    Contiguous bad frames merge into one segment.

    Args:
        frame_labels: Per-frame pass/fail labels.
        frame_interval: Duration each frame covers (1.0 / sampling_fps).
    """
    if not frame_labels:
        return []

    bad_segments: list[TimeSegment] = []
    i = 0
    n = len(frame_labels)

    while i < n:
        if not frame_labels[i].passed:
            start = frame_labels[i].timestamp_sec
            j = i + 1
            while j < n and not frame_labels[j].passed:
                j += 1
            end = frame_labels[j - 1].timestamp_sec + frame_interval
            bad_segments.append(TimeSegment(start, end))
            i = j
        else:
            i += 1

    return bad_segments


def filter_short_bad_segments(
    bad_segments: list[TimeSegment],
    min_bad_duration: float = 2.0,
) -> list[TimeSegment]:
    """Keep bad segments with duration > min_bad_duration; forgive the rest.

    Short isolated bad segments are treated as noise and ignored.
    """
    return [s for s in bad_segments if s.duration > min_bad_duration]


def merge_bad_segments(per_check_bad: list[list[TimeSegment]]) -> list[TimeSegment]:
    """Union of bad segments across all checks, merging overlapping/adjacent segments.

    Example: face bad [85, 89), participant bad [87, 91) -> merged [85, 91).
    """
    all_bad: list[TimeSegment] = []
    for check_bad in per_check_bad:
        all_bad.extend(check_bad)

    if not all_bad:
        return []

    all_bad.sort(key=lambda s: s.start_sec)
    merged = [TimeSegment(all_bad[0].start_sec, all_bad[0].end_sec)]

    for seg in all_bad[1:]:
        if seg.start_sec <= merged[-1].end_sec:
            merged[-1] = TimeSegment(
                merged[-1].start_sec,
                max(merged[-1].end_sec, seg.end_sec),
            )
        else:
            merged.append(TimeSegment(seg.start_sec, seg.end_sec))

    return merged


def compute_good_segments(
    total_duration: float,
    bad_segments: list[TimeSegment],
) -> list[TimeSegment]:
    """Invert bad segments to get good segments over [0, total_duration]."""
    good: list[TimeSegment] = []
    cursor = 0.0

    for bad in bad_segments:
        if bad.start_sec > cursor:
            good.append(TimeSegment(cursor, bad.start_sec))
        cursor = max(cursor, bad.end_sec)

    if cursor < total_duration:
        good.append(TimeSegment(cursor, total_duration))

    return good


def filter_checkable_segments(
    good_segments: list[TimeSegment],
    min_duration: float = 60.0,
) -> tuple[list[CheckableSegment], list[TimeSegment]]:
    """Filter good segments by minimum duration.

    Returns:
        (checkable_segments, discarded_segments) where checkable_segments are
        segments >= min_duration and discarded_segments are those below.
    """
    checkable: list[CheckableSegment] = []
    discarded: list[TimeSegment] = []

    for seg in good_segments:
        if seg.duration >= min_duration:
            checkable.append(CheckableSegment(
                segment_idx=len(checkable) + 1,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
            ))
        else:
            discarded.append(seg)

    return checkable, discarded
