"""Shared data structures for the validation & processing pipeline."""

from dataclasses import dataclass, field

from bachman_cortex.checks.check_results import CheckResult


@dataclass
class TimeSegment:
    """A contiguous time range in seconds."""
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class FrameLabel:
    """Per-frame pass/fail result for a single check."""
    frame_idx: int
    timestamp_sec: float
    passed: bool
    confidence: float = 0.0
    labels: list[str] | None = None


@dataclass
class CheckFrameResults:
    """Per-frame results for one check across all frames."""
    check_name: str
    frame_labels: list[FrameLabel]
    bad_segments: list[TimeSegment]


@dataclass
class CheckableSegment:
    """A contiguous good segment from Phase 1, eligible for Phase 2."""
    segment_idx: int
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class SegmentValidationResult:
    """Phase 2 result for a single checkable segment."""
    segment: CheckableSegment
    passed: bool
    check_results: dict[str, CheckResult]
    failing_checks: list[str] = field(default_factory=list)


@dataclass
class VideoProcessingResult:
    """Complete processing result for one video."""
    video_path: str
    video_name: str
    original_duration_sec: float
    metadata: dict

    # Phase 0
    metadata_passed: bool
    metadata_results: dict[str, CheckResult]

    # Phase 1
    phase1_check_frame_results: list[CheckFrameResults]
    phase1_bad_segments: list[TimeSegment]
    phase1_discarded_segments: list[TimeSegment]
    prefiltered_segments: list[CheckableSegment]

    # Phase 2
    segment_results: list[SegmentValidationResult]
    usable_segments: list[CheckableSegment]
    rejected_segments: list[CheckableSegment]

    # Phase 3
    usable_duration_sec: float
    unusable_duration_sec: float
    yield_ratio: float

    processing_time_sec: float = 0.0
    error: str | None = None

    # Preprocessing info (populated only when --hevc-to-h264 is active)
    transcode_info: dict | None = None
