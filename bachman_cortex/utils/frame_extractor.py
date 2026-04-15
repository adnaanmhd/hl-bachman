"""Frame extraction utility for video quality checks.

Extracts frames at a configurable rate from video files using OpenCV.
Designed for egocentric video processing pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Protocol


class _FrameProcessor(Protocol):
    """Callable-like interface for single-pass frame consumers.

    Used by `extract_frames` to tee the native-rate decode stream into
    additional analyzers (e.g. `MotionAnalyzer`) without re-opening the
    video.
    """

    frame_skip: int

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> None: ...


def _nvdec_available() -> bool:
    """True if this cv2 build has cudacodec + at least one CUDA device."""
    if not hasattr(cv2, "cudacodec"):
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_NVDEC_OK = _nvdec_available()


def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    motion_analyzer: Optional[_FrameProcessor] = None,
) -> tuple[list[np.ndarray], dict]:
    """Extract frames from a video at the specified sampling rate.

    Uses `cv2.cudacodec.VideoReader` (NVDEC) when the cv2 build has the
    `cudacodec` module and a CUDA device is present; otherwise falls back
    to `cv2.VideoCapture` (CPU decode). Output frames are BGR numpy
    arrays in either path.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to sample. Default 1.0 = 1 frame/second.
        max_frames: Maximum number of frames to extract. None = no limit.
        motion_analyzer: Optional stateful consumer receiving the native-rate
            stream at its `frame_skip` cadence. Lets the motion check skip a
            second full-video decode.

    Returns:
        Tuple of (frames, metadata) where:
        - frames: List of BGR numpy arrays (OpenCV format)
        - metadata: Dict with video_fps, total_frames, duration_s,
                     width, height, frames_extracted, extraction_time_s,
                     backend ("nvdec" or "cpu")
    """
    video_path = str(video_path)

    # Probe metadata via VideoCapture (cheap — just reads container headers).
    # cv2.cudacodec.VideoReader.format().fps is unreliable (can be NaN).
    probe = cv2.VideoCapture(video_path)
    if not probe.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    video_fps = probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    duration_s = total_frames / video_fps if video_fps > 0 else 0

    frame_interval = video_fps / fps  # native frames between samples
    motion_skip = motion_analyzer.frame_skip if motion_analyzer is not None else 0

    backend = "cpu"
    reader = None
    if _NVDEC_OK:
        try:
            reader = cv2.cudacodec.createVideoReader(video_path)
            backend = "nvdec"
        except cv2.error:
            reader = None

    frames: list[np.ndarray] = []
    t_start = time.perf_counter()
    next_target = 0.0
    frame_idx = 0
    sample_limit_reached = False

    if reader is not None:
        # NVDEC path: nextFrame() always decodes (GPU-cheap). Download
        # BGR only at sample/motion points.
        while True:
            ok, gpu_frame = reader.nextFrame()
            if not ok:
                break

            need_sample = (not sample_limit_reached) and (frame_idx >= next_target)
            need_motion = (motion_skip > 0) and (frame_idx % motion_skip == 0)

            if need_sample or need_motion:
                gpu_bgr = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGRA2BGR)
                frame = gpu_bgr.download()
                if need_sample:
                    frames.append(frame)
                    if max_frames and len(frames) >= max_frames:
                        sample_limit_reached = True
                    next_target += frame_interval
                if need_motion:
                    motion_analyzer.process_frame(frame, frame_idx)

            frame_idx += 1
            if sample_limit_reached and motion_skip == 0:
                break
    else:
        # CPU fallback: grab every native frame, retrieve only at sample
        # or motion points.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        while frame_idx < total_frames:
            if not cap.grab():
                break

            need_sample = (not sample_limit_reached) and (frame_idx >= next_target)
            need_motion = (motion_skip > 0) and (frame_idx % motion_skip == 0)

            if need_sample or need_motion:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                if need_sample:
                    frames.append(frame)
                    if max_frames and len(frames) >= max_frames:
                        sample_limit_reached = True
                    next_target += frame_interval
                if need_motion:
                    motion_analyzer.process_frame(frame, frame_idx)

            frame_idx += 1
            if sample_limit_reached and motion_skip == 0:
                break
        cap.release()

    extraction_time = time.perf_counter() - t_start

    metadata = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "width": width,
        "height": height,
        "frames_extracted": len(frames),
        "extraction_time_s": round(extraction_time, 3),
        "sampling_fps": fps,
        "backend": backend,
    }

    return frames, metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path> [fps]")
        sys.exit(1)

    video_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print(f"Extracting frames from {video_path} at {fps} FPS...")
    frames, meta = extract_frames(video_path, fps=fps)
    print(f"Extracted {meta['frames_extracted']} frames in {meta['extraction_time_s']}s")
    print(f"Video: {meta['width']}x{meta['height']}, {meta['duration_s']}s, {meta['video_fps']} FPS")
