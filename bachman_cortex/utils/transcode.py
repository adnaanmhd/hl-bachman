"""HEVC → H.264 lossless transcoding (opt-in via --hevc-to-h264).

Preprocessing step that converts HEVC inputs to H.264 losslessly so they
pass the Phase 0 `meta_encoding` gate without perturbing downstream ML
metrics. Also strips any `rotation` tag from the output container so
mis-tagged landscape clips are treated as landscape.

Uses NVENC when an NVIDIA GPU is present, falls back to libx264 -crf 0.
"""

import shutil
import subprocess
import time
from pathlib import Path

from bachman_cortex.utils.video_metadata import get_video_metadata


def _nvenc_available() -> bool:
    """True if nvidia-smi runs successfully on this machine."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5, check=False,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def maybe_transcode_hevc_to_h264(
    video_path: str | Path,
    preprocessed_dir: Path,
) -> tuple[str, dict]:
    """If video is HEVC, transcode to H.264 losslessly. Otherwise pass through.

    Args:
        video_path: Input video path.
        preprocessed_dir: Directory where the transcoded file is written.
            Created if missing. Output file is named `<stem>.mp4`.

    Returns:
        Tuple of (active_video_path, info).

        `active_video_path` is the string path to use for the rest of the
        pipeline — the transcoded file if conversion happened, else the
        original path.

        `info` is a dict with at least:
            performed:      bool
            skipped_reason: str | None
            source_codec:   str
            source_size_mb: float
        If performed=True, also contains:
            output_codec, output_size_mb, size_multiplier,
            encoder ("nvenc" | "libx264"),
            transcode_time_sec, output_path
    """
    video_path = str(video_path)
    metadata = get_video_metadata(video_path)
    source_codec = metadata["video_codec"].lower()
    source_size_mb = metadata["file_size_mb"]

    info: dict = {
        "performed": False,
        "skipped_reason": None,
        "source_codec": source_codec,
        "source_size_mb": source_size_mb,
    }

    if source_codec != "hevc":
        info["skipped_reason"] = f"codec is {source_codec}, not hevc"
        return video_path, info

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    output_path = preprocessed_dir / f"{Path(video_path).stem}.mp4"

    use_nvenc = _nvenc_available()
    encoder = "nvenc" if use_nvenc else "libx264"

    if use_nvenc:
        video_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p7",
            "-tune", "lossless",
        ]
    else:
        video_args = [
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "veryfast",
        ]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-display_rotation:v:0", "0",
        "-i", video_path,
        *video_args,
        "-c:a", "copy",
        str(output_path),
    ]

    print(f"Transcoding HEVC → H.264 (lossless, {encoder})...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    transcode_time_sec = round(time.perf_counter() - t0, 2)

    if result.returncode != 0:
        raise RuntimeError(
            f"Lossless transcode failed (encoder={encoder}):\n{result.stderr}"
        )

    output_size_mb = round(output_path.stat().st_size / 1024 / 1024, 1)
    size_multiplier = round(output_size_mb / source_size_mb, 2) if source_size_mb else 0.0

    print(
        f"  {transcode_time_sec}s, "
        f"{source_size_mb:.1f}MB → {output_size_mb:.1f}MB "
        f"({size_multiplier}x)"
    )

    info.update({
        "performed": True,
        "output_codec": "h264",
        "output_size_mb": output_size_mb,
        "size_multiplier": size_multiplier,
        "encoder": encoder,
        "transcode_time_sec": transcode_time_sec,
        "output_path": str(output_path),
    })
    return str(output_path), info
