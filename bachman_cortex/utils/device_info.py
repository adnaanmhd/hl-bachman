"""Capture-device type + model extraction.

Vendor registry for `device_type` and `device_model`. Detection order:

1. `telemetry-parser` — covers GoPro, Sony, Insta360, DJI, Blackmagic,
   RED, and most "telemetry-bearing" action cams. When this resolves,
   `device_type` is always `"ext_camera"` (those vendors never ship
   phones).
2. ffprobe `com.apple.quicktime.*` tags — Apple / iPhone.
3. ffprobe `com.android.*` tags — Android phone. If `manufacturer` +
   `model` are present they become the combined string; bare
   `com.android.version` with no model yields
   `device_type="phone", device_model="Unknown"` (very common on
   third-party Android apps).
4. Everything else → `Unknown, Unknown`.

Re-encoded files are a documented limitation (see `checks.md`): ffmpeg
re-encoding strips most vendor tags, so a re-encoded GoPro clip reads
as a generic `Lavf`-encoded mp4 with no identifying markers. The
registry reports `Unknown` in that case rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureDeviceInfo:
    device_type: str    # "ext_camera" | "phone" | "Unknown"
    device_model: str   # combined string or "Unknown"


_UNKNOWN = "Unknown"


def detect_capture_device(
    video_path: str | Path,
    tag_surface: dict[str, Any],
) -> CaptureDeviceInfo:
    """Resolve capture-device type + model.

    `tag_surface` is the same dict produced by
    `video_metadata.collect_tag_surface` — the flattened ffprobe tag
    view reused across the observations registry.
    """
    # ── 1. telemetry-parser (camera vendors) ──────────────────────────
    camera, model = _telemetry_parser_probe(video_path)
    if camera and model:
        combined = f"{camera} {model}".strip() if camera.lower() not in model.lower() \
            else model.strip()
        return CaptureDeviceInfo(device_type="ext_camera", device_model=combined)
    if model:
        return CaptureDeviceInfo(device_type="ext_camera", device_model=model.strip())
    if camera:
        # Camera make known but model missing — still an ext_camera.
        return CaptureDeviceInfo(device_type="ext_camera", device_model=camera.strip())

    video_tags = tag_surface.get("video_tags", {}) or {}
    format_tags = tag_surface.get("format_tags", {}) or {}

    # ── 2. Apple iPhone ───────────────────────────────────────────────
    apple = _apple_model(format_tags, video_tags)
    if apple is not None:
        return CaptureDeviceInfo(device_type="phone", device_model=apple)

    # ── 3. Android phone ──────────────────────────────────────────────
    android = _android_model(format_tags, video_tags)
    if android is not None:
        return CaptureDeviceInfo(device_type="phone", device_model=android)

    # ── 4. Unknown ────────────────────────────────────────────────────
    return CaptureDeviceInfo(device_type=_UNKNOWN, device_model=_UNKNOWN)


def _telemetry_parser_probe(video_path: str | Path) -> tuple[str | None, str | None]:
    """Return `(camera_make, camera_model)` or `(None, None)` on failure.

    Silences every failure path — unsupported-format, missing library,
    parser exceptions — because the caller treats "no signal" as the
    trigger to move to the next registry entry.
    """
    try:
        import telemetry_parser as tp
    except ImportError:
        return (None, None)
    try:
        p = tp.Parser(str(video_path))
    except Exception:
        return (None, None)
    camera = getattr(p, "camera", None)
    model = getattr(p, "model", None)
    camera = camera.strip() if isinstance(camera, str) and camera.strip() else None
    model = model.strip() if isinstance(model, str) and model.strip() else None
    return (camera, model)


def _apple_model(format_tags: dict, video_tags: dict) -> str | None:
    """Extract an Apple device-model string from ffprobe tags.

    Priority: `com.apple.quicktime.model` (usually `"iPhone 14 Pro"`).
    Falls back to presence-only signal when `com.apple.quicktime.*`
    tags exist but `model` is missing — in that case we return
    `"Unknown"` under device_type=phone, which the caller will pair
    with device_type=phone.
    """
    model = _find_key(format_tags, "com.apple.quicktime.model") \
        or _find_key(video_tags, "com.apple.quicktime.model")
    if model:
        make = _find_key(format_tags, "com.apple.quicktime.make") \
            or _find_key(video_tags, "com.apple.quicktime.make") \
            or "Apple"
        if make.lower() in model.lower():
            return model.strip()
        return f"{make.strip()} {model.strip()}"

    # No model, but any com.apple.quicktime.* tag means Apple device.
    if _has_prefix(format_tags, "com.apple.quicktime.") \
            or _has_prefix(video_tags, "com.apple.quicktime."):
        return _UNKNOWN
    return None


def _android_model(format_tags: dict, video_tags: dict) -> str | None:
    """Extract an Android device-model string.

    Full form: `com.android.manufacturer` + `com.android.model`.
    Partial form (very common on third-party Android recorders):
    just `com.android.version` present, no manufacturer/model —
    returns `"Unknown"` so the caller can still report
    `device_type=phone`.
    """
    make = _find_key(format_tags, "com.android.manufacturer") \
        or _find_key(video_tags, "com.android.manufacturer")
    model = _find_key(format_tags, "com.android.model") \
        or _find_key(video_tags, "com.android.model")
    if make and model:
        return f"{make.strip()} {model.strip()}"
    if model:
        return model.strip()
    if make:
        return make.strip()

    if _has_prefix(format_tags, "com.android.") \
            or _has_prefix(video_tags, "com.android."):
        return _UNKNOWN
    return None


def _find_key(d: dict, key: str) -> str | None:
    """Case-insensitive key lookup returning a stripped string or None."""
    if not d:
        return None
    key_l = key.lower()
    for k, v in d.items():
        if (k or "").lower() == key_l:
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _has_prefix(d: dict, prefix: str) -> bool:
    if not d:
        return False
    prefix_l = prefix.lower()
    return any((k or "").lower().startswith(prefix_l) for k in d.keys())
