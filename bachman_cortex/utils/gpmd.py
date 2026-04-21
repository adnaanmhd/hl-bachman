"""GoPro GPMD timed-metadata stream — presence detector + highlights parser.

GoPro clips carry a private timed-metadata stream (`handler_name` equal
to "GoPro MET" or the handler type is `meta` with a `gpmd` codec tag)
that encodes HyperSmooth state, lens preset, gyro, accelerometer, GPS
and more in the KLV-style GPMF format.

Two entry points:

- `detect_gpmd_stream(raw_streams)` — cheap ffprobe-only presence check.
  Used by the stabilization / FOV registries when the video path is not
  handy (e.g. during tag-surface inspection).
- `parse_gpmd_highlights(video_path)` — reads the GPMF "Highlights"
  settings block via telemetry-parser and decodes the FourCC-keyed
  entries we care about (camera model, lens preset, horizontal FOV,
  HyperSmooth / EIS state). The FourCC scan is the supplementary
  scanner sitting between telemetry-parser's high-level accessors
  (`.camera`, `.model`) and the presence-based fallbacks in the
  observations registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpmdInfo:
    """Presence-only info surfaced from ffprobe."""
    present: bool
    stream_index: int | None           # index into the ffprobe streams list, or None
    handler_name: str                  # raw handler_name of the detected stream


@dataclass(frozen=True)
class GpmdHighlights:
    """Decoded settings from the GPMF `Highlights` block.

    All fields are optional — different firmware generations surface
    different subsets. Callers MUST treat `None` as "unknown" and fall
    back to presence-based detection.
    """
    present: bool
    camera_model: str | None = None        # e.g. "HERO10 Black"
    lens_label: str | None = None          # "Wide" / "Linear" / "SuperView" / "Narrow" / "HyperView"
    fov_deg: float | None = None           # horizontal FOV in degrees (ZFOV)
    hypersmooth_state: str | None = None   # "OFF" / "LOW" / "HIGH" / "BOOST" / "Y" / "N"


_GPMD_HANDLER_NEEDLES = ("gopro met", "gpmd")


def detect_gpmd_stream(raw_streams: list[dict]) -> GpmdInfo:
    """Return presence info for the GoPro GPMD stream.

    Detection rule: any stream whose `tags.handler_name` contains
    "GoPro MET" (case-insensitive) or whose `codec_tag_string` is
    `gpmd`. Both signals mean the GPMF-encoded metadata is available
    on this file.

    `stream_index` is the flat index in `data["streams"]`, suitable
    for `-map 0:<index>` when the real parser lands.
    """
    for i, s in enumerate(raw_streams or []):
        handler = ((s.get("tags") or {}).get("handler_name", "") or "").lower()
        codec_tag = (s.get("codec_tag_string", "") or "").lower()
        if any(needle in handler for needle in _GPMD_HANDLER_NEEDLES) \
                or codec_tag == "gpmd":
            return GpmdInfo(
                present=True,
                stream_index=i,
                handler_name=handler,
            )
    return GpmdInfo(present=False, stream_index=None, handler_name="")


# ── FourCC → field decoder ────────────────────────────────────────────────

# The GPMF spec keys we care about. telemetry-parser's `.telemetry()[0]`
# returns the Highlights settings block with FourCC codes serialised as
# hex strings (e.g. "0x56464f56"). These constants are the lowercase
# hex form for direct dict lookup.
_FOURCC_CAMERA_MODEL = "0x4d494e46"   # "MINF" — internal model name
_FOURCC_VFOV = "0x56464f56"            # "VFOV" — lens preset label (single char)
_FOURCC_ZFOV = "0x5a464f56"            # "ZFOV" — horizontal FOV in degrees
_FOURCC_EISE = "0x45495345"            # "EISE" — EIS enable flag ("Y" / "N")
_FOURCC_EISA = "0x45495341"            # "EISA" — EIS active level (legacy)
_FOURCC_HSGT = "0x48534754"            # "HSGT" — HyperSmooth gate: OFF/LOW/HIGH/BOOST


_VFOV_LABELS = {
    "W": "Wide",
    "L": "Linear",
    "S": "SuperView",
    "N": "Narrow",
    "H": "HyperView",
    "M": "Max SuperView",
}


def parse_gpmd_highlights(video_path: str | Path) -> GpmdHighlights:
    """Decode the GoPro GPMF Highlights settings block.

    Uses `telemetry-parser` to read the first telemetry payload
    (which, on HERO7+ firmware, is the per-clip "Highlights" block
    containing the capture settings). When the file has no GPMF
    stream, telemetry-parser raises `OSError: Unsupported file
    format` — we return `GpmdHighlights(present=False)` and let the
    caller fall back to presence-based labels.

    Any error inside the parser (malformed KLV, unsupported firmware)
    also yields `present=False` — the caller is responsible for the
    fallback path, never for handling parser errors.
    """
    try:
        import telemetry_parser as tp
    except ImportError:
        return GpmdHighlights(present=False)

    try:
        parser = tp.Parser(str(video_path))
        raw = parser.telemetry()
    except Exception:
        return GpmdHighlights(present=False)

    if not raw or not isinstance(raw, list):
        return GpmdHighlights(present=False)

    # The Highlights block is the first payload on GoPro firmware.
    first = raw[0] if isinstance(raw[0], dict) else {}
    settings = first.get("Default") if isinstance(first.get("Default"), dict) else {}
    if not settings:
        # Some firmware places settings in block index 1. Try that once.
        if len(raw) > 1 and isinstance(raw[1], dict):
            settings = raw[1].get("Default") or {}
    if not settings:
        return GpmdHighlights(present=False)

    camera_model = _extract_camera_model(parser, settings)
    lens_label = _extract_lens_label(settings)
    fov_deg = _extract_fov_deg(settings)
    hypersmooth_state = _extract_hypersmooth_state(settings)

    return GpmdHighlights(
        present=True,
        camera_model=camera_model,
        lens_label=lens_label,
        fov_deg=fov_deg,
        hypersmooth_state=hypersmooth_state,
    )


def _extract_camera_model(parser, settings: dict) -> str | None:
    """Prefer `telemetry_parser.Parser.model` (high-level); fall back to
    the internal `MINF` FourCC in the Highlights block."""
    model = getattr(parser, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    minf = settings.get(_FOURCC_CAMERA_MODEL)
    if isinstance(minf, str) and minf.strip():
        return minf.strip()
    # Some firmware stashes the model in settings["Name"].
    name = settings.get("Name")
    if isinstance(name, str) and name.strip() and name.strip() != "Highlights":
        return name.strip()
    return None


def _extract_lens_label(settings: dict) -> str | None:
    raw = settings.get(_FOURCC_VFOV)
    if raw is None:
        return None
    label = str(raw).strip().upper()
    if not label:
        return None
    return _VFOV_LABELS.get(label[0], label)


def _extract_fov_deg(settings: dict) -> float | None:
    raw = settings.get(_FOURCC_ZFOV)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    return round(val, 2)


def _extract_hypersmooth_state(settings: dict) -> str | None:
    """Decode the HyperSmooth / EIS state into a single string.

    Priority: HSGT (newer firmware, named levels) → EISA (legacy named
    levels) → EISE (boolean enable). Returns `None` when none of the
    three keys are present or decodable.
    """
    hsgt = settings.get(_FOURCC_HSGT)
    if isinstance(hsgt, str) and hsgt.strip():
        return hsgt.strip().upper()
    eisa = settings.get(_FOURCC_EISA)
    if isinstance(eisa, str) and eisa.strip() and eisa.strip().upper() != "N/A":
        return eisa.strip().upper()
    eise = settings.get(_FOURCC_EISE)
    if isinstance(eise, str) and eise.strip():
        return eise.strip().upper()
    return None
