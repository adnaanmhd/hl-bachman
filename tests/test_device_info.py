"""Tests for utils.device_info — the capture-device registry."""

from __future__ import annotations

import sys
import types

from bachman_cortex.utils import device_info as di


def _install_fake_parser(monkeypatch, parser_factory):
    fake = types.ModuleType("telemetry_parser")
    fake.Parser = parser_factory
    monkeypatch.setitem(sys.modules, "telemetry_parser", fake)


def _make_parser(camera=None, model=None):
    def _factory(path):
        obj = types.SimpleNamespace(path=path)
        obj.camera = camera
        obj.model = model
        return obj
    return _factory


def _tag_surface(video_tags=None, format_tags=None, handler_names=None):
    return {
        "video_tags": video_tags or {},
        "format_tags": format_tags or {},
        "all_stream_tags": [],
        "all_stream_handler_names": handler_names or [],
        "side_data_types": [],
        "codec_tag_string": "",
        "encoder": "",
    }


# ── telemetry-parser-led detection (ext_camera) ───────────────────────────

def test_detect_ext_camera_combines_make_and_model(monkeypatch):
    _install_fake_parser(
        monkeypatch,
        _make_parser(camera="Sony", model="ILCE-7M4"),
    )
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface())
    assert info.device_type == "ext_camera"
    assert info.device_model == "Sony ILCE-7M4"


def test_detect_ext_camera_skips_make_duplication(monkeypatch):
    """If the model string already carries the make, don't double it."""
    _install_fake_parser(
        monkeypatch,
        _make_parser(camera="GoPro", model="GoPro HERO10 Black"),
    )
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface())
    assert info.device_model == "GoPro HERO10 Black"


def test_detect_ext_camera_model_only(monkeypatch):
    _install_fake_parser(monkeypatch, _make_parser(camera=None, model="HERO10 Black"))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface())
    assert info.device_type == "ext_camera"
    assert info.device_model == "HERO10 Black"


def test_detect_ext_camera_make_only(monkeypatch):
    _install_fake_parser(monkeypatch, _make_parser(camera="Insta360", model=None))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface())
    assert info.device_type == "ext_camera"
    assert info.device_model == "Insta360"


# ── Apple iPhone path ─────────────────────────────────────────────────────

def test_detect_apple_iphone_full_tags(monkeypatch):
    def _factory(path):
        raise OSError("Unsupported")
    _install_fake_parser(monkeypatch, _factory)
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface(
        format_tags={
            "com.apple.quicktime.make": "Apple",
            "com.apple.quicktime.model": "iPhone 14 Pro",
        },
    ))
    assert info.device_type == "phone"
    # Make is "Apple"; model already contains "iPhone", which does NOT
    # contain "apple" case-insensitively → combined.
    assert info.device_model == "Apple iPhone 14 Pro"


def test_detect_apple_iphone_model_without_make(monkeypatch):
    _install_fake_parser(monkeypatch, lambda p: (_ for _ in ()).throw(OSError()))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface(
        format_tags={"com.apple.quicktime.model": "iPhone 14 Pro"},
    ))
    assert info.device_type == "phone"
    assert info.device_model == "Apple iPhone 14 Pro"


def test_detect_apple_iphone_no_model_tag(monkeypatch):
    _install_fake_parser(monkeypatch, lambda p: (_ for _ in ()).throw(OSError()))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface(
        format_tags={"com.apple.quicktime.software": "15.1"},
    ))
    assert info.device_type == "phone"
    assert info.device_model == "Unknown"


# ── Android path ──────────────────────────────────────────────────────────

def test_detect_android_full_tags(monkeypatch):
    _install_fake_parser(monkeypatch, lambda p: (_ for _ in ()).throw(OSError()))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface(
        format_tags={
            "com.android.manufacturer": "Samsung",
            "com.android.model": "SM-S911B",
        },
    ))
    assert info.device_type == "phone"
    assert info.device_model == "Samsung SM-S911B"


def test_detect_android_version_only(monkeypatch):
    """Third-party Android recorders often emit only com.android.version."""
    _install_fake_parser(monkeypatch, lambda p: (_ for _ in ()).throw(OSError()))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface(
        format_tags={"com.android.version": "12"},
    ))
    assert info.device_type == "phone"
    assert info.device_model == "Unknown"


# ── Unknown fallback ──────────────────────────────────────────────────────

def test_detect_unknown_when_nothing_matches(monkeypatch):
    _install_fake_parser(monkeypatch, lambda p: (_ for _ in ()).throw(OSError()))
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface(
        format_tags={"major_brand": "mp42", "minor_version": "0"},
    ))
    assert info.device_type == "Unknown"
    assert info.device_model == "Unknown"


def test_detect_unknown_when_telemetry_parser_missing(monkeypatch):
    """No telemetry_parser installed — registry falls to ffprobe tags."""
    monkeypatch.setitem(sys.modules, "telemetry_parser", None)
    info = di.detect_capture_device("/tmp/f.mp4", _tag_surface())
    assert info.device_type == "Unknown"
    assert info.device_model == "Unknown"
