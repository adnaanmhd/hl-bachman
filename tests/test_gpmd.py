"""Tests for utils.gpmd — presence detector + Highlights scanner.

The scanner consumes telemetry-parser output; tests monkey-patch the
library with a fake module so no real video files are required.
"""

from __future__ import annotations

import sys
import types

import pytest

from bachman_cortex.utils import gpmd


# ── detect_gpmd_stream (ffprobe-only presence check) ───────────────────────

def test_detect_gpmd_stream_matches_handler_name():
    info = gpmd.detect_gpmd_stream([
        {"codec_type": "video"},
        {"tags": {"handler_name": "GoPro MET"}},
    ])
    assert info.present
    assert info.stream_index == 1
    assert "gopro met" in info.handler_name


def test_detect_gpmd_stream_matches_codec_tag():
    info = gpmd.detect_gpmd_stream([
        {"codec_type": "video"},
        {"codec_tag_string": "gpmd"},
    ])
    assert info.present
    assert info.stream_index == 1


def test_detect_gpmd_stream_absent():
    info = gpmd.detect_gpmd_stream([{"codec_type": "video"}])
    assert not info.present
    assert info.stream_index is None


# ── Fake telemetry-parser harness ──────────────────────────────────────────

def _install_fake_parser(monkeypatch, parser_factory):
    """Register a `telemetry_parser` stub with a custom Parser factory.

    `parser_factory` is a callable invoked with the video path and must
    return an object exposing `.camera`, `.model`, and `.telemetry()`.
    To simulate an unsupported file, make it raise.
    """
    fake = types.ModuleType("telemetry_parser")
    fake.Parser = parser_factory
    monkeypatch.setitem(sys.modules, "telemetry_parser", fake)


def _make_parser(camera=None, model=None, telemetry_payload=None):
    class _P:
        def __init__(self, path):
            self.path = path

        @property
        def camera(self):
            return camera

        @property
        def model(self):
            return model

        def telemetry(self):
            return telemetry_payload
    return _P


# ── parse_gpmd_highlights ──────────────────────────────────────────────────

def test_parse_gpmd_highlights_happy_path(monkeypatch):
    settings = {
        "Name": "HERO10 Black",
        "0x56464f56": "W",            # VFOV → Wide
        "0x5a464f56": 133.35,         # ZFOV → 133° horizontal
        "0x48534754": "HIGH",         # HSGT → HyperSmooth HIGH
    }
    _install_fake_parser(
        monkeypatch,
        _make_parser(
            camera="GoPro",
            model="HERO10 Black",
            telemetry_payload=[{"Default": settings}],
        ),
    )
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert h.present
    assert h.camera_model == "HERO10 Black"
    assert h.lens_label == "Wide"
    assert h.fov_deg == 133.35
    assert h.hypersmooth_state == "HIGH"


def test_parse_gpmd_highlights_unsupported_file_returns_absent(monkeypatch):
    def _factory(path):
        raise OSError("Unsupported file format")
    _install_fake_parser(monkeypatch, _factory)
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert not h.present
    assert h.camera_model is None
    assert h.lens_label is None


def test_parse_gpmd_highlights_empty_payload_returns_absent(monkeypatch):
    _install_fake_parser(monkeypatch, _make_parser(telemetry_payload=[]))
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert not h.present


def test_parse_gpmd_highlights_falls_back_to_block_one(monkeypatch):
    """Firmware that stashes settings in block index 1 instead of 0."""
    _install_fake_parser(
        monkeypatch,
        _make_parser(
            camera="GoPro",
            model="HERO8 Black",
            telemetry_payload=[
                {"Default": {}},
                {"Default": {"Name": "HERO8 Black", "0x56464f56": "L"}},
            ],
        ),
    )
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert h.present
    assert h.camera_model == "HERO8 Black"
    assert h.lens_label == "Linear"


def test_parse_gpmd_highlights_eise_fallback(monkeypatch):
    """HSGT / EISA missing — EISE boolean still resolves the state."""
    settings = {"Name": "HERO7 Black", "0x45495345": "Y"}
    _install_fake_parser(
        monkeypatch,
        _make_parser(model="HERO7 Black", telemetry_payload=[{"Default": settings}]),
    )
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert h.hypersmooth_state == "Y"


def test_parse_gpmd_highlights_eisa_skips_na(monkeypatch):
    """EISA=N/A should not be treated as a resolved state."""
    settings = {
        "Name": "HERO10 Black",
        "0x45495341": "N/A",
        "0x45495345": "N",
    }
    _install_fake_parser(
        monkeypatch,
        _make_parser(telemetry_payload=[{"Default": settings}]),
    )
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert h.hypersmooth_state == "N"


def test_parse_gpmd_highlights_rejects_nonpositive_fov(monkeypatch):
    settings = {"0x56464f56": "W", "0x5a464f56": 0}
    _install_fake_parser(
        monkeypatch,
        _make_parser(telemetry_payload=[{"Default": settings}]),
    )
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert h.present
    assert h.fov_deg is None


def test_parse_gpmd_highlights_missing_library_returns_absent(monkeypatch):
    """No telemetry_parser installed — caller falls back gracefully."""
    monkeypatch.setitem(sys.modules, "telemetry_parser", None)
    h = gpmd.parse_gpmd_highlights("/tmp/dummy.mp4")
    assert not h.present
