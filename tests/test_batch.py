"""Tests for batch input discovery + driver helpers.

End-to-end scoring is exercised in the step-13 validation runs (takes
real GPU), so here we stick to the filesystem walking + error
classification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bachman_cortex.batch import (
    _classify_error,
    auto_worker_count,
    iter_input_videos,
)


def _touch_mp4(p: Path) -> Path:
    p.write_bytes(b"")
    return p


def test_iter_input_videos_single_file(tmp_path):
    v = _touch_mp4(tmp_path / "a.mp4")
    assert list(iter_input_videos(v)) == [v]


def test_iter_input_videos_ignores_non_mp4(tmp_path):
    _touch_mp4(tmp_path / "a.mp4")
    (tmp_path / "b.mov").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    result = [p.name for p in iter_input_videos(tmp_path)]
    assert result == ["a.mp4"]


def test_iter_input_videos_is_case_insensitive(tmp_path):
    _touch_mp4(tmp_path / "A.MP4")
    _touch_mp4(tmp_path / "b.mp4")
    names = sorted(p.name for p in iter_input_videos(tmp_path))
    assert names == ["A.MP4", "b.mp4"]


def test_iter_input_videos_skips_hidden(tmp_path):
    _touch_mp4(tmp_path / ".hidden.mp4")
    _touch_mp4(tmp_path / "visible.mp4")
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    _touch_mp4(hidden_dir / "inside.mp4")
    names = [p.name for p in iter_input_videos(tmp_path)]
    assert names == ["visible.mp4"]


def test_iter_input_videos_recurses(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch_mp4(tmp_path / "top.mp4")
    _touch_mp4(sub / "nested.mp4")
    names = sorted(p.name for p in iter_input_videos(tmp_path))
    assert names == ["nested.mp4", "top.mp4"]


def test_iter_input_videos_handles_symlink_cycle(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch_mp4(sub / "x.mp4")
    # Create a cycle: sub/loop -> tmp_path
    (sub / "loop").symlink_to(tmp_path, target_is_directory=True)
    # Iteration must terminate and not revisit.
    result = list(iter_input_videos(tmp_path))
    names = [p.name for p in result]
    # x.mp4 discovered exactly once.
    assert names.count("x.mp4") == 1


def test_iter_input_videos_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_input_videos(tmp_path / "nope"))


def test_classify_error_maps_common_messages():
    assert _classify_error(ValueError("No video stream found")) == "audio_only"
    assert _classify_error(ValueError("Cannot open video")) == "decode_failed"
    assert _classify_error(RuntimeError("ffprobe failed on x.mp4")) \
        == "metadata_probe_failed"
    assert _classify_error(RuntimeError("corrupt header")) == "corrupt"
    # Fallback: exception class name
    assert _classify_error(KeyError("x")) == "KeyError"


def test_auto_worker_count_returns_positive_int():
    assert auto_worker_count() >= 1
