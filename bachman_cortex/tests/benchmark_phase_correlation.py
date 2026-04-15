"""Benchmark: phase-correlation motion analysis vs existing LK+SSIM pipeline.

Implements phase-correlation-based camera stability and frozen detection,
then compares timing and results against the existing LK optical-flow +
SSIM approach from motion_analysis.py.

Usage:
    python -m bachman_cortex.tests.benchmark_phase_correlation
"""

import os
import sys
import time

import cv2
import numpy as np

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.checks.motion_analysis import (
    check_camera_stability,
    check_frozen_segments,
)


# ── Phase-correlation scoring (no rotation component) ───────────────────────

def _phase_corr_score(
    translations: list[float],
    *,
    trans_threshold: float = 8.0,
    jump_threshold: float = 30.0,
    variance_threshold: float = 6.0,
    w_trans: float = 0.4375,
    w_var: float = 0.3125,
    w_jump: float = 0.25,
) -> tuple[float, dict]:
    """Convert per-frame translation magnitudes to a shakiness score in [0, 1].

    Same logic as _transforms_to_score but without the rotation component.
    Translations are already in native-res pixel units.
    """
    if not translations:
        return 0.0, {}

    t = np.array(translations, dtype=np.float64)

    avg_t = float(np.mean(t))
    std_t = float(np.std(t))
    jumps = int(np.sum(t > jump_threshold))
    n = len(t)

    t_s = min(avg_t / (trans_threshold * 3), 1.0)
    v_s = min(std_t / (variance_threshold * 2), 1.0)
    j_s = min((jumps / n) * 10, 1.0)

    score = w_trans * t_s + w_var * v_s + w_jump * j_s
    stats = dict(avg_t=round(avg_t, 2), std_t=round(std_t, 2), jumps=jumps)
    return round(score, 3), stats


# ── Phase-correlation stability check ───────────────────────────────────────

def check_camera_stability_phase_corr(
    video_path: str,
    *,
    shaky_score_threshold: float = 0.30,
    target_fps: float = 30.0,
    scale: float = 0.25,
    trans_threshold: float = 8.0,
    jump_threshold: float = 30.0,
    variance_threshold: float = 6.0,
    w_trans: float = 0.4375,
    w_var: float = 0.3125,
    w_jump: float = 0.25,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> CheckResult:
    """Camera stability check using cv2.phaseCorrelate at 0.25x resolution.

    Processes per-second windows and computes a shakiness score from
    the translation vectors (no rotation -- phase correlation only gives
    translation).  Same pass/fail logic as the LK-based check.

    Args:
        video_path: Path to the video file.
        shaky_score_threshold: Score above this => fail.
        target_fps: Analysis is subsampled to this FPS.
        scale: Downscale factor (0.25 = quarter resolution).
        trans_threshold: Translation px/frame threshold (native res).
        jump_threshold: Sudden jolt px/frame threshold.
        variance_threshold: Translation std-dev threshold.
        w_trans: Scoring weight for mean translation.
        w_var: Scoring weight for translation variance.
        w_jump: Scoring weight for sudden jumps.
        start_sec: Optional start of analysis range (seconds).
        end_sec: Optional end of analysis range (seconds).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / native_fps
    frames_per_sec = int(native_fps)

    # Sub-range support
    first_sec = int(start_sec) if start_sec is not None else 0
    last_sec = int(end_sec) if end_sec is not None else int(total_duration)

    # FPS-based subsampling
    fps_skip = max(1, round(native_fps / target_fps))

    # Compute downscaled dimensions once
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ds_w = max(1, int(width * scale))
    ds_h = max(1, int(height * scale))
    native_scale = 1.0 / scale  # multiply dx/dy by this to get native-res

    # Pre-compute Hanning window
    hann = cv2.createHanningWindow((ds_w, ds_h), cv2.CV_64F)

    score_kwargs = dict(
        trans_threshold=trans_threshold,
        jump_threshold=jump_threshold,
        variance_threshold=variance_threshold,
        w_trans=w_trans,
        w_var=w_var,
        w_jump=w_jump,
    )

    per_sec_scores: dict[int, float] = {}
    frames_analysed = 0

    for sec in range(first_sec, last_sec + 1):
        sf = sec * frames_per_sec
        ef = min(sf + frames_per_sec, total_frames)
        if sf >= total_frames:
            break

        # Seek to start of this second
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ret, frame = cap.read()
        if not ret:
            break

        prev_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (ds_w, ds_h),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)

        translations: list[float] = []
        local_idx = 0

        for fno in range(sf + 1, ef):
            ret, frame = cap.read()
            if not ret:
                break
            local_idx += 1

            if local_idx % fps_skip != 0:
                continue

            curr_gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (ds_w, ds_h),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float64)

            (dx, dy), _response = cv2.phaseCorrelate(prev_gray, curr_gray, hann)
            mag = np.sqrt(dx**2 + dy**2) * native_scale
            translations.append(mag)
            frames_analysed += 1

            prev_gray = curr_gray

        score, _ = _phase_corr_score(translations, **score_kwargs)
        per_sec_scores[sec] = score

    cap.release()

    shaky_seconds = [s for s, sc in per_sec_scores.items() if sc >= shaky_score_threshold]
    overall_score = float(np.mean(list(per_sec_scores.values()))) if per_sec_scores else 0.0
    passes = overall_score <= shaky_score_threshold

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(overall_score, 4),
        confidence=1.0,
        details={
            "overall_score": round(overall_score, 4),
            "shaky_score_threshold": shaky_score_threshold,
            "shaky_seconds": shaky_seconds,
            "shaky_seconds_count": len(shaky_seconds),
            "total_seconds": last_sec - first_sec,
            "frames_analysed": frames_analysed,
            "scale": scale,
            "native_fps": round(native_fps, 2),
            "target_fps": target_fps,
            "fps_skip": fps_skip,
            "method": "phase_correlation",
        },
    )


# ── Phase-correlation frozen detection ──────────────────────────────────────

def check_frozen_segments_phase_corr(
    video_path: str,
    *,
    max_consecutive: int = 30,
    frozen_translation_threshold: float = 0.1,
    scale: float = 0.25,
    target_fps: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> CheckResult:
    """Frozen segment detection via phase correlation translation magnitude.

    If the translation magnitude between consecutive frames is below
    `frozen_translation_threshold` (native-res px) for N consecutive
    frames, mark as frozen.  Eliminates the need for SSIM entirely.

    Args:
        video_path: Path to the video file.
        max_consecutive: Max allowed consecutive frozen frames at native FPS.
        frozen_translation_threshold: Below this (native px) = frozen pair.
        scale: Downscale factor for phase correlation.
        target_fps: If set, subsample to this FPS.
        start_sec: Optional start of analysis range (seconds).
        end_sec: Optional end of analysis range (seconds).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / native_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ds_w = max(1, int(width * scale))
    ds_h = max(1, int(height * scale))
    native_scale = 1.0 / scale
    hann = cv2.createHanningWindow((ds_w, ds_h), cv2.CV_64F)

    # Subsampling
    frame_step = max(1, round(native_fps / target_fps))
    effective_fps = native_fps / frame_step
    effective_max = max(1, round(max_consecutive * effective_fps / native_fps))

    # Sub-range support
    start_frame = int((start_sec or 0) * native_fps)
    end_frame = int((end_sec or total_duration) * native_fps)
    end_frame = min(end_frame, total_frames)

    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return CheckResult(
            status="pass", metric_value=0.0, confidence=1.0,
            details={"error": "could not read first frame"},
        )

    prev_gray = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (ds_w, ds_h),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)

    current_run = 0
    longest_run = 0
    frozen_segments: list[dict] = []
    run_start_frame = start_frame
    frames_sampled = 1
    frame_idx = start_frame

    while True:
        frame_idx += frame_step
        if frame_idx >= end_frame:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames_sampled += 1

        curr_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (ds_w, ds_h),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)

        (dx, dy), _response = cv2.phaseCorrelate(prev_gray, curr_gray, hann)
        mag = np.sqrt(dx**2 + dy**2) * native_scale

        is_frozen = mag < frozen_translation_threshold

        if is_frozen:
            if current_run == 0:
                run_start_frame = frame_idx - frame_step
            current_run += 1
        else:
            if current_run > 0:
                if current_run > longest_run:
                    longest_run = current_run
                if current_run > effective_max:
                    frozen_segments.append({
                        "start_frame": run_start_frame,
                        "length_sampled": current_run,
                        "start_sec": round(run_start_frame / native_fps, 2),
                        "duration_sec": round(current_run * frame_step / native_fps, 2),
                    })
                current_run = 0

        prev_gray = curr_gray

    # Handle run at end of video
    if current_run > 0:
        if current_run > longest_run:
            longest_run = current_run
        if current_run > effective_max:
            frozen_segments.append({
                "start_frame": run_start_frame,
                "length_sampled": current_run,
                "start_sec": round(run_start_frame / native_fps, 2),
                "duration_sec": round(current_run * frame_step / native_fps, 2),
            })

    cap.release()

    passes = longest_run <= effective_max
    metric = longest_run / effective_max if effective_max > 0 else 0.0

    longest_run_native_est = longest_run * frame_step
    frozen_duration_s = round(longest_run_native_est / native_fps, 2) if native_fps > 0 else 0

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(min(metric, 10.0), 4),
        confidence=1.0,
        details={
            "longest_frozen_run_sampled": longest_run,
            "longest_frozen_run_native_est": longest_run_native_est,
            "effective_max_consecutive": effective_max,
            "max_consecutive_native": max_consecutive,
            "frozen_translation_threshold": frozen_translation_threshold,
            "frames_sampled": frames_sampled,
            "total_native_frames": total_frames,
            "native_fps": round(native_fps, 2),
            "effective_fps": round(effective_fps, 2),
            "frame_step": frame_step,
            "frozen_duration_s": frozen_duration_s,
            "frozen_segments": frozen_segments[:10],
            "method": "phase_correlation",
        },
    )


# ── Benchmark runner ────────────────────────────────────────────────────────

def _print_result(label: str, result: CheckResult, elapsed: float):
    """Pretty-print a CheckResult with timing."""
    print(f"\n  [{label}]")
    print(f"    Status:       {result.status}")
    print(f"    Metric:       {result.metric_value}")
    print(f"    Time:         {elapsed:.2f}s")
    if result.details:
        for k, v in result.details.items():
            if k == "frozen_segments" and isinstance(v, list) and len(v) > 3:
                print(f"    {k}: [{len(v)} segments, showing first 3]")
                for seg in v[:3]:
                    print(f"      {seg}")
            elif k == "shaky_seconds" and isinstance(v, list) and len(v) > 10:
                print(f"    {k}: [{len(v)} seconds flagged]")
            else:
                print(f"    {k}: {v}")


def main():
    video_path = "/home/egocentric-humynlabs/Documents/hl-bachman/20251223-202254-01a0213.mp4"
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Quick probe
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps if fps > 0 else 0
    cap.release()

    print("=" * 70)
    print("Phase Correlation vs Lucas-Kanade+SSIM Benchmark")
    print("=" * 70)
    print(f"Video:      {video_path}")
    print(f"Resolution: {w}x{h} @ {fps:.1f} FPS")
    print(f"Duration:   {dur:.1f}s ({total} frames)")
    print()

    # ── Existing LK + SSIM checks ──────────────────────────────────────────
    print("-" * 70)
    print("EXISTING METHOD: Lucas-Kanade + SSIM")
    print("-" * 70)

    t0 = time.perf_counter()
    lk_stability = check_camera_stability(video_path)
    t_lk_stab = time.perf_counter() - t0
    _print_result("Camera Stability (LK)", lk_stability, t_lk_stab)

    t0 = time.perf_counter()
    ssim_frozen = check_frozen_segments(video_path)
    t_ssim_frozen = time.perf_counter() - t0
    _print_result("Frozen Segments (SSIM)", ssim_frozen, t_ssim_frozen)

    total_existing = t_lk_stab + t_ssim_frozen

    # ── New phase-correlation checks ────────────────────────────────────────
    print()
    print("-" * 70)
    print("NEW METHOD: Phase Correlation")
    print("-" * 70)

    t0 = time.perf_counter()
    pc_stability = check_camera_stability_phase_corr(video_path)
    t_pc_stab = time.perf_counter() - t0
    _print_result("Camera Stability (PhaseCorr)", pc_stability, t_pc_stab)

    t0 = time.perf_counter()
    pc_frozen = check_frozen_segments_phase_corr(video_path)
    t_pc_frozen = time.perf_counter() - t0
    _print_result("Frozen Segments (PhaseCorr)", pc_frozen, t_pc_frozen)

    total_new = t_pc_stab + t_pc_frozen

    # ── Comparison summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Check':<30} {'LK+SSIM':>10} {'PhaseCorr':>10} {'Speedup':>10}")
    print("-" * 62)
    speedup_stab = t_lk_stab / t_pc_stab if t_pc_stab > 0 else float("inf")
    speedup_frozen = t_ssim_frozen / t_pc_frozen if t_pc_frozen > 0 else float("inf")
    speedup_total = total_existing / total_new if total_new > 0 else float("inf")

    print(f"{'Camera Stability':<30} {t_lk_stab:>9.2f}s {t_pc_stab:>9.2f}s {speedup_stab:>9.1f}x")
    print(f"{'Frozen Detection':<30} {t_ssim_frozen:>9.2f}s {t_pc_frozen:>9.2f}s {speedup_frozen:>9.1f}x")
    print(f"{'TOTAL':<30} {total_existing:>9.2f}s {total_new:>9.2f}s {speedup_total:>9.1f}x")

    print(f"\n{'Result Agreement':}")
    print(f"  Stability: LK={lk_stability.status} (score={lk_stability.metric_value}) "
          f"vs PC={pc_stability.status} (score={pc_stability.metric_value})")
    print(f"  Frozen:    SSIM={ssim_frozen.status} (metric={ssim_frozen.metric_value}) "
          f"vs PC={pc_frozen.status} (metric={pc_frozen.metric_value})")

    lk_shaky = set(lk_stability.details.get("shaky_seconds", []))
    pc_shaky = set(pc_stability.details.get("shaky_seconds", []))
    if lk_shaky or pc_shaky:
        overlap = lk_shaky & pc_shaky
        print(f"\n  Shaky seconds overlap: {len(overlap)} common out of "
              f"LK={len(lk_shaky)}, PC={len(pc_shaky)}")

    lk_frozen_segs = ssim_frozen.details.get("frozen_segments", []) if ssim_frozen.details else []
    pc_frozen_segs = pc_frozen.details.get("frozen_segments", []) if pc_frozen.details else []
    print(f"  Frozen segments: SSIM found {len(lk_frozen_segs)}, "
          f"PC found {len(pc_frozen_segs)}")

    print()


if __name__ == "__main__":
    main()
