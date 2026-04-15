#!/usr/bin/env python
"""Run the validation & processing pipeline on a batch of videos.

Usage:
    # Single video
    python -m bachman_cortex.run_batch /path/to/video.mp4

    # Multiple videos
    python -m bachman_cortex.run_batch /path/to/video1.mp4 /path/to/video2.mp4

    # Directory of videos
    python -m bachman_cortex.run_batch /path/to/videos/

    # With options
    python -m bachman_cortex.run_batch /path/to/videos/ --fps 2 --min-segment 90 --output results/
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bachman_cortex.pipeline import ValidationProcessingPipeline, PipelineConfig
from bachman_cortex.reporting import write_video_report, write_batch_report
from bachman_cortex.data_types import VideoProcessingResult, CheckableSegment


def collect_videos(paths: list[str]) -> list[Path]:
    """Collect .mp4 video files from paths (files or directories)."""
    videos = []
    invalid_paths = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == ".mp4":
            videos.append(path)
        elif path.is_dir():
            videos.extend(sorted(f for f in path.glob("*.mp4") if not f.name.startswith("._")))
            videos.extend(sorted(f for f in path.glob("*.MP4") if not f.name.startswith("._")))
        else:
            invalid_paths.append(p)
    if invalid_paths:
        print(f"WARNING: {len(invalid_paths)} path(s) are not valid files or directories:")
        for p in invalid_paths:
            print(f"  - {p}")
        if len(invalid_paths) > 1:
            joined = " ".join(invalid_paths)
            print(f"  Hint: if this was a single path with spaces, quote it:")
            print(f'    hl-validate "{joined}"')
    # Deduplicate preserving order
    seen = set()
    unique = []
    for v in videos:
        resolved = v.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(v)
    return unique


def _auto_detect_workers() -> int:
    """Determine worker count based on available resources."""
    cpu_count = os.cpu_count() or 1
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Each worker loads all models (~6-8 GB) into its own CUDA context
            return max(1, min(int(vram_gb // 8), cpu_count // 2, 4))
    except ImportError:
        pass
    return max(1, min(cpu_count // 4, 4))


def _result_to_dict(result: VideoProcessingResult) -> dict:
    """Convert VideoProcessingResult to a JSON-serializable dict."""
    return dataclasses.asdict(result)


# Per-worker pipeline instance, initialized once via _init_worker
_worker_pipeline: ValidationProcessingPipeline | None = None


def _init_worker(config_dict: dict) -> None:
    """Pool initializer: load models once per worker process."""
    global _worker_pipeline
    if "stability_lk_win_size" in config_dict and isinstance(config_dict["stability_lk_win_size"], list):
        config_dict["stability_lk_win_size"] = tuple(config_dict["stability_lk_win_size"])
    config = PipelineConfig(**config_dict)
    _worker_pipeline = ValidationProcessingPipeline(config)
    _worker_pipeline.load_models()


def _process_video_worker(args_tuple: tuple) -> dict:
    """Process a single video in a worker process."""
    video_path_str, output_dir_str = args_tuple
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    video_name = video_path.stem
    video_output_dir = output_dir / video_name

    try:
        result = _worker_pipeline.process_video(str(video_path), video_output_dir)
        write_video_report(result, video_output_dir, _worker_pipeline.config)

        # Save per-video JSON
        result_dict = _result_to_dict(result)
        video_json = video_output_dir / f"{video_name}.json"
        with open(video_json, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        return result_dict

    except Exception as e:
        traceback.print_exc()
        error_result = {
            "video_path": str(video_path),
            "video_name": video_name,
            "error": str(e),
            "processing_time_sec": 0.0,
            "original_duration_sec": 0.0,
            "usable_duration_sec": 0.0,
            "unusable_duration_sec": 0.0,
            "yield_ratio": 0.0,
        }
        return error_result


def main():
    parser = argparse.ArgumentParser(
        description="Run validation & processing pipeline on a batch of videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bachman_cortex.run_batch video.mp4
  python -m bachman_cortex.run_batch videos_dir/
  python -m bachman_cortex.run_batch *.mp4 --output results/
  python -m bachman_cortex.run_batch videos/ --fps 2 --min-segment 90
        """,
    )
    parser.add_argument("paths", nargs="+", help="Video files or directories to process (.mp4 only)")
    parser.add_argument("--output", "-o", default="bachman_cortex/results", help="Output directory for reports (default: bachman_cortex/results)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate in FPS (default: 1.0)")
    parser.add_argument("--min-segment", type=float, default=60.0, help="Minimum checkable segment duration in seconds (default: 60)")
    parser.add_argument("--min-bad-segment", type=float, default=2.0, help="Bad segments with duration > this many seconds are kept; shorter or equal ones are forgiven (default: 2)")
    parser.add_argument("--hevc-to-h264", action="store_true", help="Before Phase 0, losslessly transcode any HEVC input to H.264 (NVENC if available, else libx264 -crf 0). Strips rotation tag. Disabled by default.")
    parser.add_argument("--hand-detector-repo", default="bachman_cortex/models/weights/hands23_detector", help="Path to Hands23 repo")
    parser.add_argument("--scrfd-root", default="bachman_cortex/models/weights/insightface", help="Path to InsightFace models")
    parser.add_argument("--workers", type=int, default=0, help="Parallel video workers (0=auto-detect, 1=sequential)")
    parser.add_argument("--yolo-model", default="yolo11s.pt", help="YOLO model for object detection (default: yolo11s.pt)")

    args = parser.parse_args()

    # Collect videos
    videos = collect_videos(args.paths)
    if not videos:
        print("No .mp4 video files found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to process.")
    for v in videos:
        print(f"  - {v.name} ({os.path.getsize(v) / 1024 / 1024:.1f} MB)")

    # Setup output directory with sequential run numbering
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = [
        d for d in base_output_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    existing_numbers = []
    for d in existing_runs:
        try:
            existing_numbers.append(int(d.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    next_run = max(existing_numbers, default=0) + 1
    output_dir = base_output_dir / f"run_{next_run:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline
    config = PipelineConfig(
        sampling_fps=args.fps,
        min_checkable_segment_sec=args.min_segment,
        min_bad_segment_sec=args.min_bad_segment,
        transcode_hevc=args.hevc_to_h264,
        scrfd_root=args.scrfd_root,
        hand_detector_repo=args.hand_detector_repo,
        yolo_model=args.yolo_model,
    )

    # Determine worker count
    workers = args.workers if args.workers > 0 else _auto_detect_workers()
    print(f"Using {workers} worker(s)")

    # Process videos
    all_results: list[VideoProcessingResult] = []
    all_results_dicts: list[dict] = []
    batch_start = time.perf_counter()

    if workers <= 1:
        # Sequential: single pipeline instance, models loaded once
        pipeline = ValidationProcessingPipeline(config)

        for i, video_path in enumerate(videos):
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(videos)}] {video_path.name}")
            print(f"{'='*70}")

            video_name = video_path.stem
            video_output_dir = output_dir / video_name

            try:
                result = pipeline.process_video(str(video_path), video_output_dir)
                write_video_report(result, video_output_dir, config)
                all_results.append(result)
                result_dict = _result_to_dict(result)
                all_results_dicts.append(result_dict)

                # Save per-video JSON
                video_json = video_output_dir / f"{video_name}.json"
                with open(video_json, "w") as f:
                    json.dump(result_dict, f, indent=2, default=str)

            except Exception as e:
                traceback.print_exc()
                # Create a minimal error result
                error_result = VideoProcessingResult(
                    video_path=str(video_path),
                    video_name=video_name,
                    original_duration_sec=0.0,
                    metadata={},
                    metadata_passed=False,
                    metadata_results={},
                    phase1_check_frame_results=[],
                    phase1_bad_segments=[],
                    phase1_discarded_segments=[],
                    prefiltered_segments=[],
                    segment_results=[],
                    usable_segments=[],
                    rejected_segments=[],
                    usable_duration_sec=0.0,
                    unusable_duration_sec=0.0,
                    yield_ratio=0.0,
                    error=str(e),
                )
                all_results.append(error_result)
                all_results_dicts.append(_result_to_dict(error_result))
    else:
        # Parallel: models loaded once per worker via _init_worker
        config_dict = dataclasses.asdict(config)
        work_items = [
            (str(v), str(output_dir)) for v in videos
        ]

        print(f"Processing {len(videos)} videos with {workers} parallel workers...")
        # Use spawn so each worker initializes its own CUDA context. Default fork
        # on Linux copies the parent's CUDA state (initialized above by
        # _auto_detect_workers), which makes torch.cuda unusable in children and
        # silently falls back to CPU.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(config_dict,)) as pool:
            for i, result_dict in enumerate(pool.imap_unordered(_process_video_worker, work_items)):
                all_results_dicts.append(result_dict)
                name = result_dict.get("video_name", "?")
                status = "error" if result_dict.get("error") else "done"
                yield_r = result_dict.get("yield_ratio", 0)
                t = result_dict.get("processing_time_sec", 0)
                print(f"  [{i+1}/{len(videos)}] {name} ({status}, "
                      f"yield={yield_r:.1%}, {t:.1f}s)")

        # For batch report in multi-worker mode, reconstruct results
        for rd in all_results_dicts:
            prefiltered = [
                CheckableSegment(**c) for c in rd.get("prefiltered_segments", [])
            ]
            usable = [
                CheckableSegment(**c) for c in rd.get("usable_segments", [])
            ]
            r = VideoProcessingResult(
                video_path=rd.get("video_path", ""),
                video_name=rd.get("video_name", ""),
                original_duration_sec=rd.get("original_duration_sec", 0.0),
                metadata=rd.get("metadata", {}),
                metadata_passed=rd.get("metadata_passed", False),
                metadata_results={},
                phase1_check_frame_results=[],
                phase1_bad_segments=[],
                phase1_discarded_segments=[],
                prefiltered_segments=prefiltered,
                segment_results=[],
                usable_segments=usable,
                rejected_segments=[],
                usable_duration_sec=rd.get("usable_duration_sec", 0.0),
                unusable_duration_sec=rd.get("unusable_duration_sec", 0.0),
                yield_ratio=rd.get("yield_ratio", 0.0),
                processing_time_sec=rd.get("processing_time_sec", 0.0),
                error=rd.get("error"),
                transcode_info=rd.get("transcode_info"),
            )
            all_results.append(r)

    wall_clock_sec = time.perf_counter() - batch_start

    # Write batch report
    batch_report_path = write_batch_report(
        all_results, output_dir, config, wall_clock_sec=wall_clock_sec
    )

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(all_results)} videos processed in {wall_clock_sec:.1f}s")
    print(f"Run directory: {output_dir}/")
    print(f"Batch report: {batch_report_path}")
    print(f"{'='*70}")

    # Save full batch JSON
    batch_json = output_dir / "batch_results.json"
    with open(batch_json, "w") as f:
        json.dump(all_results_dicts, f, indent=2, default=str)

    # Update index
    _update_index(base_output_dir, output_dir, all_results)


def _update_index(
    base_output_dir: Path,
    run_dir: Path,
    all_results: list[VideoProcessingResult],
):
    """Append this run to the index.md in the base output directory."""
    index_path = base_output_dir / "index.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = run_dir.name
    num_videos = len(all_results)

    total_yield = 0.0
    total_original = sum(r.original_duration_sec for r in all_results)
    total_usable = sum(r.usable_duration_sec for r in all_results)
    if total_original > 0:
        total_yield = total_usable / total_original

    total_errors = sum(1 for r in all_results if r.error)

    status_summary = f"yield={total_yield:.1%}, {total_usable:.0f}s usable"
    if total_errors:
        status_summary += f", {total_errors} errors"

    entry_line = (
        f"| [{run_name}]({run_name}/batch_report.md) | {timestamp} | "
        f"{num_videos} | {status_summary} |"
    )

    if index_path.exists():
        content = index_path.read_text()
        content = content.rstrip("\n") + "\n" + entry_line + "\n"
    else:
        content = (
            "# Pipeline Run Index\n"
            "\n"
            "| Run | Timestamp | Videos | Result |\n"
            "|-----|-----------|--------|--------|\n"
            f"{entry_line}\n"
        )

    index_path.write_text(content)


if __name__ == "__main__":
    main()
