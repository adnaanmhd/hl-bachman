"""Rebuild batch_report.md / batch_results.json / batch_results.csv from
the per-video JSONs inside an `hl-score` run directory.

Useful when a batch was aborted before the final aggregation step at
`batch.py:183` — the per-video artefacts are already on disk but the
batch-level outputs were never written.

Usage:
    python scripts/reconstruct_batch_report.py <run_dir>
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from bachman_cortex.data_types import (
    BatchScoreReport,
    MetadataCheckResult,
    QualityMetricResult,
    QualitySegment,
    TechnicalCheckResult,
    VideoScoreReport,
)
from bachman_cortex.reporting import aggregate_batch_stats, write_batch_report


def _seg_value(raw):
    if raw == "NaN":
        return float("nan")
    return raw


def _load_report(path: Path) -> VideoScoreReport:
    d = json.loads(path.read_text())
    return VideoScoreReport(
        video_path=d["video_path"],
        video_name=d["video_name"],
        generated_at=d["generated_at"],
        processing_wall_time_s=d["processing_wall_time_s"],
        duration_s=d["duration_s"],
        metadata_checks=[MetadataCheckResult(**c) for c in d.get("metadata_checks", [])],
        technical_checks=[TechnicalCheckResult(**c) for c in d.get("technical_checks", [])],
        quality_metrics=[
            QualityMetricResult(
                metric=m["metric"],
                percent_frames=m["percent_frames"],
                segments=[
                    QualitySegment(
                        start_s=s["start_s"],
                        end_s=s["end_s"],
                        duration_s=s["duration_s"],
                        value=_seg_value(s["value"]),
                        value_label=s["value_label"],
                    )
                    for s in m.get("segments", [])
                ],
                skipped=m.get("skipped", False),
            )
            for m in d.get("quality_metrics", [])
        ],
        technical_skipped=d.get("technical_skipped", False),
        quality_skipped=d.get("quality_skipped", False),
    )


def main(run_dir: Path) -> None:
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    jsons = sorted(
        p for p in run_dir.glob("*/*.json")
        if p.stem == p.parent.name
    )
    if not jsons:
        sys.exit(f"no per-video JSONs found under {run_dir}")

    reports = [_load_report(p) for p in jsons]
    meta, tech, qual = aggregate_batch_stats(reports)

    batch = BatchScoreReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        video_count=len(reports),
        total_duration_s=round(sum(r.duration_s for r in reports), 2),
        total_wall_time_s=round(sum(r.processing_wall_time_s for r in reports), 3),
        metadata_check_stats=meta,
        technical_check_stats=tech,
        quality_metric_stats=qual,
        videos=reports,
        errors=[],
    )

    paths = write_batch_report(batch, run_dir)
    print(f"reconstructed from {len(reports)} per-video reports")
    for kind, p in paths.items():
        print(f"  {kind}: {p}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: reconstruct_batch_report.py <run_dir>")
    main(Path(sys.argv[1]))
