# Bachman

## What This Is

A local web app that validates and quality-checks egocentric videos of humans performing tasks in commercial, residential, or agricultural settings. The user points it at a directory of videos, picks a category and task, and the app runs a three-phase pipeline (metadata gate → review-aware quality checks → yield report) with an in-browser human review step for ambiguous segments. It wraps the existing `bachman_cortex` pipeline rather than replacing it.

## Core Value

Turn a raw batch of egocentric videos into reviewed, trimmed, yield-scored segment clips — with a human kept in the loop for ambiguous check results — running entirely on a local RTX GPU box.

## Requirements

### Validated

<!-- Inferred from existing bachman_cortex codebase (verified against .planning/codebase/ARCHITECTURE.md, STACK.md). These are the capabilities the new app wraps. -->

- ✓ 4-phase CLI validation pipeline (metadata → segment filtering → segment validation → yield calc) — existing
- ✓ Batch video processing with per-worker model isolation — existing
- ✓ 4-video parallel throughput on NVIDIA RTX via multiprocessing — existing
- ✓ ML-backed checks: face presence (SCRFD), participants (YOLO), hand visibility/interaction (Hands23), POV hand angle — existing
- ✓ Deterministic checks: luminance/blur, motion camera stability, motion frozen segments, view obstruction — existing
- ✓ FFprobe-based metadata extraction (codec, resolution, FPS, duration, rotation) — existing
- ✓ Lossless NVENC / libx264 retranscoding path with stream-copied audio — existing
- ✓ Per-video JSON reports and batch summaries — existing

### Active

<!-- Hypotheses for the new web-app layer. Each is a v1 "must have" (see Key Decisions). -->

- [ ] Local web app (FastAPI-style backend + browser frontend) starts via a shell script
- [ ] Directory-picker flow: user points at a video directory, picks category + task from searchable dropdowns sourced from a static YAML/JSON config
- [ ] Phase 1 metadata gate runs with the brief's acceptance conditions (MP4, H.264/HEVC, ≥1920x1080 displayed, ≥28 FPS, ≥60 s, landscape)
- [ ] Phase 2 runs the existing checks but with a three-way status (pass / review / fail) per the brief's status mapping, producing fail, review, and pass segments with the brief's duration thresholds (2 s for face/participants, 3 s otherwise) and merging overlaps
- [ ] Review-segment retention logic: keep a review segment only if it is ≥60 s, or adjacent to a passing segment, or adjacent to non-failed segments that together with it sum ≥60 s; otherwise label `segment_too_short` and mark UNUSABLE
- [ ] In-browser human review UI: for each retained review segment, play the trimmed clip, show which check(s) flagged it, accept pass/fail verdict
- [ ] Post-review segment reconciliation: merge pass-review segments with adjacent passing/short-but-valid segments for Phase 3; merge fail-review segments with adjacent fail segments and label REJECTED
- [ ] Phase 3 segment trimming: ffmpeg-cut segment clips into `passing_videos/`, `rejected_videos/`, `unusable_videos/` (per-segment, not per-source-video) using the project's existing lossless NVENC/libx264 conventions
- [ ] Phase 3 per-video report (JSON + CSV): category, task, video name, durations, yield %, full segment list with status/reason, rejected-segment frame-level breakdown per check
- [ ] Phase 3 batch report: category, task, yield, totals, top rejection reasons with counts, yield stats (min/max/mean/median)
- [ ] SQLite-backed batch state: jobs, per-video progress, and review decisions persist; closing the browser does not kill the batch; reopening resumes at the current state
- [ ] Preserve 4-video parallel throughput on the existing RTX GPU setup

### Out of Scope

- Cloud deployment — user explicitly wants local-only for now
- Multi-user auth / accounts — single-user local tool
- OPTIMIZATION_PLAN_V3.md and YOLOV5N_FINETUNE_PLAN.md work — superseded, do not pull in
- Rewriting `bachman_cortex` pipeline internals from scratch — wrap as-is; only add the review-aware segment layer on top
- Replacing the existing CLI (`validate.sh`, `run_batch.py`) — left working in parallel for CLI users
- Editing the category/task taxonomy from the UI — static config file is the source of truth
- Live re-processing of already-reviewed segments — review decisions are final per batch run
- Mobile or tablet frontend — desktop browser only

## Context

**Codebase state (brownfield):** The repo contains a working CLI pipeline at `bachman_cortex/` with four phases. ARCHITECTURE.md, STACK.md, and the other codebase docs under `.planning/codebase/` are current as of 2026-04-15 and should be the source of truth for existing behavior.

**Hardware target:** Local NVIDIA RTX GPU workstation. Existing pipeline uses CUDA 12.1, PyTorch GPU wheels, onnxruntime-gpu, and `cv2.cudacodec`; 4-video parallel batch achieved via multiprocessing with one model set per worker. Video Codec SDK 12.2.72 is already present in-tree for NVENC/NVDEC.

**Pipeline semantics are changing:** The existing pipeline is binary pass/fail at the check level. The new app introduces a three-way pass/review/fail per the brief's confidence thresholds (e.g. face presence: pass <0.5, review 0.5–0.8, fail ≥0.8). The segment adjacency and ≥60 s retention rules are also new. This is additive on top of the existing check implementations — the checks themselves don't change.

**Existing plans in repo root:** `OPTIMIZATION_PLAN.md`, `OPTIMIZATION_PLAN_V2.md`, `OPTIMIZATION_PLAN_V3.md`, and `YOLOV5N_FINETUNE_PLAN.md` exist but are explicitly superseded by this project.

**User-noted preferences already in memory:**
- ffmpeg retranscodes default to NVENC `-preset p7 -tune lossless` (fallback `libx264 -crf 0`), audio stream-copied — apply to segment trimming
- Prefer removing dead code over preserving it for "consistency" when refactoring wrappers around the existing pipeline

## Constraints

- **Tech stack (backend):** Python 3.11+, reuse the existing `bachman_cortex` venv and its dependencies — Why: single-environment install, shared model loading, avoids double-maintaining ML deps
- **Tech stack (frontend):** Browser-based, runs on `localhost` — Why: user chose local web app form factor; HTML5 `<video>` is the simplest path to in-browser clip playback for review
- **Deployment:** Local only, no cloud — Why: user explicitly out-of-scope for cloud; simplifies auth, storage, and GPU access
- **Performance:** Maintain existing 4-video parallel throughput on RTX — Why: user hard requirement; the web layer must not serialize inference
- **Persistence:** SQLite (single-file, embedded) — Why: local-only app; resume-on-reopen requires durable state without a DB server
- **Launch:** Shell script (extends existing `validate.sh` pattern) — Why: user chose this; matches existing onboarding (venv activation, model weight download, GPU detection)
- **Pipeline integration:** Wrap `bachman_cortex` as a library; do not refactor internals — Why: user chose "wrap as-is"; preserves existing CLI path and reduces blast radius of the new app
- **Segment trimming:** ffmpeg with NVENC lossless preset (libx264 -crf 0 fallback), stream-copied audio — Why: user preference already in memory; keeps clip quality identical to source
- **Category/task taxonomy:** Static YAML/JSON config file — Why: user chose this; no admin UI; user-editable by hand

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Local web app (FastAPI-style backend + browser UI) over desktop/CLI | Need in-browser clip playback for the review step; local web avoids desktop packaging overhead | — Pending |
| Wrap `bachman_cortex` as a library, don't refactor internals | Existing pipeline is working; review-aware layer is additive; reduces regression risk | — Pending |
| Three-way pass/review/fail status with per-check confidence bands (per brief) | Binary pass/fail misses ambiguous cases that are worth a human look; review-retention rules prevent UI spam from short review segments | — Pending |
| Output is trimmed segment clips, not moved source files | A single source video contains mixed-quality segments; segment-level output is the honest unit | — Pending |
| SQLite-backed job + review-decision persistence | Batches are long; closing a browser tab should not waste a GPU run | — Pending |
| Categories/tasks from static YAML/JSON config, no admin UI | Single-user local tool; config-as-code is cheaper than building/maintaining an editor | — Pending |
| Shell-script launcher (extends `validate.sh` pattern) | Matches existing onboarding; avoids docker-compose + GPU-in-container complexity | — Pending |
| `OPTIMIZATION_PLAN*.md` and `YOLOV5N_FINETUNE_PLAN.md` are explicitly superseded | User confirmed; prevents downstream phases from dragging in stale scope | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-15 after initialization*
