# Scoring Engine — Check Specifications

The scoring engine evaluates each video through three gated stages:

1. **Metadata** — cheap ffprobe reads; any single fail skips stages 2 and 3.
2. **Technical** — quality of the capture itself (luminance, stability, frozen, pixelation).
3. **Quality metrics** — readings about what's in frame (hands, HOI, angle, participants, obstruction). Not pass/fail at the video level; reported as `percent_frames` + merged segments.

A single decode pass drives all of stages 2 and 3. If the technical stage fails, quality metrics are rendered as `SKIPPED` in the MD/JSON report — but the raw per-frame quality columns remain in the parquet for debugging.

See `SCORING_ENGINE_PLAN.md` for the locked design decisions.

---

## Stage 1: Metadata checks

Any single failure marks the video metadata-failed; technical and quality are reported SKIPPED, and no decode is performed.

| Check       | Rule                                                                                  | Source  |
| ----------- | ------------------------------------------------------------------------------------- | ------- |
| format      | Container is MP4 (ffprobe: `mov,mp4,m4a,3gp,3g2,mj2`).                                | ffprobe |
| encoding    | Video codec is H.264 or HEVC (H.265).                                                 | ffprobe |
| resolution  | Displayed dims ≥ 1920×1080 (after rotation).                                          | ffprobe |
| frame_rate  | ≥ 28 FPS.                                                                             | ffprobe |
| duration    | ≥ 59 s.                                                                               | ffprobe |
| orientation | Rotation ∈ {0, 90, 270} AND displayed landscape (displayed_width > displayed_height). | ffprobe |

Thresholds are configurable in the TOML config under `[metadata]`.

---

## Stage 2: Technical checks

All four checks run on every video that passes the metadata gate. Any one failure marks technical-failed.

| Check      | Rule                                                                                                   | Cadence / resolution         |
| ---------- | ------------------------------------------------------------------------------------------------------ | ---------------------------- |
| luminance  | ≥ 80% good frames (neither dead black <15, too dark 15–45, blown out >230, nor flicker window).        | 10 FPS, 360p                 |
| stability  | Whole-video mean jitter score ≤ 0.181 (high-pass-filtered LK optical flow).                            | 30 FPS cap, 360p (0.5×)      |
| frozen     | No run > 60 consecutive native-frame-equivalents with near-zero motion (trans < 0.1 px, rot < 0.001°). | Derived from motion samples. |
| pixelation | ≥ 80% frames with blockiness ratio ≤ 1.5.                                                              | 10 FPS, 720p                 |

Thresholds are configurable under `[technical.luminance]`, `[technical.stability]`, `[technical.frozen]`, `[technical.pixelation]`.

---

## Stage 3: Quality metrics

Six per-frame metrics, each reported as `percent_frames` plus merged segments. `face_presence` as a standalone metric is removed — SCRFD face detection now feeds the participants signal.

All quality checks run at 1 FPS on 720p frames.

| Metric                 | Per-frame PASS                                                                                                   | Value (raw per-frame)                       | Value on fail                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------ |
| both_hands_visibility  | Hands23 detects L **and** R, each conf ≥ 0.7.                                                                    | max(L_conf, R_conf)                         | 0.0                                                    |
| single_hand_visibility | ≥1 hand with Hands23 conf ≥ 0.7.                                                                                 | max Hands23 conf                            | 0.0                                                    |
| hand_obj_interaction   | ≥1 detected hand with contact state ∈ {P, F}.                                                                    | contact char of qualifying hand (P/F)       | most-confident hand's char (N/S/O); null if zero hands |
| hand_angle             | all detected hands' angles ≤ 40° (zero hands → fail).                                                            | mean angle of detected hands (degrees)      | mean angle; NaN if no hands                            |
| participants           | ≥1 "other person" signal from {YOLO (≥0.6, wearer-filtered), SCRFD (≥0.6, wearer-filtered), extra Hands23 hand}. | max(YOLO_conf, SCRFD_conf, extra_hand_conf) | 0.0                                                    |
| obstructed             | Heuristic triggers ≥2 of 4 signals on the central 80% crop.                                                      | `true`                                      | `false`                                                |

Both_hands and single_hand may pass for the same frame (not mutually exclusive).

### Wearer filter (applies to YOLO persons AND SCRFD faces)

- Exclude detections that overlap a Hands23 hand (wearer's own body/face via reflection).
- Exclude detections anchored in the bottom-centre region of the frame.
- Exclude bboxes whose height is < 15% of the frame height.

### Segment merging

Runs shorter than `merge_threshold_s` (default 1.0s) absorb into their preceding neighbour (flipping state). First-run edge case: absorb into the following run. Single left-to-right pass — no cascade re-merging. Parquet is un-merged; merging applies only to JSON / MD report output. `percent_frames` is computed from the un-merged per-frame array.

**Segment value semantics (plan §1):**

- Conf-based metrics: max confidence across _qualifying_ (originally pass) frames in the merged window. Absorbed fail-frames never contribute.
- Hand angle: mean angle over all frames in the merged segment.
- Hand-obj contact char: most common contact char over all frames in the merged segment.
- Obstructed: `true` / `false` per the merged segment's state.

---

## Output artefacts

Per video:

- `report.md` — human-readable Markdown tables.
- `{video_name}.json` — same content as MD but structured, JSON-safe (NaN serialised as the string `"NaN"`).
- `{video_name}.parquet` — one row per native frame, dense schema (see `SCORING_ENGINE_PLAN.md` §4). Omitted when metadata failed (nothing decoded).

Per batch:

- `batch_report.md`, `batch_results.json`, `batch_results.csv` in the run-dir root.

Output layout: `results/run_NNN/{video_name}/report.md`, etc. `NNN` is zero-padded to 3 digits and auto-extends past 999.
