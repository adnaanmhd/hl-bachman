# Video Validation Checks

## Video Metadata

| Check       | Acceptance Condition                    | How It Is Estimated              |
| ----------- | --------------------------------------- | -------------------------------- |
| Format      | MP4 container (MPEG-4)                  | Container metadata via FFprobe   |
| Encoding    | H.264 video codec                       | Codec metadata via FFprobe       |
| Resolution  | >= 1920 x 1080 pixels                   | Width & height metadata          |
| Frame Rate  | >= 28 FPS                                | FPS metadata                     |
| Duration    | >= 60 seconds                            | Duration metadata                |
| Orientation | Rotation = 0 or 180 degrees and width > height | Width, height, rotation metadata |

## Frame-Level Quality

| Check                | Acceptance Condition          | How It Is Estimated           |
| -------------------- | ----------------------------- | ----------------------------- |
| Average Brightness   | Mean grayscale intensity >= 40 | Frame brightness analysis     |
| Brightness Stability | Std Dev <= 60 across frames    | Brightness variance over time |
| Near-Black Frames    | Mean pixel >= 10 in all frames | Darkness detection            |

## Motion Analysis

| Check            | Acceptance Condition                           | How It Is Estimated       |
| ---------------- | ---------------------------------------------- | ------------------------- |
| Camera Stability | Mean optical flow <= 15 px in >= 80% frame pairs | Farneback dense optical flow |
| Frozen Segments  | No > 30 consecutive frames with SSIM > 0.99    | Native FPS frame similarity  |

## Luminance & Blur

Per-frame classification using the decision table below, followed by segment-level
aggregation. Acceptance condition: (accept + review) frames >= 80% of total frames.

| Condition              | Mean Luminance | Normalized Tenengrad | Raw Tenengrad | Decision |
| ---------------------- | -------------- | -------------------- | ------------- | -------- |
| Dead black             | < 20           | --                   | --            | Reject   |
| Too dark               | 20 - 40        | --                   | --            | Reject   |
| Low light / noise zone | 40 - 70        | Unreliable -- ignore | < 80          | Reject   |
| Low light / noise zone | 40 - 70        | Unreliable -- ignore | 80 - 200      | Review   |
| Low light / noise zone | 40 - 70        | Unreliable -- ignore | > 200         | Accept   |
| Normal range           | 70 - 210       | < 0.04               | --            | Reject   |
| Normal range           | 70 - 210       | 0.04 - 0.10          | --            | Review   |
| Normal range           | 70 - 210       | 0.10 - 0.30          | --            | Accept   |
| Normal range           | 70 - 210       | > 0.30               | --            | Accept   |
| Soft overexposed       | 210 - 235      | < 0.04               | --            | Reject   |
| Soft overexposed       | 210 - 235      | 0.04 - 0.10          | --            | Review   |
| Soft overexposed       | 210 - 235      | > 0.10               | --            | Accept   |
| Blown out              | > 235          | --                   | --            | Reject   |

**Tenengrad computation:** Sobel gradient magnitude. Raw = mean(Gx^2 + Gy^2).
Normalized = raw / (mean_luminance^2 + epsilon). In the low-light noise zone
(luminance 40-70), normalized Tenengrad is unreliable due to noise amplification,
so raw Tenengrad is used instead.

**Segment analysis:** Frames are classified per the table above, then collapsed
into contiguous good/bad segments. "Review" frames count as good for segmentation.
The video passes if the ratio of good frames (accept + review) to total frames
meets the 80% threshold.

## ML Detection

| Check                   | Acceptance Condition                                    | How It Is Estimated              |
| ----------------------- | ------------------------------------------------------- | -------------------------------- |
| Face Presence           | Face detection confidence < 0.8 in all frames           | Per-frame face detection         |
| Participants            | Persons detected <= 1 in >= 95% frames                   | Person detection                 |
| Hand Visibility         | >= 90% frames with both hands detection confidence >= 0.7 | Per-frame hands detection        |
| Hand-Object Interaction | Interaction detected in >= 70% frames                    | Hand + object proximity analysis |
| View Obstruction        | <= 10% frames obstructed                                 | Occlusion detection              |
| Task Clarity            | Dominant action confidence >= 0.6 in >= 80% frames       | Action recognition               |
| Privacy Safety          | Sensitive object detections = 0 in all frames           | Detection of documents/screens   |

## Pipeline Behavior

- **Metadata gate:** If any video metadata check fails, all other checks are skipped.
- **Independent categories:** Frame quality, motion analysis, luminance & blur, and ML
  detection run independently -- a failure in one does not skip others.
- **Statuses:** pass, fail, review (for borderline results), skipped (metadata gate).
