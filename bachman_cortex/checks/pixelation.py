"""Pixelation (block-boundary gradient ratio) accumulator.

Measures per-frame blockiness: ratio of average gradient magnitude at
8x8 block boundaries vs. block interiors. Compression artefacts and
upscaled low-res sources produce blockiness >> 1.0.

Fails the video if fewer than `good_frame_ratio` of sampled frames
land below `max_blockiness_ratio`.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PixelationThresholds:
    good_frame_ratio: float = 0.80
    max_blockiness_ratio: float = 1.5
    block_size: int = 8


@dataclass
class PixelationFinalizeResult:
    pass_fail: bool
    detected: str
    ratio_array: list[float]       # per-sampled-frame
    sample_indices: list[int]
    good_ratio: float
    mean_blockiness: float


def compute_blockiness(frame_bgr: np.ndarray, block_size: int = 8) -> float:
    """Ratio of gradient energy at block boundaries vs. interior.

    ~1.0 → no block artefacts, >> 1.0 → pixelation.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    h_grad = np.abs(np.diff(gray, axis=1))
    v_grad = np.abs(np.diff(gray, axis=0))

    h_boundary_cols = np.arange(block_size - 1, w - 1, block_size)
    h_interior_mask = np.ones(w - 1, dtype=bool)
    h_interior_mask[h_boundary_cols] = False

    h_boundary_energy = (float(np.mean(h_grad[:, h_boundary_cols]))
                         if len(h_boundary_cols) > 0 else 0.0)
    h_interior_energy = (float(np.mean(h_grad[:, h_interior_mask]))
                         if h_interior_mask.any() else 1.0)

    v_boundary_rows = np.arange(block_size - 1, h - 1, block_size)
    v_interior_mask = np.ones(h - 1, dtype=bool)
    v_interior_mask[v_boundary_rows] = False

    v_boundary_energy = (float(np.mean(v_grad[v_boundary_rows, :]))
                         if len(v_boundary_rows) > 0 else 0.0)
    v_interior_energy = (float(np.mean(v_grad[v_interior_mask, :]))
                         if v_interior_mask.any() else 1.0)

    boundary = (h_boundary_energy + v_boundary_energy) / 2.0
    interior = (h_interior_energy + v_interior_energy) / 2.0
    return boundary / (interior + 1e-6)


@dataclass
class PixelationAccumulator:
    thresholds: PixelationThresholds = field(default_factory=PixelationThresholds)

    _ratios: list[float] = field(init=False, default_factory=list)
    _sample_indices: list[int] = field(init=False, default_factory=list)

    def process_frame(self, frame_720p: np.ndarray, frame_idx: int) -> None:
        ratio = compute_blockiness(frame_720p, self.thresholds.block_size)
        self._ratios.append(ratio)
        self._sample_indices.append(frame_idx)

    def finalize(self) -> PixelationFinalizeResult:
        th = self.thresholds
        n = len(self._ratios)
        if n == 0:
            return PixelationFinalizeResult(
                pass_fail=False,
                detected="no_samples",
                ratio_array=[],
                sample_indices=[],
                good_ratio=0.0,
                mean_blockiness=0.0,
            )

        good = sum(1 for r in self._ratios if r <= th.max_blockiness_ratio)
        good_ratio = good / n
        passes = good_ratio >= th.good_frame_ratio
        mean = float(np.mean(self._ratios))
        return PixelationFinalizeResult(
            pass_fail=passes,
            detected=f"good_ratio={good_ratio:.3f}, mean={mean:.3f}",
            ratio_array=list(self._ratios),
            sample_indices=list(self._sample_indices),
            good_ratio=round(good_ratio, 4),
            mean_blockiness=round(mean, 4),
        )
