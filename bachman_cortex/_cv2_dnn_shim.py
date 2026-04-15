"""Compat shim for `cv2.dnn.blobFromImage[s]` when cv2 ships without dnn.

Our custom OpenCV-CUDA build turns `BUILD_opencv_dnn=OFF` because OpenCV
4.10 + CMake 4.x chokes on `pyopencv_dnn.hpp` during Python-binding
generation (unresolved `dnn::` namespace, cascading through mcc). The
only thing the rest of this project needs from `cv2.dnn` is the
`blobFromImage` / `blobFromImages` preprocessing helper that `insightface`
uses for SCRFD input — pure resize + mean-subtract + scale + NCHW layout,
no DNN runtime. Shimming those two functions lets SCRFD run on our
CUDA-enabled cv2 without pulling the broken bindings back in.
"""

from __future__ import annotations

import sys
import types
import numpy as np
import cv2


def blobFromImage(
    image: np.ndarray,
    scalefactor: float = 1.0,
    size=None,
    mean=None,
    swapRB: bool = False,
    crop: bool = False,
    ddepth=None,
) -> np.ndarray:
    """Mirror of `cv2.dnn.blobFromImage` for BGR/GRAY input.

    Returns an NCHW float32 blob. `crop` is accepted for signature
    compatibility but treated as letterbox-equivalent (no cropping) to
    match `blobFromImage`'s behaviour when crop=False (the common case).
    """
    img = image
    if size is not None and tuple(size) != (0, 0):
        img = cv2.resize(img, tuple(size))
    if swapRB and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    if mean is not None:
        img = img - np.asarray(mean, dtype=np.float32)
    if scalefactor != 1.0:
        img = img * float(scalefactor)
    if img.ndim == 3:
        blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    elif img.ndim == 2:
        blob = img[np.newaxis, np.newaxis, ...]
    else:
        raise ValueError(f"blobFromImage: unsupported ndim {img.ndim}")
    return np.ascontiguousarray(blob)


def blobFromImages(
    images,
    scalefactor: float = 1.0,
    size=None,
    mean=None,
    swapRB: bool = False,
    crop: bool = False,
    ddepth=None,
) -> np.ndarray:
    """Mirror of `cv2.dnn.blobFromImages`. Stacks per-image NCHW blobs."""
    blobs = [
        blobFromImage(im, scalefactor, size, mean, swapRB, crop, ddepth)
        for im in images
    ]
    return np.concatenate(blobs, axis=0) if blobs else np.empty((0,), dtype=np.float32)


def install() -> bool:
    """Install the shim if cv2.dnn.blobFromImage is missing. Returns True if installed.

    The custom dnn-less CUDA build ships an empty `cv2.dnn` submodule (directory
    present, no functions), so a bare `hasattr(cv2, "dnn")` check is not enough —
    we patch the functions directly onto whatever `cv2.dnn` happens to be.
    """
    dnn = getattr(cv2, "dnn", None)
    if dnn is not None and hasattr(dnn, "blobFromImage"):
        return False
    if dnn is None:
        dnn = types.ModuleType("cv2.dnn")
        cv2.dnn = dnn
        sys.modules["cv2.dnn"] = dnn
    dnn.blobFromImage = blobFromImage
    dnn.blobFromImages = blobFromImages
    return True
