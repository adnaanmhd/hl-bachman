"""bachman_cortex package init.

Installs a `cv2.dnn` compat shim before any downstream import, because our
custom OpenCV-CUDA build omits the dnn module (see _cv2_dnn_shim.py).
Insightface calls `cv2.dnn.blobFromImage` for SCRFD preprocessing; the
shim reproduces that with numpy.
"""

from bachman_cortex import _cv2_dnn_shim as _cv2_dnn_shim
_cv2_dnn_shim.install()
