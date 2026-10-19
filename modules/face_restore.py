"""
Real-time face restoration (GPEN-BFR) used after the swap.

inswapper only outputs a 128x128 face, which looks soft once it is scaled up to the size of
the face in the frame. GPEN restores skin, eye and teeth detail on the swapped face, which
makes the result look sharp and natural. With TensorRT it costs a few milliseconds per face.
"""
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.accel import get_providers
from modules.utilities import resolve_relative_path

MODELS = {
    'GPEN-256': ('../models/gpen_bfr_256.onnx', 256),
    'GPEN-512': ('../models/gpen_bfr_512.onnx', 512),
}

# 5-point FFHQ alignment template (eyes, nose, mouth corners), normalised to 0..1
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32)

SESSIONS: Dict[str, Tuple[Any, int]] = {}
PASTE_MASKS: Dict[int, np.ndarray] = {}
THREAD_LOCK = threading.Lock()


def _strip_initializer_inputs(path: str) -> None:
    """The published GPEN models list their weights as graph inputs, which blocks constant folding and
    confuses TensorRT. Rewrite the file once without them."""
    import onnx
    model = onnx.load(path)
    initializers = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    if len(inputs) != len(model.graph.input):
        del model.graph.input[:]
        model.graph.input.extend(inputs)
        onnx.save(model, path)


def get_session(model_name: str) -> Optional[Tuple[Any, int]]:
    if model_name not in MODELS:
        return None
    with THREAD_LOCK:
        if model_name not in SESSIONS:
            path, size = MODELS[model_name]
            path = resolve_relative_path(path)
            _strip_initializer_inputs(path)
            # In FP16 GPEN's restored face comes out washed out with flattened skin texture, so keep
            # TensorRT in FP32/TF32 (exact, and for GPEN-256 even faster than FP16: ~6.5 ms)
            session = onnxruntime.InferenceSession(path, providers=get_providers(fp16=False))
            SESSIONS[model_name] = (session, size)
    return SESSIONS[model_name]


def _get_paste_mask(size: int) -> np.ndarray:
    """Feathered square mask in the aligned crop space, so the restored face fades into the frame."""
    if size not in PASTE_MASKS:
        mask = np.zeros((size, size), dtype=np.float32)
        pad = int(size * 0.08)
        mask[pad:size - pad, pad:size - pad] = 1.0
        PASTE_MASKS[size] = cv2.GaussianBlur(mask, (0, 0), size * 0.04)
    return PASTE_MASKS[size]


def restore_face(frame: np.ndarray, kps: np.ndarray, model_name: Optional[str] = None,
                 blend: Optional[float] = None) -> np.ndarray:
    """Restores the face described by the 5 keypoints `kps` inside `frame` and returns the new frame."""
    model_name = model_name or modules.globals.live_enhancer
    blend = modules.globals.enhancer_blend if blend is None else blend
    loaded = get_session(model_name)
    if loaded is None or kps is None or blend <= 0:
        return frame
    session, size = loaded

    matrix = cv2.estimateAffinePartial2D(kps.astype(np.float32), FFHQ_TEMPLATE * size, method=cv2.LMEDS)[0]
    if matrix is None:
        return frame
    crop = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    output = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    restored = cv2.cvtColor(np.clip((output.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if blend < 1:
        restored = cv2.addWeighted(restored, blend, crop, 1 - blend, 0)

    h, w = frame.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    # Only paste back over the region the aligned crop covers
    corners = cv2.transform(np.array([[[0, 0], [size, 0], [0, size], [size, size]]], dtype=np.float32), inverse)[0]
    x1, y1 = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(corners.max(axis=0)).astype(int), [w, h])
    if x2 <= x1 or y2 <= y1:
        return frame
    inverse[:, 2] -= (x1, y1)
    region = frame[y1:y2, x1:x2]
    rw, rh = x2 - x1, y2 - y1
    restored_back = cv2.warpAffine(restored, inverse, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(_get_paste_mask(size), inverse, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0)
    # Edits the frame in place (callers own it) and returns it for convenience
    frame[y1:y2, x1:x2] = cv2.blendLinear(restored_back, region, mask, 1.0 - mask)
    return frame


def restore_faces(frame: np.ndarray, kps_list) -> np.ndarray:
    """Restores every face (given by its 5 keypoints in frame coordinates) in an already swapped frame."""
    for kps in kps_list:
        frame = restore_face(frame, kps)
    return frame
