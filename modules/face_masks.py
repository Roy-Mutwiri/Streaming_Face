"""
Realism helpers for the swap: segmentation masks and skin-tone matching.

Occlusion mask (XSeg): finds what is actually visible of the face, so hands, hair, cups,
glasses or a microphone in front of the target face stay in front instead of being painted over.
Region mask (BiSeNet face parser): follows the real skin / brows / eyes / nose / lips boundary
instead of a convex hull of landmarks, so the swap does not spill onto hair, ears or background.
Tone match: shifts the swapped face's colour statistics toward the target's skin, so the face
matches the neck, ears and lighting of the person it is pasted on.

All functions work on a face crop that is already aligned (the swapper's arcface crop).
"""
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.accel import get_providers
from modules.model_fixes import gpu_friendly_conv_transpose
from modules.trt_engine import create_session as create_trt_session
from modules.utilities import resolve_relative_path

OCCLUDER_MODEL = '../models/xseg_1.onnx'
PARSER_MODEL = '../models/bisenet_resnet_34.onnx'
# BiSeNet (CelebAMask-HQ) labels kept as "face": skin, brows, eyes, glasses, nose, mouth, lips
PARSER_FACE_REGIONS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
PARSER_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PARSER_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SESSIONS: Dict[str, Any] = {}
THREAD_LOCK = threading.Lock()


def _get_session(path: str) -> Any:
    """
    TensorRT FP32 engine when available (XSeg ~5 ms, parser ~4 ms), else onnxruntime CUDA FP32.
    (onnxruntime's own TensorRT provider crashes the process on XSeg, so it is never used here.)
    """
    with THREAD_LOCK:
        if path not in SESSIONS:
            model_path = resolve_relative_path(path)
            if path == OCCLUDER_MODEL:
                # XSeg's asymmetric-padded ConvTranspose layers otherwise run on the CPU
                model_path = gpu_friendly_conv_transpose(model_path, model_path.replace('.onnx', '_gpu.onnx'))
            session = None
            if modules.globals.use_tensorrt and 'CUDAExecutionProvider' in modules.globals.execution_providers:
                session = create_trt_session(model_path, 'fp32')
            if session is None:
                providers = [p for p in get_providers() if not (isinstance(p, tuple) and p[0] == 'TensorrtExecutionProvider')]
                session = onnxruntime.InferenceSession(model_path, providers=providers)
            SESSIONS[path] = session
    return SESSIONS[path]


def _soften(mask: np.ndarray) -> np.ndarray:
    """Feathers a 0..1 mask while keeping its boundary where the model put it."""
    return (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2


def occlusion_mask(aligned_face: np.ndarray) -> np.ndarray:
    """1 where the face is visible, 0 where something covers it. Same size as aligned_face."""
    session = _get_session(OCCLUDER_MODEL)
    blob = cv2.resize(aligned_face, (256, 256))[np.newaxis].astype(np.float32) / 255.0
    mask = session.run(None, {session.get_inputs()[0].name: blob})[0][0, :, :, 0]
    mask = cv2.resize(mask.clip(0, 1).astype(np.float32), aligned_face.shape[1::-1])
    return _soften(mask)


def region_mask(aligned_face: np.ndarray) -> np.ndarray:
    """1 on facial skin/features (no hair, ears, neck, background). Same size as aligned_face."""
    session = _get_session(PARSER_MODEL)
    rgb = cv2.cvtColor(cv2.resize(aligned_face, (512, 512)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - PARSER_MEAN) / PARSER_STD).transpose(2, 0, 1)[np.newaxis]
    feed = {session.get_inputs()[0].name: blob}
    if hasattr(session, 'run_torch'):
        # argmax over the 19 class maps on the GPU; only the 512x512 mask comes back
        def to_mask(outputs):
            import torch
            labels = outputs[0][0].argmax(0)
            regions = torch.tensor(PARSER_FACE_REGIONS, device=labels.device)
            return torch.isin(labels, regions).to(torch.uint8).cpu().numpy()
        mask = session.run_torch(feed, postprocess=to_mask).astype(np.float32)
    else:
        labels = session.run(None, feed)[0][0].argmax(0)
        mask = np.isin(labels, PARSER_FACE_REGIONS).astype(np.float32)
    mask = cv2.resize(mask, aligned_face.shape[1::-1])
    return _soften(mask)


def match_tone(swapped_face: np.ndarray, target_face: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Moves the swapped face's colour toward the target face (LAB mean, plus colour spread),
    measured only where mask > 0.5. strength 0 = untouched, 1 = full match.
    Brightness contrast (L spread) is left alone so the source's facial detail survives.
    """
    if strength <= 0:
        return swapped_face
    region = (mask > 0.5).astype(np.uint8)
    if cv2.countNonZero(region) < 50:
        return swapped_face
    swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
    s_mean, s_std = (v.ravel() for v in cv2.meanStdDev(swapped_lab, mask=region))
    t_mean, t_std = (v.ravel() for v in cv2.meanStdDev(target_lab, mask=region))
    scale = np.array([1.0, (t_std[1] + 1e-3) / (s_std[1] + 1e-3), (t_std[2] + 1e-3) / (s_std[2] + 1e-3)], dtype=np.float32)
    swapped_lab = swapped_lab.astype(np.float32)
    matched = (swapped_lab - s_mean.astype(np.float32)) * scale + t_mean.astype(np.float32)
    result = swapped_lab + (matched - swapped_lab) * strength
    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def estimate_noise(region: np.ndarray) -> float:
    """
    Camera noise level (sigma, 0-255 scale) of an image region, from the median absolute
    high-pass response (robust to edges). Webcams are typically 2-5, clean video below 1.
    """
    h, w = region.shape[:2]
    if h > 160 and w > 160:  # the centre of the face is plenty for a noise estimate
        region = region[(h - 160) // 2:(h + 160) // 2, (w - 160) // 2:(w + 160) // 2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_pass = gray - cv2.GaussianBlur(gray, (3, 3), 0)
    return float(np.median(np.abs(high_pass)) * 1.4826 * 1.25)  # 1.25: calibrated so known synthetic noise reads back 1:1


NOISE_TILE: Optional[np.ndarray] = None
RNG = np.random.default_rng()


def add_grain(region: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Adds luminance grain of the given sigma where mask is set, so a restored (clean) face
    carries the same camera noise as the rest of the frame. Uses a random window of a
    pre-generated noise texture each call (fresh grain every frame, no per-frame RNG cost)."""
    global NOISE_TILE
    if sigma < 0.5:
        return region
    h, w = region.shape[:2]
    if NOISE_TILE is None or NOISE_TILE.shape[0] < h or NOISE_TILE.shape[1] < w:
        size = max(1024, h, w)
        NOISE_TILE = RNG.standard_normal((size * 2, size * 2), dtype=np.float32)
    y, x = RNG.integers(0, NOISE_TILE.shape[0] - h), RNG.integers(0, NOISE_TILE.shape[1] - w)
    grain = cv2.multiply(NOISE_TILE[y:y + h, x:x + w], mask, scale=sigma)
    return cv2.add(region, cv2.merge([grain, grain, grain]), dtype=cv2.CV_8U)


def build_mask(aligned_target: np.ndarray, box_mask: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
    """Final blend mask in aligned-face space for the current mask mode."""
    mode = mode or modules.globals.face_mask_mode
    mask = box_mask
    if mode in ('Occlusion', 'Occlusion+Parser'):
        mask = mask * occlusion_mask(aligned_target)
    if mode in ('Parser', 'Occlusion+Parser'):
        mask = mask * region_mask(aligned_target)
    return mask
