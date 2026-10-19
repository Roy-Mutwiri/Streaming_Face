"""
Virtual background for the Live preview.

The person is cut out with Robust Video Matting (RVM, mobilenetv3), a recurrent matting model
made for video: it keeps state between frames, so hair edges are soft and the matte does not
flicker. The background is then replaced by one of:
  - 'None'                      untouched camera background
  - 'Blur' / 'Blur Strong'      the real room, defocused like a wide-aperture camera
  - built-in streaming rooms    generated once (no image files needed)
  - any image in backgrounds/   your own pictures (jpg/png/webp)

Light wrap (background light bleeding onto the person's edges) and a faint colour tint from the
scene make the cut-out sit in the new room instead of looking pasted on.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.accel import get_providers
from modules.utilities import resolve_relative_path

MATTING_MODEL = '../models/rvm_mobilenetv3_fp32.onnx'
BACKGROUND_DIR = resolve_relative_path('../backgrounds')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
CANVAS = (1280, 720)


# ---------------------------------------------------------------------------------------------
# Built-in streaming room scenes (drawn with OpenCV, 1280x720)
# ---------------------------------------------------------------------------------------------

def _vertical_gradient(top, bottom, size=CANVAS) -> np.ndarray:
    w, h = size
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return (np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t).repeat(w, axis=1)


def _glow(canvas: np.ndarray, draw: Callable[[np.ndarray], None], radius: float, strength: float = 1.0) -> None:
    """Draws shapes on a black layer, blooms them and adds them to the canvas (light emission)."""
    layer = np.zeros_like(canvas)
    draw(layer)
    bloom = cv2.GaussianBlur(layer, (0, 0), radius) * 2.2 + cv2.GaussianBlur(layer, (0, 0), radius * 4) * 1.4
    canvas += (layer + bloom) * strength


def _bokeh(canvas: np.ndarray, rng: np.random.Generator, colors, count: int, y_range, size_range, alpha: float) -> None:
    layer = np.zeros_like(canvas)
    w = canvas.shape[1]
    for _ in range(count):
        center = (int(rng.uniform(0, w)), int(rng.uniform(*y_range)))
        radius = int(rng.uniform(*size_range))
        color = np.array(colors[rng.integers(len(colors))], np.float32) * rng.uniform(0.5, 1.0)
        cv2.circle(layer, center, radius, color.tolist(), -1, cv2.LINE_AA)
    canvas += cv2.GaussianBlur(layer, (0, 0), 3) * alpha


def _shelf(canvas: np.ndarray, x: int, y: int, width: int, color, items_rng: np.random.Generator, item_colors) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + 10), color, -1)
    cx = x + 10
    while cx < x + width - 30:
        iw, ih = int(items_rng.uniform(14, 40)), int(items_rng.uniform(30, 75))
        c = np.array(item_colors[items_rng.integers(len(item_colors))], np.float32) * items_rng.uniform(0.5, 1.0)
        cv2.rectangle(canvas, (cx, y - ih), (cx + iw, y), c.tolist(), -1)
        cx += iw + int(items_rng.uniform(4, 18))


def _hexagon(center, radius) -> np.ndarray:
    angles = np.deg2rad(np.arange(0, 360, 60) + 30)
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], 1).astype(np.int32)


def _neon_gaming_room() -> np.ndarray:
    rng = np.random.default_rng(1)
    c = _vertical_gradient((40, 14, 18), (14, 6, 8))  # BGR: deep navy/purple wall
    _bokeh(c, rng, [(255, 80, 200), (255, 200, 60), (200, 60, 255)], 40, (40, 420), (6, 22), 0.35)
    for x, y, color in [(260, 180, (255, 60, 220)), (330, 220, (255, 200, 40)), (330, 140, (255, 60, 220)), (400, 180, (200, 40, 255))]:
        _glow(c, lambda l, x=x, y=y, color=color: cv2.fillPoly(l, [_hexagon((x, y), 36)], color), 10, 0.55)
    _shelf(c, 820, 230, 330, (35, 25, 30), rng, [(200, 120, 255), (255, 200, 80), (80, 220, 255)])
    _shelf(c, 820, 360, 330, (35, 25, 30), rng, [(200, 120, 255), (255, 200, 80), (80, 220, 255)])
    _glow(c, lambda l: cv2.line(l, (0, 40), (1280, 40), (255, 60, 230), 6), 12, 0.7)       # ceiling LED strip
    _glow(c, lambda l: cv2.line(l, (0, 560), (1280, 560), (255, 220, 40), 5), 14, 0.6)     # desk RGB strip
    cv2.rectangle(c, (0, 565), (1280, 720), (22, 14, 16), -1)                               # desk
    return c


def _cozy_streamer() -> np.ndarray:
    rng = np.random.default_rng(2)
    c = _vertical_gradient((60, 80, 110), (30, 42, 60))  # warm beige wall
    for _ in range(2):
        pts = np.stack([np.linspace(0, 1280, 60), 70 + 30 * np.sin(np.linspace(0, 6, 60)) + rng.uniform(0, 30)], 1).astype(np.int32)
        cv2.polylines(c, [pts], False, (40, 45, 50), 2, cv2.LINE_AA)
        _glow(c, lambda l, pts=pts: [cv2.circle(l, tuple(int(v) for v in p), 4, (120, 210, 255), -1) for p in pts[::3]], 6, 0.9)  # fairy lights
    _shelf(c, 80, 300, 380, (40, 60, 85), rng, [(60, 90, 150), (90, 140, 120), (150, 110, 70), (70, 70, 140)])
    _shelf(c, 80, 450, 380, (40, 60, 85), rng, [(60, 90, 150), (90, 140, 120), (150, 110, 70), (70, 70, 140)])
    _glow(c, lambda l: cv2.circle(l, (1080, 330), 60, (90, 190, 255), -1), 40, 0.6)         # lamp
    cv2.rectangle(c, (1060, 390), (1100, 600), (30, 40, 55), -1)
    _bokeh(c, rng, [(120, 210, 255), (90, 170, 255)], 25, (0, 250), (5, 14), 0.4)
    cv2.rectangle(c, (0, 600), (1280, 720), (30, 45, 65), -1)
    return c


def _cyber_blue_studio() -> np.ndarray:
    rng = np.random.default_rng(3)
    c = _vertical_gradient((70, 30, 10), (20, 8, 4))
    for x in (140, 330, 950, 1140):
        _glow(c, lambda l, x=x: cv2.line(l, (x, 60), (x, 520), (255, 220, 60), 8), 14, 0.65)  # vertical light tubes
    _glow(c, lambda l: cv2.line(l, (0, 600), (1280, 600), (255, 150, 30), 4), 16, 0.5)
    _bokeh(c, rng, [(255, 230, 120), (255, 160, 40)], 30, (60, 500), (5, 16), 0.3)
    floor = c[440:520][::-1] * 0.35  # glossy floor reflection
    c[600:680] = c[600:680] * 0.5 + cv2.resize(floor, (1280, 80)) * 0.5
    return c


def _sunset_loft() -> np.ndarray:
    rng = np.random.default_rng(4)
    c = _vertical_gradient((45, 40, 50), (25, 22, 28))
    window = _vertical_gradient((150, 110, 255), (90, 170, 255), (640, 380))  # sunset sky (BGR)
    x = 0
    while x < 640:  # city skyline
        bw, bh = int(rng.uniform(30, 80)), int(rng.uniform(90, 260))
        cv2.rectangle(window, (x, 380 - bh), (x + bw, 380), (60, 45, 55), -1)
        for wy in range(380 - bh + 12, 375, 18):
            for wx in range(x + 6, x + bw - 6, 14):
                if rng.random() < 0.35:
                    cv2.rectangle(window, (wx, wy), (wx + 5, wy + 7), (120, 210, 255), -1)
        x += bw + int(rng.uniform(2, 10))
    c[120:500, 320:960] = window
    for i, xx in enumerate(range(320, 961, 160)):  # window frame
        cv2.line(c, (xx, 120), (xx, 500), (30, 28, 32), 10)
    cv2.rectangle(c, (320, 120), (960, 500), (30, 28, 32), 14)
    _glow(c, lambda l: cv2.rectangle(l, (330, 130), (950, 490), (40, 90, 160), -1), 60, 0.25)  # window light spill
    cv2.rectangle(c, (0, 600), (1280, 720), (35, 40, 55), -1)
    return c


def _pro_studio_dark() -> np.ndarray:
    w, h = CANVAS
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    spot = np.exp(-(((xx - w * 0.5) / (w * 0.42)) ** 2 + ((yy - h * 0.45) / (h * 0.55)) ** 2))
    base = np.array([38, 36, 34], np.float32)
    c = base * (0.35 + 1.1 * spot[..., None])
    rng = np.random.default_rng(5)
    c += rng.normal(0, 2.0, c.shape).astype(np.float32)  # backdrop texture
    return c


def _purple_haze() -> np.ndarray:
    rng = np.random.default_rng(6)
    c = _vertical_gradient((90, 20, 70), (40, 10, 30))
    _bokeh(c, rng, [(255, 120, 220), (255, 90, 160), (200, 160, 255)], 70, (0, 720), (10, 40), 0.35)
    _glow(c, lambda l: cv2.ellipse(l, (640, 120), (420, 60), 0, 0, 360, (255, 80, 200), -1), 50, 0.35)
    return c


BUILT_IN_SCENES: Dict[str, Callable[[], np.ndarray]] = {
    'Neon Gaming Room': _neon_gaming_room,
    'Cozy Streamer': _cozy_streamer,
    'Cyber Blue Studio': _cyber_blue_studio,
    'Sunset Loft': _sunset_loft,
    'Pro Studio (Dark)': _pro_studio_dark,
    'Purple Haze': _purple_haze,
}
BLUR_OPTIONS = {'Blur': 12, 'Blur Strong': 28}

_SCENE_CACHE: Dict[str, np.ndarray] = {}
_SCENE_LOCK = threading.Lock()


def preload(name: str) -> None:
    """Loads the matting model and renders the scene ahead of time."""
    MATTER(np.zeros((360, 640, 3), np.uint8))
    MATTER.reset()
    if name not in BLUR_OPTIONS and name != 'None':
        scene = _scene(name, CANVAS)
        if scene is not None:
            _scene_extras(name, scene)


def background_options() -> List[str]:
    """Dropdown entries: None, blurs, built-in rooms, then images found in backgrounds/."""
    os.makedirs(BACKGROUND_DIR, exist_ok=True)
    custom = sorted(f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(IMAGE_EXTENSIONS))
    return ['None'] + list(BLUR_OPTIONS) + list(BUILT_IN_SCENES) + custom


def _scene(name: str, size) -> Optional[np.ndarray]:
    """Background image for `name`, resized/cropped to `size` (w, h), cached."""
    key = f'{name}|{size[0]}x{size[1]}'
    with _SCENE_LOCK:
        if key not in _SCENE_CACHE:
            if name in BUILT_IN_SCENES:
                image = np.clip(BUILT_IN_SCENES[name](), 0, 255).astype(np.uint8)
                image = cv2.GaussianBlur(image, (0, 0), 2.5)  # gentle depth of field, like a real room behind the streamer
            else:
                image = cv2.imread(os.path.join(BACKGROUND_DIR, name))
                if image is None:
                    return None
            _SCENE_CACHE[key] = _cover(image, size)
        return _SCENE_CACHE[key]


def _cover(image: np.ndarray, size) -> np.ndarray:
    """Scales and centre-crops an image to fill size (w, h) without distortion."""
    w, h = size
    scale = max(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, (int(np.ceil(image.shape[1] * scale)), int(np.ceil(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    y, x = (resized.shape[0] - h) // 2, (resized.shape[1] - w) // 2
    return resized[y:y + h, x:x + w].astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Matting
# ---------------------------------------------------------------------------------------------

def _firm(alpha: np.ndarray, low: float = 0.08, high: float = 0.55) -> np.ndarray:
    """
    Smoothstep contrast on the matte: RVM leaves the body semi-transparent where it is unsure
    (e.g. against a dark chair), which shows the new room through the person. Values above
    `high` become solid, below `low` vanish, and the soft transition at the edges is kept.
    """
    t = np.clip((alpha - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


class Matter:
    """Recurrent person matting (RVM). Call reset() when a new video stream starts."""

    def __init__(self):
        self.session = None
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.state = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        self.last_alpha: Optional[np.ndarray] = None
        self.calls = 0

    def _get_session(self):
        if self.session is None:
            providers = [p for p in get_providers() if not (isinstance(p, tuple) and p[0] == 'TensorrtExecutionProvider')]
            self.session = onnxruntime.InferenceSession(resolve_relative_path(MATTING_MODEL), providers=providers)
        return self.session

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Alpha matte (float32, 0..1, frame size) of the people in a BGR frame."""
        h, w = frame.shape[:2]
        # The matte is refreshed every `matting_interval` frames and reused in between: the GPU is
        # already busy with the swap, and a person moves only a few pixels in 1/30 s
        self.calls += 1
        interval = max(1, int(modules.globals.matting_interval))
        if self.last_alpha is not None and self.last_alpha.shape == (h, w) and self.calls % interval:
            return self.last_alpha
        # Matte at <=640px wide and upscale: RVM's matte is smooth, so this costs no visible quality
        scale = min(1.0, 640 / w)
        small = cv2.resize(frame, (int(w * scale) // 4 * 4, int(h * scale) // 4 * 4), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        src = cv2.dnn.blobFromImage(small, 1 / 255.0, swapRB=True)
        ratio = min(1.0, 512 / max(small.shape[:2]))  # RVM's internal working resolution (~512px long side)
        with self.lock:
            session = self._get_session()
            _, alpha, *self.state = session.run(None, {
                'src': src, 'r1i': self.state[0], 'r2i': self.state[1], 'r3i': self.state[2], 'r4i': self.state[3],
                'downsample_ratio': np.array([ratio], np.float32)})
        alpha = _firm(alpha[0, 0])
        self.last_alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR) if alpha.shape != (h, w) else alpha
        return self.last_alpha


MATTER = Matter()
# Matting runs on this thread in parallel with face detection
MATTE_POOL = ThreadPoolExecutor(max_workers=1)


_SOFT_CACHE: Dict[str, tuple] = {}


def _blur_small(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur done at quarter resolution and scaled back (same look, ~16x less work)."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), max(sigma / 4, 0.5)), (w, h), interpolation=cv2.INTER_LINEAR)


def _scene_extras(name: str, scene: np.ndarray):
    """Per-scene cache: uint8 scene, softened scene for light wrap, and the scene's colour tint."""
    key = f'{name}|{scene.shape[1]}x{scene.shape[0]}'
    if key not in _SOFT_CACHE:
        tint = scene.reshape(-1, 3).mean(0) / max(float(scene.mean()), 1.0)
        _SOFT_CACHE[key] = (scene.astype(np.uint8), _blur_small(scene, 8).astype(np.uint8), tuple(float(0.9 + 0.1 * t) for t in tint) + (1.0,))
    return _SOFT_CACHE[key]


def compose(frame: np.ndarray, alpha: np.ndarray, background: Optional[str] = None) -> np.ndarray:
    """Puts the person (alpha) from `frame` in front of the selected background."""
    background = background or modules.globals.live_background
    if background == 'None' or alpha is None:
        return frame
    h, w = frame.shape[:2]
    if alpha.shape != (h, w):
        alpha = cv2.resize(alpha, (w, h))
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    small_size = (w // 4, h // 4)
    alpha_small = cv2.resize(alpha, small_size, interpolation=cv2.INTER_AREA)
    if background in BLUR_OPTIONS:
        # Blur only the room: blur the frame with the person removed and renormalise, so no halo.
        # Done entirely at quarter resolution -- the result is blurred anyway.
        sigma = BLUR_OPTIONS[background] / 4
        room_weight = 1.0 - alpha_small
        room = cv2.GaussianBlur(cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA).astype(np.float32) * room_weight[..., None], (0, 0), sigma)
        weight = cv2.GaussianBlur(room_weight, (0, 0), sigma) + 1e-3
        scene = cv2.resize(np.clip(room / weight[..., None], 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        scene_soft = scene
    else:
        image = _scene(background, (w, h))
        if image is None:
            return frame
        scene, scene_soft, tint = _scene_extras(background, image)
        frame = cv2.multiply(frame, tint)  # faint colour tint from the scene's light on the person
    # Light wrap: background light bleeding onto the person's edges, like a real lens
    edge_small = np.clip(cv2.GaussianBlur(1.0 - alpha_small, (0, 0), 1.5) * alpha_small * 1.6, 0, 0.45)
    edge = cv2.resize(edge_small, (w, h), interpolation=cv2.INTER_LINEAR)
    person = cv2.blendLinear(frame, scene_soft, 1.0 - edge, edge)
    return cv2.blendLinear(person, scene, alpha, 1.0 - alpha)
