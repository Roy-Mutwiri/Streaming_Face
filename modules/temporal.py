"""
Temporal smoothing of face landmarks for live video.

The detector's keypoints move by a pixel or two between frames even when the head is still, so
the swapped face shimmers. A One-Euro filter (Casiez et al.) smooths strongly while the face is
still and opens up as it moves, so there is no visible lag. Each face is tracked across frames by
its position and gets its own filter.
"""
import math
import time
from typing import Dict, List, Optional

import numpy as np

from modules.typing import Face


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.value: Optional[np.ndarray] = None
        self.derivative: Optional[np.ndarray] = None
        self.time = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, value: np.ndarray, now: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if self.value is None or self.value.shape != value.shape:
            self.value, self.derivative, self.time = value.copy(), np.zeros_like(value), now
            return value
        dt = max(now - self.time, 1e-3)
        self.time = now
        a_d = self._alpha(dt, self.d_cutoff)
        self.derivative = a_d * (value - self.value) / dt + (1 - a_d) * self.derivative
        a = self._alpha(dt, self.min_cutoff + self.beta * np.abs(self.derivative))
        self.value = a * value + (1 - a) * self.value
        return self.value.copy()


class _Track:
    def __init__(self, center: np.ndarray, now: float):
        self.center, self.last_seen = center, now
        self.filters: Dict[str, OneEuroFilter] = {}

    def smooth(self, name: str, value, now: float):
        if value is None:
            return None
        if name not in self.filters:
            self.filters[name] = OneEuroFilter()
        return self.filters[name](value, now)


class FaceSmoother:
    """Matches faces to tracks by position each frame and smooths their landmarks."""

    def __init__(self, max_age: float = 0.5):
        self.tracks: List[_Track] = []
        self.max_age = max_age

    def reset(self) -> None:
        self.tracks = []

    def __call__(self, faces: List[Face], now: Optional[float] = None) -> List[Face]:
        now = time.perf_counter() if now is None else now
        self.tracks = [t for t in self.tracks if now - t.last_seen < self.max_age]
        unmatched = list(self.tracks)
        smoothed = []
        for face in sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0])):
            center = np.array([(face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2], dtype=np.float32)
            width = face.bbox[2] - face.bbox[0]
            track = min(unmatched, key=lambda t: np.linalg.norm(t.center - center), default=None)
            if track is None or np.linalg.norm(track.center - center) > width * 0.5:
                track = _Track(center, now)
                self.tracks.append(track)
            else:
                unmatched.remove(track)
            track.center, track.last_seen = center, now
            out = Face(dict(face))
            out.bbox = track.smooth('bbox', face.bbox, now)
            out.kps = track.smooth('kps', face.kps, now)
            out.landmark_2d_106 = track.smooth('landmark_2d_106', face.landmark_2d_106, now)
            smoothed.append(out)
        return smoothed
