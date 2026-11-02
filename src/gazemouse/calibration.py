"""Calibration: show dots, collect eye features per dot, fit and validate a GazeModel."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .gaze_model import GazeModel

CALIB_X = (0.06, 0.35, 0.65, 0.94)
CALIB_Y = (0.08, 0.36, 0.64, 0.92)
VALID_X = (0.22, 0.78)
VALID_Y = (0.25, 0.75)
MIN_SAMPLES = 5        # per dot, or the dot does not count
OUTLIER_RADIUS = 0.05  # around the dot's median (iris_x, iris_y_corner)
MIN_CALIB_DOTS = 12
MIN_VALID_DOTS = 4


@dataclass
class CalibStep:
    phase: Literal["calib", "valid", "done"]
    target: tuple[int, int] | None
    progress: float  # 0..1 within the current dot


@dataclass
class CalibrationResult:
    ok: bool
    reason: str | None  # "face_lost" or None
    model: GazeModel | None
    mean_error_px: float | None
    error_pct: float | None
    calib_dots: int
    valid_dots: int


def _grid(xs, ys, w, h):
    return [(round(x * w), round(y * h)) for y in ys for x in xs]


class CalibrationSession:
    """Time-driven: the caller feeds every camera frame's features (None when no face)."""

    def __init__(self, screen_w: int, screen_h: int, settle_s: float = 0.8, sample_s: float = 1.0):
        self.w, self.settle_s, self.sample_s = screen_w, settle_s, sample_s
        calib = _grid(CALIB_X, CALIB_Y, screen_w, screen_h)
        valid = _grid(VALID_X, VALID_Y, screen_w, screen_h) + [(round(screen_w / 2), round(screen_h / 2))]
        self.dots = [("calib", d) for d in calib] + [("valid", d) for d in valid]

    def start(self, t: float) -> None:
        self.index, self.dot_t0 = 0, t
        self.samples = [[] for _ in self.dots]
        self._result = None

    def update(self, t: float, feat: np.ndarray | None) -> CalibStep:
        duration = self.settle_s + self.sample_s
        while self.index < len(self.dots) and t - self.dot_t0 > duration:
            self.index += 1
            self.dot_t0 += duration
        if self.index == len(self.dots):
            return CalibStep("done", None, 1.0)
        elapsed = t - self.dot_t0
        if feat is not None and self.settle_s < elapsed <= duration:
            self.samples[self.index].append(np.asarray(feat, float))
        phase, target = self.dots[self.index]
        return CalibStep(phase, target, min(max(elapsed / duration, 0.0), 1.0))

    def result(self) -> CalibrationResult:
        if self._result is None:
            self._result = self._compute()
        return self._result

    def _kept(self, i: int) -> np.ndarray | None:
        """The dot's samples minus outliers (blinks, glances away), or None if too few."""
        s = self.samples[i]
        if len(s) < MIN_SAMPLES:
            return None
        s = np.array(s)
        dist = np.linalg.norm(s[:, :2] - np.median(s[:, :2], axis=0), axis=1)
        kept = s[dist <= OUTLIER_RADIUS]
        return kept if len(kept) else None

    def _compute(self) -> CalibrationResult:
        calib, valid = [], []
        for i, (phase, target) in enumerate(self.dots):
            kept = self._kept(i)
            if kept is not None:
                (calib if phase == "calib" else valid).append((target, kept))
        if len(calib) < MIN_CALIB_DOTS or len(valid) < MIN_VALID_DOTS:
            return CalibrationResult(False, "face_lost", None, None, None, len(calib), len(valid))
        feats = np.vstack([k for _, k in calib])
        targets = np.vstack([np.tile(t, (len(k), 1)) for t, k in calib])
        model = GazeModel()
        model.fit(feats, targets)
        errors = [np.linalg.norm(model.predict(k.mean(axis=0)) - t) for t, k in valid]
        mean_err = float(np.mean(errors))
        return CalibrationResult(True, None, model, mean_err, mean_err / self.w * 100,
                                 len(calib), len(valid))
