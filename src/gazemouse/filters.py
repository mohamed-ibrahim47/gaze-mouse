"""Smoothing for the gaze point."""
import math

import numpy as np


class OneEuro:
    """One Euro filter: heavy smoothing while the gaze is still, little lag when it jumps.

    The cutoff frequency rises with the (smoothed) speed: cutoff = min_cutoff + beta * |speed|.
    Speed is in px/s, so beta is per px/s.
    """

    D_CUTOFF = 1.0  # Hz, for smoothing the speed estimate itself

    def __init__(self, min_cutoff: float = 0.6, beta: float = 0.004):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.reset()

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        speed = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.D_CUTOFF, dt) * (speed - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


# Fractions of the screen width, so every resolution feels the same.
DEAD_ZONE = 0.02   # the gaze must move this far from the cursor before the cursor follows
STOP_ZONE = 0.004  # ...and the glide ends once the cursor is this close to the gaze
GLIDE_GAIN = 6.0   # 1/s: glide speed = gain x distance left, so it eases in to the target
MAX_SPEED = 1.2    # screen widths per second


class CalmCursor:
    """Rests while the gaze only wobbles; glides at a capped speed when the gaze really moves.

    One Euro makes the wobble smaller but never zero, and follows a jump almost at once.
    This runs after it: a still cursor to blink at, and a move the eye can follow.
    """

    def __init__(self, screen_w: float, dead_zone: float = DEAD_ZONE, max_speed: float = MAX_SPEED):
        self.dead, self.stop, self.max_speed = dead_zone * screen_w, STOP_ZONE * screen_w, max_speed * screen_w
        self.reset()

    def reset(self) -> None:
        self.x = self.t = None
        self.moving = False

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        left = float(np.linalg.norm(x - self.x))
        if left > self.dead:
            self.moving = True
        if self.moving:
            step = min(min(self.max_speed, GLIDE_GAIN * left) * dt, left)
            if left > 0:
                self.x = self.x + (x - self.x) * (step / left)
            if left - step <= self.stop:
                self.moving = False
        return self.x
