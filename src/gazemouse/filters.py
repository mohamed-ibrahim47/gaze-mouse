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
