"""Turn MediaPipe face landmarks into the 8-value eye feature vector the gaze model uses."""
from dataclasses import dataclass

import numpy as np

FEATURES = ("iris_x", "iris_y_corner", "iris_y_lid", "openness",
            "head_fx", "head_fy", "head_tx", "head_ty")

# MediaPipe refined face mesh indices. The right eye is on the image-left (camera not mirrored).
RIGHT_EYE = dict(iris=468, outer=33, inner=133, upper=159, lower=145)
LEFT_EYE = dict(iris=473, outer=263, inner=362, upper=386, lower=374)


@dataclass
class FaceObservation:
    points: np.ndarray   # (478, 2) landmark positions in camera pixels
    blink_left: float    # MediaPipe blendshape eyeBlinkLeft, 0..1
    blink_right: float   # MediaPipe blendshape eyeBlinkRight, 0..1
    head: np.ndarray     # (4, 4) facial transformation matrix
    t: float             # capture time, seconds


def _eye(p, e):
    """Iris position inside one eye, scaled by eye width: x along the corner axis
    (0 inner .. 1 outer), y below the corner line (skull-fixed) and below the lid
    midline (lids follow gaze), plus lid openness."""
    inner, outer = p[e["inner"]], p[e["outer"]]
    axis = outer - inner
    width = np.linalg.norm(axis)
    ux = axis / width
    uy = np.array([-ux[1], ux[0]])
    if uy[1] < 0:  # same "down" for both eyes, whichever way the corner axis points
        uy = -uy
    rel = p[e["iris"]] - inner
    lid_mid = (p[e["upper"]] + p[e["lower"]]) / 2 - inner
    return (rel @ ux / width,
            rel @ uy / width,
            (rel - lid_mid) @ uy / width,
            np.linalg.norm(p[e["upper"]] - p[e["lower"]]) / width)


def extract_features(obs: FaceObservation, blink_ratio: float = 0.18) -> np.ndarray | None:
    """Feature vector in FEATURES order, or None while either eye is (nearly) closed."""
    rx, ryc, ryl, ro = _eye(obs.points, RIGHT_EYE)
    lx, lyc, lyl, lo = _eye(obs.points, LEFT_EYE)
    if min(ro, lo) < blink_ratio:
        return None
    m = obs.head
    return np.array([
        (lx + (1 - rx)) / 2,   # right eye's x runs the other way across the image
        (lyc + ryc) / 2,
        (lyl + ryl) / 2,
        (lo + ro) / 2,
        m[0, 2], m[1, 2],      # where the face points
        m[0, 3], m[1, 3],      # where the head is
    ])
