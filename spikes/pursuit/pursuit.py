"""Pursuit calibration spike: the gliding-dot path, the eye's lag behind it, model fits and scoring.

    python pursuit.py runs/<file>.npz     (re-scores a saved run)

Blocks in a run: 0 calibration dots (head still), 1 test dots (head still), 2 test dots (head free),
3 gliding dot (head still), 4 gliding dot (head turns with it), 5 still dots held while the head nods
and turns. Dot blocks store the dot's position in `target`; gliding blocks store `t_rel` (seconds
since the block began), and the target is path(t_rel - lag), because the eye trails the dot. Hold
dots store `t_rel` since that dot appeared.

"Angles" models predict the angle from the eyes to the gaze point (head angle + eye angle) and turn
it into pixels using where the head is (3D face origin), so head position and distance are handled
by geometry instead of being learned.
"""
import itertools
import sys

import numpy as np

from gazemouse.calibration import MIN_SAMPLES, OUTLIER_RADIUS

# Lissajous path: smooth (no corners for the eye to cut), sweeps the whole screen, repeats after 30 s.
PATH_S = 30.0
PERIOD_X, PERIOD_Y = 7.5, 10.0   # 4 sweeps across, 3 up and down; about 13 deg/s at most on a laptop
AMP_X, AMP_Y = 0.44, 0.42        # fractions of the screen, so the dot reaches 6% / 8% from the edges
PHASES = {3: 0.0, 4: np.pi}      # the head-free pass runs the curve the other way round
LEAD_IN_S = 1.0                  # the eye is still catching the dot
LAGS = np.arange(0.0, 1.0, 1 / 60)  # camera + processing delay: 250 ms at 30 frames/s, 600 ms at 10
RIDGE = 1e-3                     # per sample, on standardised inputs
ROBUST_K = 3.0                   # refit without frames further than this x the median miss

HOLD_SETTLE_S = 1.0             # skip the first second of each held dot
HOLD = ((0.5, 0.5, 10.0), (0.25, 0.3, 7.0), (0.75, 0.7, 7.0))  # x, y fraction of the screen, seconds

TEST_BLOCKS = {1: "test, head still", 2: "test, head free"}
SOURCES = {"dots": 0, "glide still": 3, "glide head": 4, "hold": 5}
# name -> (space, inputs, training sources)
MODELS = {
    "Dots (current)": ("px", "eye", ("dots",)),
    "Dots+glides, head (px)": ("px", "eye+head", ("dots", "glide still", "glide head")),
    "Angles: dots+glides": ("angle", "eye+head", ("dots", "glide still", "glide head")),
    "Angles: dots+glides+hold": ("angle", "eye+head", ("dots", "glide still", "glide head", "hold")),
    "Angles: dots+glide+hold": ("angle", "eye+head", ("dots", "glide still", "hold")),
}
CURRENT, CANDIDATE = "Dots (current)", "Angles: dots+glide+hold"


def path(t_rel, phase: float = 0.0) -> np.ndarray:
    """(N, 2) dot position as a fraction of the screen. Starts at the centre."""
    t = np.atleast_1d(np.asarray(t_rel, float))
    return np.column_stack([0.5 + AMP_X * np.sin(2 * np.pi * t / PERIOD_X + phase),
                            0.5 + AMP_Y * np.sin(2 * np.pi * t / PERIOD_Y)])


def head_angles(rt: np.ndarray) -> np.ndarray:
    """(N, 4, 4) MediaPipe head transforms -> (N, 2) yaw, pitch in degrees (where the face points)."""
    R = np.asarray(rt, float).reshape(-1, 4, 4)[:, :3, :3]
    R = R / np.cbrt(np.linalg.det(R))[:, None, None]
    fwd = R[:, :, 2]
    return np.degrees(np.column_stack([np.arctan2(fwd[:, 0], fwd[:, 2]), np.arctan2(fwd[:, 1], fwd[:, 2])]))


def inputs(kind: str, feat, rt, origin) -> np.ndarray:
    parts = [np.atleast_2d(feat)[:, :4]]
    if "head" in kind:
        parts.append(head_angles(rt))
    elif "yaw" in kind:
        parts.append(head_angles(rt)[:, :1])
    if "pos" in kind:
        parts.append(np.atleast_2d(origin))
    return np.column_stack(parts)


class Ridge:
    def __init__(self, A: np.ndarray, targets: np.ndarray, lam: float = RIDGE):
        self.mean, self.std = A.mean(axis=0), A.std(axis=0) + 1e-9
        Z = self._design(A)
        reg = lam * len(A) * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        self.weights = np.linalg.solve(Z.T @ Z + reg, Z.T @ targets)

    def _design(self, A):
        return np.column_stack([np.ones(len(A)), (A - self.mean) / self.std])

    def __call__(self, A: np.ndarray) -> np.ndarray:
        return self._design(A) @ self.weights


def robust_ridge(A: np.ndarray, targets: np.ndarray) -> Ridge:
    """Fit, drop the frames it misses by far (blinks, catch-up jumps, glances away), fit again."""
    model = Ridge(A, targets)
    miss = np.linalg.norm(model(A) - targets, axis=1)
    keep = miss <= ROBUST_K * np.median(miss)
    return Ridge(A[keep], targets[keep]) if keep.sum() >= 20 else model


class AngleModel:
    """Ridge from inputs to the angle (degrees) from the eyes to the gaze point; pixels via head position.
    The webcam is taken to sit at the top middle of the screen, in the screen's plane."""

    def __init__(self, A, target_px, origin, screen_px, screen_cm):
        self.w = float(screen_px[0])
        self.pxcm = self.w / float(screen_cm[0])
        best = None
        for sx, sy in itertools.product((-1, 1), (-1, 1)):  # which way the face origin's axes point
            self.sx, self.sy = sx, sy
            model = robust_ridge(A, self._angles(target_px, origin))
            miss = np.median(np.linalg.norm(self._pixels(model(A), origin) - target_px, axis=1))
            if best is None or miss < best[0]:
                best = (miss, sx, sy, model)
        _, self.sx, self.sy, self.model = best

    def _angles(self, px, origin):
        tx, ty = (px[:, 0] - self.w / 2) / self.pxcm, px[:, 1] / self.pxcm
        return np.degrees(np.column_stack([np.arctan2(tx - self.sx * origin[:, 0], origin[:, 2]),
                                           np.arctan2(ty - self.sy * origin[:, 1], origin[:, 2])]))

    def _pixels(self, angles, origin):
        r = np.radians(angles)
        return np.column_stack([(self.sx * origin[:, 0] + origin[:, 2] * np.tan(r[:, 0])) * self.pxcm + self.w / 2,
                                (self.sy * origin[:, 1] + origin[:, 2] * np.tan(r[:, 1])) * self.pxcm])

    def __call__(self, A, origin):
        return self._pixels(self.model(A), np.atleast_2d(origin))


def clean(run: dict) -> np.ndarray:
    """Mask of dot-block frames to keep: per dot, drop outliers the way CalibrationSession does."""
    keep = np.zeros(len(run["block"]), bool)
    for b in (0, 1, 2):
        for d in np.unique(run["dot"][run["block"] == b]):
            idx = np.flatnonzero((run["block"] == b) & (run["dot"] == d))
            f = run["feat"][idx, :2]
            ok = idx[np.linalg.norm(f - np.median(f, axis=0), axis=1) <= OUTLIER_RADIUS]
            if len(ok) >= MIN_SAMPLES:
                keep[ok] = True
    return keep


def glide_targets(run: dict, block: int, lag: float, size) -> tuple[np.ndarray, np.ndarray]:
    """(mask, targets in px) for a gliding block, with the eye `lag` seconds behind the dot."""
    m = (run["block"] == block) & (run["t_rel"] >= LEAD_IN_S)
    return m, path(run["t_rel"][m] - lag, PHASES[block]) * size


def estimate_lag(run: dict, size) -> float:
    """The lag at which the eye features fit the head-still glide best."""
    best, best_miss = 0.0, np.inf
    for lag in LAGS:
        m, y = glide_targets(run, 3, lag, size)
        if m.sum() < 30:
            return 0.0
        A = inputs("eye", run["feat"][m], run["rt"][m], run["origin"][m])
        miss = np.median(np.linalg.norm(Ridge(A, y)(A) - y, axis=1))
        if miss < best_miss:
            best, best_miss = float(lag), miss
    return best


def fit(run: dict, screen_px) -> tuple[dict, float]:
    """(name -> predict(feat, rt, origin) -> (N, 2) physical px, lag in seconds)."""
    size = np.asarray(screen_px, float)
    lag = estimate_lag(run, size)
    keep = clean(run)
    data = {"dots": (keep & (run["block"] == 0), run["target"]),
            "hold": ((run["block"] == 5) & (run["t_rel"] >= HOLD_SETTLE_S), run["target"])}
    for name in ("glide still", "glide head"):
        m, y = glide_targets(run, SOURCES[name], lag, size)
        full = np.zeros((len(m), 2))
        full[m] = y
        data[name] = (m, full)
    models = {}
    for name, (space, kind, sources) in MODELS.items():
        masks = [data[s][0] for s in sources]
        if not all(mk.sum() >= 20 for mk in masks):
            continue
        A = np.vstack([inputs(kind, run["feat"][mk], run["rt"][mk], run["origin"][mk]) for mk in masks])
        y = np.vstack([data[s][1][data[s][0]] for s in sources])
        if space == "angle":
            org = np.vstack([run["origin"][mk] for mk in masks])
            model = AngleModel(A, y, org, size, run["screen_cm"])
            models[name] = (lambda feat, rt, origin, model=model, kind=kind:
                            model(inputs(kind, feat, rt, origin), origin))
        else:
            model = robust_ridge(A, y)
            models[name] = lambda feat, rt, origin, model=model, kind=kind: model(inputs(kind, feat, rt, origin))
    return models, lag


def score(run: dict, models: dict, screen_px) -> dict:
    """name -> {block: (mean error, typical (median) error, shake, dots, mean |x|, mean |y|)}, % of width."""
    w = screen_px[0]
    keep = clean(run)
    out = {}
    for name, predict in models.items():
        out[name] = {}
        for b in TEST_BLOCKS:
            errs, jitter = [], []
            for d in np.unique(run["dot"][run["block"] == b]):
                i = keep & (run["block"] == b) & (run["dot"] == d)
                if not i.any():
                    continue
                p = predict(run["feat"][i], run["rt"][i], run["origin"][i])
                errs.append(p.mean(axis=0) - run["target"][i][0])
                jitter.append(np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean()))
            if errs:
                e = np.array(errs)
                n = np.linalg.norm(e, axis=1)
                out[name][b] = (n.mean() / w * 100, np.median(n) / w * 100, np.mean(jitter) / w * 100, len(e),
                                np.abs(e[:, 0]).mean() / w * 100, np.abs(e[:, 1]).mean() / w * 100)
    return out


def table(scores: dict) -> str:
    lines = ["Error and shake, in % of screen width (lower is better; the target is 6% or less)",
             "average = mean over the dots, typical = the middle dot (ignores one bad dot), x/y = per axis",
             ""]
    for b, title in TEST_BLOCKS.items():
        lines.append(f"{title:<27}{'average':>9}{'typical':>9}{'shake':>8}{'dots':>6}{'x':>6}{'y':>6}")
        for name, s in scores.items():
            if b in s:
                err, med, jit, n, ex, ey = s[b]
                lines.append(f"  {name:<25}{err:>9.1f}{med:>9.1f}{jit:>8.2f}{n:>6}{ex:>6.1f}{ey:>6.1f}")
        lines.append("")
    return "\n".join(lines)


def head_report(run: dict) -> str:
    names = {0: "dots", 3: "glide, head still", 4: "glide, head turning", 5: "hold, head moving",
             1: "test, head still", 2: "test, head free"}
    lines = ["How much the head turned (degrees: yaw range left-right, pitch range up-down, 5th-95th %):"]
    for b, name in names.items():
        m = run["block"] == b
        if m.sum() >= 5:
            a = head_angles(run["rt"][m])
            lo, hi = np.percentile(a, 5, axis=0), np.percentile(a, 95, axis=0)
            lines.append(f"  {name:<22} yaw {hi[0] - lo[0]:5.1f}   pitch {hi[1] - lo[1]:5.1f}   "
                         f"distance {np.asarray(run['origin'])[m, 2].mean():.0f} cm")
    return "\n".join(lines)


def report(run: dict, scores: dict, lag: float) -> str:
    return (table(scores) + f"Eye lag behind the gliding dot: {lag * 1000:.0f} ms\n" + head_report(run) + "\n")


def main(path_: str) -> None:
    run = dict(np.load(path_))
    models, lag = fit(run, run["screen_px"])
    print(f"{path_}: {len(run['block'])} frames, screen {run['screen_px']} px")
    print(report(run, score(run, models, run["screen_px"]), lag))


if __name__ == "__main__":
    main(sys.argv[1])
