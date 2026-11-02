import numpy as np

from gazemouse.calibration import CalibrationSession

DT = 1 / 30
W, H = 2560, 1600


def run(session, feat_for_step, t0=0.0):
    """Drive a session at 30 fps until done. The feature callback sees the previous
    step, like a real frame that was captured while the last dot was on screen."""
    session.start(t0)
    t, step = t0, session.update(t0, None)
    while step.phase != "done":
        t += DT
        step = session.update(t, feat_for_step(step))
    return session.result()


def linear_features(target, rng, noise=0.0005):
    """Features whose iris_x / iris_y_corner map linearly onto a W x H screen."""
    x, y = target
    f = np.zeros(8)
    f[0] = 0.42 + x / W * 0.15
    f[1] = -0.06 + y / H * 0.04
    f[2] = -0.03 + y / H * 0.01
    f[3] = 0.33 - y / H * 0.05
    f[:2] += rng.normal(0, noise, 2) * [1, 0.25]
    f[4:] = rng.normal(0, 1, 4)
    return f


def clean(seed=0):
    rng = np.random.default_rng(seed)
    return lambda step: linear_features(step.target, rng)


def test_clean_data_calibrates_accurately():
    r = run(CalibrationSession(W, H), clean())
    assert r.ok and r.reason is None
    assert r.calib_dots == 16 and r.valid_dots == 5
    assert r.error_pct < 1.0
    assert r.error_pct == r.mean_error_px / W * 100


def test_settle_period_ignored():
    # Garbage while the eyes are still travelling to the dot (first 0.8 s), clean after.
    rng = np.random.default_rng(1)
    settle_fraction = 0.8 / 1.8

    def garbage_then_clean(step):
        if step.progress <= settle_fraction:
            return rng.uniform(-5, 5, 8)
        return linear_features(step.target, rng)

    r = run(CalibrationSession(W, H), garbage_then_clean)
    assert r.ok and r.error_pct < 1.0


def test_face_lost_reports_failure():  # Review Focus 4
    r = run(CalibrationSession(W, H), lambda step: None)
    assert not r.ok and r.reason == "face_lost" and r.model is None


def test_mostly_lost_face_fails():
    # Face visible on only 8 of the 16 calibration dots.
    s = CalibrationSession(W, H)
    rng = np.random.default_rng(2)
    seen = []

    def half_missing(step):
        if step.phase == "calib" and step.target not in seen:
            seen.append(step.target)
        if step.phase == "calib" and seen.index(step.target) % 2 == 1:
            return None
        return linear_features(step.target, rng)

    r = run(s, half_missing)
    assert not r.ok and r.reason == "face_lost" and r.calib_dots == 8


def test_dot_positions_scale_with_screen():
    s = CalibrationSession(1366, 768)
    s.start(0.0)
    assert s.update(0.0, None).target == (round(0.06 * 1366), round(0.08 * 768))


def test_phases_in_order_with_progress():
    s = CalibrationSession(W, H)
    s.start(0.0)
    phases, t, step = [], 0.0, s.update(0.0, None)
    while step.phase != "done":
        assert 0.0 <= step.progress <= 1.0
        if not phases or phases[-1][0] != step.phase or phases[-1][1] != step.target:
            phases.append((step.phase, step.target))
        t += DT
        step = s.update(t, None)
    assert [p for p, _ in phases].count("calib") == 16
    assert [p for p, _ in phases].count("valid") == 5
    assert phases[16][1] == (round(0.22 * W), round(0.25 * H))
    assert step.target is None
    assert abs(t - 21 * 1.8) < 0.1  # each dot is settle + sample long
