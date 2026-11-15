import numpy as np

from gazemouse.filters import OneEuro

DT = 1 / 60


def run(f, points, t0=0.0):
    return np.array([f(np.asarray(p, float), t0 + i * DT) for i, p in enumerate(points)])


def test_one_euro_still_input_settles():
    out = run(OneEuro(), [(1000, 500)] * 60)
    assert np.allclose(out[-1], (1000, 500))


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(1)
    noisy = np.array([1000, 500]) + rng.uniform(-20, 20, (600, 2))
    out = run(OneEuro(), noisy)
    assert out[60:].std(axis=0).max() < 0.5 * noisy[60:].std(axis=0).min()


def test_one_euro_follows_jump():
    pts = [(1000, 500)] * 60 + [(1800, 500)] * 60
    out = run(OneEuro(), pts)
    after_0_3_s = out[60 + int(0.3 / DT)]
    assert after_0_3_s[0] - 1000 > 600


def test_reset_forgets_history():
    f = OneEuro()
    run(f, [(0, 0)] * 30)
    f.reset()
    assert np.allclose(f(np.array([900.0, 900.0]), 10.0), (900, 900))


# --- CalmCursor: rests while the gaze only wobbles, glides at a capped speed when it moves ---

from gazemouse.filters import DEAD_ZONE, MAX_SPEED, CalmCursor

CW = 2000


def test_calm_starts_where_the_gaze_is():
    assert np.allclose(CalmCursor(CW)(np.array([700.0, 300.0]), 0.0), (700, 300))


def test_calm_ignores_wobble_inside_the_dead_zone():
    rng = np.random.default_rng(2)
    r = 0.9 * DEAD_ZONE * CW / np.sqrt(2)
    wobble = np.array([1000, 500]) + rng.uniform(-r, r, (300, 2))
    out = run(CalmCursor(CW), np.vstack([[(1000, 500)], wobble]))
    assert np.ptp(out, axis=0).max() == 0


def test_calm_glides_to_a_new_point_no_faster_than_the_cap():
    pts = [(200, 500)] * 10 + [(1800, 500)] * 240
    out = run(CalmCursor(CW), pts)
    steps = np.abs(np.diff(out[:, 0]))
    assert steps.max() <= MAX_SPEED * CW * DT + 1e-6
    assert out[10 + 6][0] < 1000              # 0.1 s after the jump it is still on its way
    assert abs(out[-1][0] - 1800) < 0.005 * CW  # ...and it arrives
    assert np.all(np.diff(out[:, 0]) >= -1e-9)  # no overshoot


def test_calm_rests_again_after_arriving():
    f = CalmCursor(CW)
    out = run(f, [(200, 500)] * 10 + [(1800, 500)] * 240)
    still = run(f, [(1800 + 0.5 * DEAD_ZONE * CW, 500)] * 30, t0=250 * DT)
    assert np.allclose(still, out[-1])


def test_calm_reset_jumps_to_the_next_point():
    f = CalmCursor(CW)
    f(np.array([200.0, 500.0]), 0.0)
    f.reset()
    assert np.allclose(f(np.array([1800.0, 500.0]), 0.1), (1800, 500))
