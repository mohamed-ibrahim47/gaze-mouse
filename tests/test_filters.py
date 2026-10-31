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
