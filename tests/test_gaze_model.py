import json

import numpy as np
import pytest

from gazemouse.gaze_model import GazeModel


def synthetic(n=300, seed=0):
    """Features whose iris_x / iris_y_corner map linearly onto a 2560x1600 screen."""
    rng = np.random.default_rng(seed)
    T = rng.uniform([0, 0], [2560, 1600], (n, 2))
    F = np.zeros((n, 8))
    F[:, 0] = 0.42 + T[:, 0] / 2560 * 0.15
    F[:, 1] = -0.06 + T[:, 1] / 1600 * 0.04
    F[:, 2] = -0.03 + T[:, 1] / 1600 * 0.01
    F[:, 3] = 0.33 - T[:, 1] / 1600 * 0.05
    F[:, :2] += rng.normal(0, 0.001, (n, 2)) * [1, 0.25]  # ~17 px irreducible x noise
    F[:, 4:] = rng.normal(0, 1, (n, 4))       # head values: noise the model must ignore
    return F, T


def test_recovers_linear_mapping():
    F, T = synthetic()
    m = GazeModel()
    m.fit(F, T)
    err = np.linalg.norm(np.array([m.predict(f) for f in F]) - T, axis=1).mean()
    assert err < 25


def test_roundtrip_dict_same_predictions():
    F, T = synthetic()
    m = GazeModel()
    m.fit(F, T)
    m2 = GazeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(F[0]), m.predict(F[0]))


def test_too_few_samples():
    F, T = synthetic()
    with pytest.raises(ValueError):
        GazeModel().fit(F[:5], T[:5])


def test_head_columns_are_ignored():
    # Spike: raw head pose as inputs made accuracy worse, so the model must not use them.
    F, T = synthetic()
    m = GazeModel()
    m.fit(F, T)
    moved = F[0].copy()
    moved[4:] += 10.0
    assert np.allclose(m.predict(moved), m.predict(F[0]))


def test_predict_returns_two_floats():
    F, T = synthetic()
    m = GazeModel()
    m.fit(F, T)
    p = m.predict(F[0])
    assert p.shape == (2,) and p.dtype == np.float64
