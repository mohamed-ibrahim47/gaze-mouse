import numpy as np
import pytest

from gazemouse.features import FEATURES, FaceObservation, extract_features

# Right eye sits on the image-left, left eye on the image-right (camera is not mirrored).
R_OUT, R_IN, R_UP, R_DN, R_IRIS = 33, 133, 159, 145, 468
L_IN, L_OUT, L_UP, L_DN, L_IRIS = 362, 263, 386, 374, 473


def make_obs(iris_shift=(0.0, 0.0), lid_gap=20.0):
    p = np.zeros((478, 2))
    p[R_OUT], p[R_IN] = (100, 200), (160, 200)
    p[L_IN], p[L_OUT] = (240, 200), (300, 200)
    half = lid_gap / 2
    p[R_UP], p[R_DN] = (130, 200 - half), (130, 200 + half)
    p[L_UP], p[L_DN] = (270, 200 - half), (270, 200 + half)
    dx, dy = iris_shift
    p[R_IRIS] = (130 + dx, 200 + dy)
    p[L_IRIS] = (270 + dx, 200 + dy)
    return FaceObservation(points=p, blink_left=0.0, blink_right=0.0, head=np.eye(4), t=0.0)


def test_feature_names_and_order():
    assert FEATURES == ("iris_x", "iris_y_corner", "iris_y_lid", "openness",
                        "head_fx", "head_fy", "head_tx", "head_ty")


def test_centered_iris_gives_half():
    f = extract_features(make_obs())
    assert f.shape == (8,)
    assert f[0] == pytest.approx(0.5, abs=1e-6)


def test_both_eyes_agree_on_horizontal():
    # Looking toward image-right: left eye's x grows, right eye's x shrinks; the flip makes them agree.
    f = extract_features(make_obs(iris_shift=(6, 0)))
    assert f[0] == pytest.approx(0.5 + 6 / 60, abs=1e-6)


def test_vertical_sign_same_for_both_eyes():
    # Regression: the 2026 spike averaged opposite-signed vertical values to ~0.
    f = extract_features(make_obs(iris_shift=(0, 4)))
    assert f[1] == pytest.approx(4 / 60, abs=1e-6)
    assert f[2] == pytest.approx(4 / 60, abs=1e-6)  # lids centred on the corner line here


def test_openness_is_lid_gap_over_eye_width():
    assert extract_features(make_obs(lid_gap=18))[3] == pytest.approx(18 / 60, abs=1e-6)


def test_closed_eyes_return_none():
    assert extract_features(make_obs(lid_gap=2)) is None


def test_head_values_come_from_matrix():
    obs = make_obs()
    obs.head[0, 2], obs.head[1, 2], obs.head[0, 3], obs.head[1, 3] = 0.1, -0.2, 5.0, -3.0
    assert np.allclose(extract_features(obs)[4:], [0.1, -0.2, 5.0, -3.0])
