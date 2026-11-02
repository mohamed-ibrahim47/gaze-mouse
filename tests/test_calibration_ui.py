import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from gazemouse.calibration_ui import COUNTDOWN_S, CalibrationWindow, result_text
from gazemouse.calibration import CalibrationResult

W, H = 1280, 800


def feat_at(target, rng):
    x, y = target
    f = np.zeros(8)
    f[0] = 0.42 + x / W * 0.15
    f[1] = -0.06 + y / H * 0.04
    f[2] = -0.03 + y / H * 0.01
    f[3] = 0.33 - y / H * 0.05
    f[:2] += rng.normal(0, 0.0003, 2)
    return f


def drive(win, feat, qapp, seconds=60):
    """Feed 30 fps features (the window's current dot -> feat) until finished, painting as we go."""
    done = []
    win.finished.connect(done.append)
    t = 100.0
    for i in range(int(seconds * 30)):
        win.on_features(feat(win.step) if win.step and win.step.target else None, t)
        t += 1 / 30
        if i % 30 == 0:
            assert not win.grab().isNull()
        qapp.processEvents()
        if win.showing_result:
            break
    QTest.qWait(30)
    qapp.processEvents()
    return done


def test_full_session_emits_a_good_result(qapp):
    rng = np.random.default_rng(0)
    win = CalibrationWindow(W, H, result_s=0.01)
    done = drive(win, lambda step: feat_at(step.target, rng), qapp)
    assert len(done) == 1 and done[0].ok and done[0].error_pct < 1
    assert "good" in win.text


def test_countdown_comes_first(qapp):
    win = CalibrationWindow(W, H, result_s=0.01)
    win.on_features(None, 10.0)
    win.on_features(None, 10.0 + COUNTDOWN_S - 0.1)
    assert win.step is None and "Look at each red dot" in win.text
    win.on_features(None, 10.0 + COUNTDOWN_S + 0.1)
    assert win.step.phase == "calib"


def test_no_face_emits_failure(qapp):
    win = CalibrationWindow(W, H, result_s=0.01)
    done = drive(win, lambda step: None, qapp)
    assert len(done) == 1 and not done[0].ok and "Couldn't see your eyes" in win.text


def test_esc_cancels(qapp):
    win = CalibrationWindow(W, H, result_s=0.01)
    done = []
    win.finished.connect(done.append)
    QTest.keyClick(win, Qt.Key_Escape)
    qapp.processEvents()
    assert len(done) == 1 and done[0].reason == "cancelled"


def test_features_after_the_end_are_ignored(qapp):
    rng = np.random.default_rng(1)
    win = CalibrationWindow(W, H, result_s=0.01)
    done = drive(win, lambda step: feat_at(step.target, rng), qapp)
    win.on_features(None, 999.0)
    qapp.processEvents()
    assert len(done) == 1


@pytest.mark.parametrize("pct,word", [(4.0, "good"), (6.0, "good"), (7.5, "usable but weak")])
def test_result_text_grades_accuracy(pct, word):
    r = CalibrationResult(True, None, None, pct, pct, 16, 5)
    assert f"{pct:.1f}%" in result_text(r) and word in result_text(r)
