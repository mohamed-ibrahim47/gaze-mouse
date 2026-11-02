"""The whole loop without a camera: fake observations go in, fake mouse calls come out."""
import numpy as np
import pytest

from gazemouse.app import GazeApp, is_better, parse_args
from gazemouse.calibration import CalibrationResult
from gazemouse.camera import CameraInfo
from gazemouse.features import LEFT_EYE, RIGHT_EYE, FaceObservation
from gazemouse.gaze_model import GazeModel
from gazemouse.profiles import load_profile, save_profile
from gazemouse.snap import BackgroundTargets
from test_controller import FakeActuator

W, H = 1280, 800


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GAZEMOUSE_HOME", str(tmp_path))
    return tmp_path


class FakeTracker:
    def __init__(self):
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class NoTargets:
    def targets_near(self, point, radius):
        return []


def linear_model():
    """A model that reads the screen point straight back from iris_x / iris_y_corner."""
    T = np.random.default_rng(0).uniform([0, 0], [W, H], (100, 2))
    F = np.zeros((100, 8))
    F[:, 0] = T[:, 0] / W
    F[:, 1] = F[:, 2] = T[:, 1] / H
    F[:, 3] = 0.4  # what face() below measures
    m = GazeModel()
    m.fit(F, T)
    return m


def face(t, blink=0.0, point=(W / 2, H / 2)):
    """An observation whose eye geometry makes extract_features read iris_x, iris_y_corner = point."""
    fx, fy = point[0] / W, point[1] / H
    p = np.zeros((478, 2))
    for eye, sign, x0, along in ((RIGHT_EYE, -1, 300, 1 - fx), (LEFT_EYE, 1, 400, fx)):  # right eye mirrored
        inner, outer = np.array([x0, 200.0]), np.array([x0 + sign * 50, 200.0])
        p[eye["inner"]], p[eye["outer"]] = inner, outer
        p[eye["upper"]] = inner + [sign * 25, -10]
        p[eye["lower"]] = inner + [sign * 25, 10]
        p[eye["iris"]] = inner + [sign * 50 * along, 50 * fy]
    return FaceObservation(p, blink, blink, np.eye(4), t)


def make_app(qapp, **kw):
    return GazeApp(parse_args(kw.pop("argv", [])), W, H, tracker=FakeTracker(), actuator=FakeActuator(),
                   provider=NoTargets(), result_s=0.0, **kw)


def with_profile(meta=None):
    save_profile("default", linear_model(), {"screen": [W, H], **(meta or {})})


def good(pct=4.0):
    return CalibrationResult(True, None, linear_model(), pct / 100 * W, pct, 16, 5)


FAILED = CalibrationResult(False, "face_lost", None, None, None, 3, 0)


def test_saved_profile_skips_calibration(qapp):
    with_profile()
    app = make_app(qapp)
    app.start()
    assert app.calib is None and app.controller is not None and app.tracker.started
    app.quit()


def test_no_profile_calibrates(qapp):
    app = make_app(qapp)
    app.start()
    assert app.calib is not None and app.controller is None
    app.quit()


def test_calibrate_flag_forces_it(qapp):
    with_profile()
    app = make_app(qapp, argv=["--calibrate"])
    app.start()
    assert app.calib is not None
    app.quit()


def test_profile_from_another_screen_recalibrates(qapp):
    save_profile("default", linear_model(), {"screen": [1920, 1080]})
    app = make_app(qapp)
    app.start()
    assert app.calib is not None
    app.quit()


def test_observations_go_to_the_calibration_window(qapp):
    app = make_app(qapp)
    app.start()
    seen = []
    app.calib.on_features = lambda feat, t: seen.append((feat is not None, t))
    app.on_observed(face(1.0), 1.0)
    app.on_observed(None, 1.1)
    assert seen == [(True, 1.0), (False, 1.1)]
    app.quit()


def test_good_result_is_saved_and_control_starts(qapp):
    app = make_app(qapp)
    app.start()
    app.camera_info = CameraInfo(1280, 720, 30.0)
    app.on_calibrated(good(4.0))
    assert app.controller is not None and app.calib is None
    model, meta = load_profile("default")
    assert meta["error_pct"] == 4.0 and meta["screen"] == [W, H] and meta["camera"]["width"] == 1280
    assert "date" in meta
    app.quit()


def test_weak_or_failed_retries_twice_then_keeps_the_best(qapp):
    app = make_app(qapp)
    app.start()
    app.on_calibrated(good(9.0))
    assert app.calib is not None and app.controller is None
    app.on_calibrated(FAILED)
    assert app.calib is not None
    app.on_calibrated(good(7.0))
    assert app.calib is None and app.controller is not None
    assert load_profile("default")[1]["error_pct"] == 7.0
    assert "7.0%" in app.warning
    app.quit()


def test_all_attempts_failed_quits_with_an_error(qapp):
    app = make_app(qapp)
    app.start()
    for _ in range(3):
        app.on_calibrated(FAILED)
    assert app.exit_code == 1 and app.tracker.stopped
    assert app.controller is None


def test_cancel_with_no_model_quits(qapp):
    app = make_app(qapp)
    app.start()
    app.on_calibrated(CalibrationResult(False, "cancelled", None, None, None, 0, 0))
    assert app.exit_code == 0 and app.tracker.stopped


def test_recalibration_failure_keeps_the_old_model(qapp):
    with_profile({"error_pct": 5.0})
    app = make_app(qapp)
    app.start()
    old = app.model
    app.recalibrate()
    assert app.calib is not None
    app.on_calibrated(CalibrationResult(False, "cancelled", None, None, None, 0, 0))
    assert app.model is old and app.controller is not None and app.exit_code is None
    app.quit()


def test_gaze_moves_the_cursor(qapp):
    with_profile()
    app = make_app(qapp)
    app.start()
    for i in range(30):
        app.on_observed(face(i / 30, point=(300, 200)), i / 30)
    x, y = app.actuator.last_move()
    assert abs(x - 300) < 25 and abs(y - 200) < 25
    app.quit()


def test_long_blink_end_to_end_clicks(qapp):
    with_profile()
    app = make_app(qapp)
    app.start()
    t = 0.0
    for blink, seconds in ((0.0, 1.0), (0.9, 0.5), (0.0, 1.0)):
        for _ in range(int(seconds * 30)):
            app.on_observed(face(t, blink=blink, point=(300, 200)), t)
            t += 1 / 30
    assert app.actuator.named("click") == [("click", ("left",))]
    app.quit()


def test_lost_face_reaches_the_controller_and_resets_smoothing(qapp):
    with_profile()
    app = make_app(qapp)
    app.start()
    for i in range(10):
        app.on_observed(face(i / 30, point=(300, 200)), i / 30)
    app.on_observed(None, 0.5)
    assert app.state.gaze is None
    app.on_observed(face(0.6, point=(900, 600)), 0.6)
    assert app.state.gaze == pytest.approx((900, 600), abs=5)  # no slow glide from the old point
    app.quit()


def test_recalibrate_lets_go_of_a_held_button(qapp):
    with_profile()
    app = make_app(qapp)
    app.start()
    from gazemouse.controller import Mode
    app.controller.mode = Mode.DRAGGING
    app.recalibrate()
    assert app.actuator.calls[-1] == ("button_up", ("left",))
    app.quit()


def test_quit_stops_everything_and_releases(qapp):
    with_profile()
    app = make_app(qapp)
    app.start()
    from gazemouse.controller import Mode
    app.controller.mode = Mode.DRAGGING
    bg = app.targets
    assert isinstance(bg, BackgroundTargets)
    app.quit()
    assert app.tracker.stopped and not bg.thread.is_alive()
    assert app.actuator.calls[-1] == ("button_up", ("left",))
    app.quit()  # twice is harmless


def test_camera_failure_quits_with_the_message(qapp):
    app = make_app(qapp)
    app.start()
    app.on_tracker_failed("Could not open the webcam.")
    assert app.exit_code == 1 and "webcam" in app.warning


def test_is_better():
    assert is_better(good(5.0), None) and is_better(good(5.0), good(6.0)) and not is_better(good(7.0), good(6.0))
    assert not is_better(FAILED, None) and not is_better(FAILED, good(9.0))
