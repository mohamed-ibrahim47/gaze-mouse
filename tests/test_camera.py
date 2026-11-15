import numpy as np
import pytest

import cv2

from gazemouse.camera import Camera, CameraInfo, slow_or_small

FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeCap:
    """Stands in for cv2.VideoCapture. `reads_after_720p` is what read() does once 720p is requested."""

    def __init__(self, opens=True, reads=True, breaks_on_720p=False):
        self.opens, self.reads, self.breaks = opens, reads, breaks_on_720p
        self.asked_720p, self.released = False, False
        self.props = {}

    def isOpened(self):
        return self.opens

    def read(self):
        if not self.reads or (self.breaks and self.asked_720p):
            return False, None
        return True, FRAME

    def set(self, prop, value):
        self.props[prop] = value
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.asked_720p = True
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720, cv2.CAP_PROP_FPS: 30}[prop]

    def release(self):
        self.released = True


def opener(table):
    """(index, backend) -> list of caps handed out in turn; anything else fails to open."""
    calls = []

    def open_capture(index, backend):
        calls.append((index, backend))
        caps = table.get((index, backend))
        return caps.pop(0) if caps else FakeCap(opens=False)

    open_capture.calls = calls
    return open_capture


def test_first_working_backend_wins():
    good = FakeCap()
    op = opener({(0, cv2.CAP_MSMF): [good]})
    cam = Camera(open_capture=op)
    assert cam.read() is not None and op.calls == [(0, cv2.CAP_MSMF)]
    assert good.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280


def test_falls_back_to_next_backend_and_releases_failures():
    bad, good = FakeCap(reads=False), FakeCap()
    op = opener({(0, cv2.CAP_MSMF): [bad], (0, cv2.CAP_DSHOW): [good]})
    cam = Camera(open_capture=op)
    assert cam.cap is good and bad.released


def test_camera_that_breaks_on_720p_reopens_at_default():
    broken, fresh = FakeCap(breaks_on_720p=True), FakeCap()
    op = opener({(0, cv2.CAP_MSMF): [broken, fresh]})
    cam = Camera(open_capture=op)
    assert cam.cap is fresh and broken.released and not fresh.asked_720p


def test_explicit_index_only_tries_that_index():
    op = opener({(2, cv2.CAP_MSMF): [FakeCap()]})
    Camera(index=2, open_capture=op)
    assert {i for i, _ in op.calls} == {2}


def test_no_camera_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="webcam"):
        Camera(open_capture=opener({}))


def test_read_returns_frame_and_time():
    cam = Camera(open_capture=opener({(0, cv2.CAP_MSMF): [FakeCap()]}))
    frame, t = cam.read()
    assert frame.shape == (720, 1280, 3) and t > 0


def test_info_measures_fps_over_30_frames():
    cam = Camera(open_capture=opener({(0, cv2.CAP_MSMF): [FakeCap()]}))
    assert cam.info() == CameraInfo(1280, 720, 30)  # too early to measure: the driver's claim
    cam.times = [i / 50 for i in range(30)]
    assert cam.info().fps == pytest.approx(50, abs=0.1)


def test_failed_read_returns_none_and_close_releases():
    cap = FakeCap()
    cam = Camera(open_capture=opener({(0, cv2.CAP_MSMF): [cap]}))
    cap.reads = False
    assert cam.read() is None
    cam.close()
    assert cap.released


def test_slow_or_small():
    assert slow_or_small(CameraInfo(1280, 720, 30)) == []
    assert len(slow_or_small(CameraInfo(640, 480, 15))) == 2
