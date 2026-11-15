"""TrackerThread with fake camera and face model, delivered through real queued signals."""
import threading
import time

import numpy as np
from PySide6.QtCore import QCoreApplication

from gazemouse.tracker import TrackerThread

FRAME = np.zeros((4, 4, 3), np.uint8)


class FakeCamera:
    def __init__(self, reads):
        self.reads, self.times, self.closed = reads, [], False

    def read(self):
        return self.reads.pop(0)() if self.reads else (time.sleep(0.005), None)[1]

    def info(self):
        return "info"

    def close(self):
        self.closed = True


def frame():
    return FRAME, time.perf_counter()


def pump(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        QCoreApplication.processEvents()
        time.sleep(0.002)
    return cond()


def make(cameras, process=lambda f, t: "obs", **kw):
    th = TrackerThread(camera_factory=lambda: cameras.pop(0), tracker_factory=lambda: process,
                       reopen_after_s=0.05, reopen_every_s=0.01, **kw)
    got = []
    th.ready.connect(lambda: got.append(th.take()))
    failed = []
    th.failed.connect(failed.append)
    return th, got, failed


def test_frames_reach_the_gui_thread(qapp):
    th, got, _ = make([FakeCamera([frame, frame])])
    th.start()
    try:
        assert pump(lambda: len([g for g in got if g]) >= 1)
        assert got[0][0] == "obs"
    finally:
        th.stop()


def test_a_slow_gui_only_ever_gets_the_latest_frame(qapp):  # review 6
    th = TrackerThread(camera_factory=lambda: None, tracker_factory=lambda: None)
    fired = []
    th.ready.connect(lambda: fired.append(1), type=__import__("PySide6.QtCore").QtCore.Qt.DirectConnection)
    for i in range(5):
        th._publish("obs", float(i))
    assert th.take() == ("obs", 4.0) and th.take() is None and len(fired) == 1


def test_startup_failure_is_reported(qapp):
    def no_camera():
        raise RuntimeError("Could not open the webcam.")

    th = TrackerThread(camera_factory=no_camera, tracker_factory=lambda: None)
    failed = []
    th.failed.connect(failed.append)
    th.start()
    assert pump(lambda: failed == ["Could not open the webcam."])
    th.stop()


def test_a_crash_in_the_face_model_does_not_kill_the_thread(qapp):  # review 1
    calls = []

    def flaky(f, t):
        calls.append(t)
        if len(calls) == 1:
            raise IndexError("empty blendshapes")
        return "obs"

    th, got, failed = make([FakeCamera([frame, frame, frame])], process=flaky)
    th.start()
    try:
        assert pump(lambda: any(g and g[0] == "obs" for g in got))
        assert failed == [] and th.isRunning()
    finally:
        th.stop()


def test_a_camera_that_stops_is_reopened(qapp):  # review 2
    dead, fresh = FakeCamera([frame]), FakeCamera([frame, frame, frame])
    th, got, _ = make([dead, fresh])
    th.start()
    try:
        assert pump(lambda: dead.closed and not fresh.reads)
    finally:
        th.stop()


def test_reopen_keeps_trying_until_the_camera_is_back(qapp):  # review 2
    fresh = FakeCamera([frame, frame])
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            return FakeCamera([])
        if len(attempts) < 4:
            raise RuntimeError("busy")
        return fresh

    th = TrackerThread(camera_factory=factory, tracker_factory=lambda: (lambda f, t: "obs"),
                       reopen_after_s=0.02, reopen_every_s=0.01)
    failed = []
    th.failed.connect(failed.append)
    th.start()
    try:
        assert pump(lambda: not fresh.reads) and failed == []
    finally:
        th.stop()


def test_stop_ends_the_thread_and_closes_the_camera(qapp):
    cam = FakeCamera([])
    th, _, _ = make([cam])
    th.start()
    time.sleep(0.02)
    th.stop()
    assert not th.isRunning() and cam.closed
