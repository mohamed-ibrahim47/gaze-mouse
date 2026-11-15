"""Background thread: camera + face model -> the latest observation for the GUI thread."""
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .camera import Camera
from .landmarks import MODEL_PATH, FaceTracker

READ_RETRY_S = 0.05
REOPEN_AFTER_S = 3.0  # this long without a frame: the camera is gone (unplugged, taken, sleep)
REOPEN_EVERY_S = 2.0


class TrackerThread(QThread):
    """Only the newest frame is ever waiting: `ready` fires, the GUI calls take(). If the GUI falls
    behind it skips frames instead of working through a growing queue of old ones."""

    ready = Signal()
    camera_info = Signal(object)  # CameraInfo
    failed = Signal(str)          # couldn't start at all

    def __init__(self, camera_index: int | None = None, model_path: Path = MODEL_PATH, camera_factory=None,
                 tracker_factory=None, reopen_after_s: float = REOPEN_AFTER_S,
                 reopen_every_s: float = REOPEN_EVERY_S):
        super().__init__()
        self.camera_factory = camera_factory or (lambda: Camera(camera_index))
        self.tracker_factory = tracker_factory or (lambda: FaceTracker(model_path).process)
        self.reopen_after_s, self.reopen_every_s = reopen_after_s, reopen_every_s
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._latest = None

    def stop(self) -> None:
        self._stopping.set()
        self.wait(3000)

    def take(self):
        """The newest (FaceObservation | None, t), once."""
        with self._lock:
            latest, self._latest = self._latest, None
        return latest

    def _publish(self, obs, t: float) -> None:
        with self._lock:
            nothing_waiting = self._latest is None
            self._latest = (obs, t)
        if nothing_waiting:
            self.ready.emit()

    def _reopen(self):
        while not self._stopping.wait(self.reopen_every_s):
            try:
                return self.camera_factory()
            except Exception:
                pass  # still busy or unplugged
        return None

    def run(self) -> None:
        try:
            camera = self.camera_factory()
            process = self.tracker_factory()
        except Exception as e:
            self.failed.emit(str(e))
            return
        try:
            self.camera_info.emit(camera.info())
            reported = False
            last_frame = time.perf_counter()
            while not self._stopping.is_set():
                frame = camera.read()
                if frame is None:
                    if time.perf_counter() - last_frame >= self.reopen_after_s:
                        camera.close()
                        camera = self._reopen()
                        if camera is None:
                            return
                        last_frame = time.perf_counter()
                    else:
                        self._stopping.wait(READ_RETRY_S)
                    continue
                last_frame = frame[1]
                try:
                    obs = process(*frame)
                except Exception as e:  # one bad frame must not end tracking
                    print("Face tracking error:", e, file=sys.stderr)
                    obs = None
                self._publish(obs, frame[1])
                if not reported and len(camera.times) >= 30:
                    self.camera_info.emit(camera.info())  # now with the measured fps
                    reported = True
        finally:
            if camera is not None:
                camera.close()
