"""Background thread: camera + face model -> one observation per frame."""
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .camera import Camera
from .landmarks import MODEL_PATH, FaceTracker

READ_RETRY_S = 0.05


class TrackerThread(QThread):
    observed = Signal(object, float)  # FaceObservation | None, t
    camera_info = Signal(object)      # CameraInfo
    failed = Signal(str)

    def __init__(self, camera_index: int | None = None, model_path: Path = MODEL_PATH):
        super().__init__()
        self.camera_index, self.model_path = camera_index, model_path

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(3000)

    def run(self) -> None:
        try:
            camera = Camera(self.camera_index)
            tracker = FaceTracker(self.model_path)
        except Exception as e:
            self.failed.emit(str(e))
            return
        try:
            self.camera_info.emit(camera.info())
            reported = False
            while not self.isInterruptionRequested():
                frame = camera.read()
                if frame is None:  # a dropped frame, or the camera was unplugged
                    self.observed.emit(None, time.perf_counter())
                    self.msleep(int(READ_RETRY_S * 1000))
                    continue
                self.observed.emit(tracker.process(*frame), frame[1])
                if not reported and len(camera.times) >= 30:
                    self.camera_info.emit(camera.info())  # now with the measured fps
                    reported = True
        finally:
            camera.close()
