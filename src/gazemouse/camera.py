"""Open the webcam, whichever backend and index it answers on."""
import os
import time
from dataclasses import dataclass

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # else MSMF takes ~20 s to open

import cv2  # noqa: E402
import numpy as np  # noqa: E402

BACKENDS = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
INDEXES = (0, 1, 2)
WANT_W, WANT_H = 1280, 720
MIN_FPS = 25
FPS_FRAMES = 30


@dataclass
class CameraInfo:
    width: int
    height: int
    fps: float


def slow_or_small(info: CameraInfo) -> list[str]:
    """Why this camera may track badly; empty if it's fine."""
    problems = []
    if info.width < WANT_W or info.height < WANT_H:
        problems.append(f"camera is {info.width}x{info.height}, want at least {WANT_W}x{WANT_H}")
    if info.fps < MIN_FPS:
        problems.append(f"camera runs at {info.fps:.0f} fps, want at least {MIN_FPS}")
    return problems


def _reads(cap) -> bool:
    try:
        return bool(cap.read()[0])
    except cv2.error:
        return False


class Camera:
    def __init__(self, index: int | None = None, open_capture=cv2.VideoCapture):
        self.open_capture = open_capture
        self.times = []
        self.cap = self._open(INDEXES if index is None else (index,))

    def _open(self, indexes):
        for backend in BACKENDS:
            for index in indexes:
                cap = self.open_capture(index, backend)
                if cap.isOpened() and _reads(cap):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # raw 720p is ~10 fps on USB 2
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WANT_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WANT_H)
                    if _reads(cap):
                        return cap
                    cap.release()  # some cameras (Remote Desktop's) break on 720p: default size then
                    cap = self.open_capture(index, backend)
                    if cap.isOpened() and _reads(cap):
                        return cap
                cap.release()
        raise RuntimeError("Could not open the webcam. Is another app using it, or is camera privacy off?")

    def read(self) -> tuple[np.ndarray, float] | None:
        try:
            ok, frame = self.cap.read()
        except cv2.error:
            return None
        if not ok:
            return None
        t = time.perf_counter()
        if len(self.times) < FPS_FRAMES:
            self.times.append(t)
        return frame, t

    def info(self) -> CameraInfo:
        """fps is measured once 30 frames have been read; until then it's the driver's claim."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if len(self.times) >= FPS_FRAMES:
            fps = (FPS_FRAMES - 1) / (self.times[FPS_FRAMES - 1] - self.times[0])
        return CameraInfo(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps)

    def close(self) -> None:
        self.cap.release()
