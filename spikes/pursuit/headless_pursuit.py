"""Run the pursuit session offscreen on a still photo (no camera). Checks the flow, not accuracy."""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2, numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import pursuit, pursuit_spike, pipeline
pursuit.PATH_S, pursuit.LEAD_IN_S = 4.0, 0.5
pursuit.HOLD, pursuit.HOLD_SETTLE_S = ((0.5, 0.5, 2.0), (0.25, 0.3, 1.5), (0.75, 0.7, 1.5)), 0.3
pursuit_spike.SETTLE_S, pursuit_spike.SAMPLE_S = 0.1, 0.5
pursuit_spike.TEXTS = {k: (v[0], 1.0) for k, v in pursuit_spike.TEXTS.items()}
pipeline.WIDTH_FRAMES = 5
img = cv2.resize(cv2.imread(sys.argv[1]), (1280, 720))
class FakeCamera:
    times = [0.0] * 30
    def __init__(self): self.n = 0
    def read(self):
        self.n += 1
        return np.roll(img, int(40 * np.sin(self.n / 7)), axis=1), time.perf_counter()
    def info(self):
        from gazemouse.camera import CameraInfo; return CameraInfo(1280, 720, 30.0)
    def close(self): pass
qt = QApplication([])
s = pursuit_spike.Spike(1920, 1080, (34.4, 19.4))
s.thread.camera_factory = FakeCamera
s.start()
def check():
    if s.live or s.text == pursuit_spike.FAILED:
        s.grab().save(sys.argv[2]); QTimer.singleShot(1500, s.close)
    else: QTimer.singleShot(300, check)
check(); QTimer.singleShot(120000, qt.quit)
qt.exec()
print("live rings:", {k: None if v[2] is None else np.round(v[2]) for k, v in s.live.items()}, "| text:", s.text)
print("blocks recorded:", {int(b): int(sum(1 for r in s.rows if r[0] == b)) for b in (0, 1, 2, 3, 4, 5)})
