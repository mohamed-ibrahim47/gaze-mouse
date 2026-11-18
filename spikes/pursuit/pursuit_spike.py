"""Throwaway spike 3: calibrate by following a gliding dot, and learn head turning from it.

About 3 minutes in one sitting:
  16 dots (head still)  ->  gliding dot, head still  ->  gliding dot, head turns with it
  ->  3 still dots held while the head nods and turns  ->  9 test dots, head still
  ->  9 test dots, head free.
Every model learns from this sitting's calibration parts and is scored on the same test dots. Then
two gaze points are drawn live: green = today's dot calibration, blue = a new model (number keys switch
which one). Esc quits at any time.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from gazemouse.actions import primary_screen_size, set_dpi_awareness
from gazemouse.calibration import CALIB_X, CALIB_Y
from gazemouse.filters import OneEuro
from gazemouse.landmarks import MODEL_PATH
from gazemouse.tracker import TrackerThread

import pursuit
from pipeline import WetTracker

TEST = (0.2, 0.5, 0.8)
SETTLE_S, SAMPLE_S = 0.8, 1.0
FRAME_MS = 16  # repaint rate while the dot glides
DEFAULT_SCREEN_CM = (34.4, 19.4)
LOW_FPS_NOTE = ("Warning: under 20 frames/s, so every model is slow and shaky. The webcam slows down in a dim "
                "room: add light in front of you, plug the laptop in, and try again.\n")
BACKGROUND = QColor(24, 26, 32)
RED, ORANGE, WHITE = QColor(235, 60, 50), QColor(255, 165, 40), QColor(235, 235, 235)
GREEN, BLUE = QColor(70, 220, 110), QColor(80, 160, 255)
TEXTS = {
    "intro": ("Part 1 of 6. Sit as you normally do and keep your head still.\n"
              "Look at each red dot until it moves.", 6.0),
    "glide_still": ("Part 2 of 6. A dot will glide around the screen.\n"
                    "Keep your head still and follow it with your eyes only.", 6.0),
    "glide_head": ("Part 3 of 6. Follow the gliding dot again,\n"
                   "but this time let your head turn with it, the way you naturally would.", 6.0),
    "hold": ("Part 4 of 6. A red dot will stand still. Keep your eyes on it\n"
             "while you slowly move your head: nod up and down, turn left and right.\n"
             "The dot jumps to a new place twice; keep doing the same there.", 8.0),
    "test_still": ("Part 5 of 6. Test: keep your head still and look at each orange dot.", 5.0),
    "test_free": ("Part 6 of 6. Test: look at each orange dot and move your head freely.\n"
                  "Turn toward the dots, tilt, lean, as you like.", 6.0),
}
BUSY = "Working it out..."
FAILED = "Couldn't see your eyes well enough. Add light in front of you and try again."


def run_dir() -> Path:
    base = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    d = base / "runs"
    d.mkdir(exist_ok=True)
    return d


class Spike(QWidget):
    def __init__(self, w: int, h: int, screen_cm, camera_index=None):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.w, self.h, self.screen_cm = w, h, screen_cm
        rng = np.random.default_rng(1)
        grid = lambda xs, ys: [(round(x * w), round(y * h)) for y in ys for x in xs]  # noqa: E731
        test = grid(TEST, TEST)
        self.plan = [("text", "intro"), ("dots", (0, grid(CALIB_X, CALIB_Y))),
                     ("text", "glide_still"), ("glide", 3),
                     ("text", "glide_head"), ("glide", 4),
                     ("text", "hold"), ("hold", 5),
                     ("text", "test_still"), ("dots", (1, [test[i] for i in rng.permutation(9)])),
                     ("text", "test_free"), ("dots", (2, [test[i] for i in rng.permutation(9)]))]
        self.stage, self.stage_t0 = 0, None
        self.rows = []       # (block, dot, target, t_rel, sample)
        self.frames = []
        self.dot = None      # (x, y, progress, colour) to paint
        self.glide = None    # phase of the gliding dot while one is on screen
        self.text, self.small = TEXTS["intro"][0], ""
        self.models, self.names, self.blue = None, [], 0
        self.live = {}       # name -> [colour, filter, point]
        self.report = ""
        self.tracker_obj = None
        self.thread = TrackerThread(camera_index, tracker_factory=self._make_tracker)
        self.thread.ready.connect(self.on_frame)
        self.thread.failed.connect(self.on_failed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def _make_tracker(self):
        self.tracker_obj = WetTracker(MODEL_PATH)
        return self.tracker_obj.process

    def start(self) -> None:
        self.setGeometry(QGuiApplication.primaryScreen().geometry())
        self.showFullScreen()
        self.activateWindow()
        self.thread.start()

    def on_failed(self, why: str) -> None:
        self.text = why
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif self.models and Qt.Key_1 <= key <= Qt.Key_9 and key - Qt.Key_1 < len(self.names):
            self.blue = key - Qt.Key_1
            self._set_live()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.thread.stop()
        QApplication.quit()

    def on_frame(self) -> None:
        latest = self.thread.take()
        if latest is None:
            return
        sample, t = latest
        self.frames.append(t)
        if self.models is not None:
            self._live(sample, t)
        elif self.stage < len(self.plan):
            self._advance(sample, t)
        self.update()

    def _advance(self, sample, t: float) -> None:
        if self.stage_t0 is None:
            self.stage_t0 = t
        kind, payload = self.plan[self.stage]
        elapsed = t - self.stage_t0
        if kind == "text":
            text, seconds = TEXTS[payload]
            ready = self.tracker_obj is not None and self.tracker_obj.face_width_cm is not None
            self.text, self.dot = f"{text}\n\n{max(seconds - elapsed, 0):.0f}", None
            if elapsed >= seconds and ready:
                self._next(t)
            return
        usable = sample is not None and sample.usable
        if kind == "hold":
            start = 0.0
            for i, (fx, fy, seconds) in enumerate(pursuit.HOLD):
                if elapsed < start + seconds:
                    within = elapsed - start
                    pos = (round(fx * self.w), round(fy * self.h))
                    self.text, self.dot = "", (*pos, min(within / 1.0, 1.0), RED)
                    if usable:
                        self.rows.append((payload, i, pos, within, sample))
                    return
                start += seconds
            self._next(t)
            return
        if kind == "glide":
            if elapsed >= pursuit.PATH_S:
                self._next(t)
                return
            if self.glide is None:
                self.glide = pursuit.PHASES[payload]
                self.timer.start(FRAME_MS)  # smooth motion between camera frames
            self.text = ""
            if usable:
                target = pursuit.path(elapsed, self.glide)[0] * (self.w, self.h)
                self.rows.append((payload, -1, tuple(target), elapsed, sample))
            return
        block, dots = payload
        per = SETTLE_S + SAMPLE_S
        i = int(elapsed // per)
        if i >= len(dots):
            self._next(t)
            return
        within = elapsed - i * per
        self.text, self.dot = "", (*dots[i], within / per, RED if block == 0 else ORANGE)
        if within > SETTLE_S and usable:
            self.rows.append((block, i, dots[i], 0.0, sample))

    def _next(self, t: float) -> None:
        self.stage, self.stage_t0, self.dot, self.glide = self.stage + 1, t, None, None
        self.timer.stop()
        if self.stage == len(self.plan):
            self.text = BUSY
            QTimer.singleShot(50, self._finish)

    def _finish(self) -> None:
        rows = self.rows
        run = {
            "block": np.array([r[0] for r in rows]), "dot": np.array([r[1] for r in rows]),
            "target": np.array([r[2] for r in rows], float), "t_rel": np.array([r[3] for r in rows]),
            "feat": np.array([r[4].feat for r in rows]), "origin": np.array([r[4].origin for r in rows]),
            "rt": np.array([r[4].rt for r in rows]), "lm": np.array([r[4].lm for r in rows]),
            "screen_px": np.array([self.w, self.h]), "screen_cm": np.array(self.screen_cm),
            "face_width_cm": np.array(self.tracker_obj.face_width_cm),
        }
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            np.savez_compressed(run_dir() / f"pursuit-{stamp}.npz", **run)
            self.models, lag = pursuit.fit(run, (self.w, self.h))
            scores = pursuit.score(run, self.models, (self.w, self.h))
        except Exception as e:  # too few good frames to fit
            print("Could not score the run:", repr(e))
            self.models, self.text = None, FAILED
            self.update()
            return
        fps = (len(self.frames) - 1) / (self.frames[-1] - self.frames[0])
        self.report = (pursuit.report(run, scores, lag) +
                       f"Speed: {fps:.0f} frames/s. Screen {self.w}x{self.h} px. {len(rows)} frames kept.\n" +
                       (LOW_FPS_NOTE if fps < 20 else ""))
        (run_dir() / f"pursuit-result-{stamp}.txt").write_text(self.report, encoding="utf-8")
        print(self.report)
        self.names = [n for n in self.models if n != pursuit.CURRENT]
        self.blue = self.names.index(pursuit.CANDIDATE) if pursuit.CANDIDATE in self.names else 0
        self.text = ""
        self._set_live()

    def _set_live(self) -> None:
        blue = self.names[self.blue]
        self.live = {pursuit.CURRENT: [GREEN, OneEuro(), None], blue: [BLUE, OneEuro(), None]}
        keys = "   ".join(f"{i + 1} = {n}" for i, n in enumerate(self.names))
        self.small = (self.report + f"\nLive now: green ring = {pursuit.CURRENT}, blue ring = {blue}.\n"
                      f"Switch blue: {keys}\n"
                      "Look around and move your head; see which ring stays on what you look at. Esc quits.")
        self.update()

    def _live(self, sample, t: float) -> None:
        if sample is None or not sample.usable:
            return
        for name, entry in self.live.items():
            p = self.models[name](sample.feat[None], sample.rt[None], sample.origin[None])[0]
            entry[2] = entry[1](p, t)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        dpr = self.devicePixelRatioF()
        dot = self.dot
        if self.glide is not None and self.stage_t0 is not None:
            x, y = pursuit.path(time.perf_counter() - self.stage_t0, self.glide)[0] * (self.w, self.h)
            dot = (x, y, 0.6, RED)
        if dot is not None:
            x, y, progress, colour = dot
            r = 8 + 26 * (1 - progress)
            p.setPen(Qt.NoPen)
            p.setBrush(colour)
            p.drawEllipse(QRectF(x / dpr - r, y / dpr - r, 2 * r, 2 * r))
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QRectF(x / dpr - 3, y / dpr - 3, 6, 6))
        if self.text:
            font = QFont()
            font.setPixelSize(30)
            p.setFont(font)
            p.setPen(WHITE)
            p.drawText(self.rect().adjusted(60, 0, -60, 0), Qt.AlignCenter | Qt.TextWordWrap, self.text)
        if self.small:
            mono = QFont("Consolas")
            mono.setPixelSize(17)
            p.setFont(mono)
            p.setPen(WHITE)
            p.drawText(self.rect().adjusted(40, 30, -40, -30), Qt.AlignLeft | Qt.AlignTop, self.small)
        for colour, _, point in self.live.values():
            if point is not None:
                p.setPen(QPen(colour, 4))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QRectF(point[0] / dpr - 22, point[1] / dpr - 22, 44, 44))
        p.end()


def main() -> int:
    if not MODEL_PATH.exists():
        print("Face model missing. Run scripts/fetch_model.py in gaze-mouse.")
        return 1
    set_dpi_awareness()
    w, h = primary_screen_size()
    qt = QApplication(sys.argv[:1])
    mm = QGuiApplication.primaryScreen().physicalSize()
    cm = (mm.width() / 10, mm.height() / 10) if mm.width() > 100 and mm.height() > 50 else DEFAULT_SCREEN_CM
    camera = int(sys.argv[1]) if len(sys.argv) > 1 else None
    spike = Spike(w, h, cm, camera)
    spike.start()
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
