"""The full-screen calibration window: countdown, dots, result."""
import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from .calibration import CalibrationResult, CalibrationSession, CalibStep
from .overlay import to_logical

COUNTDOWN_S = 3.0
RESULT_S = 5.0
GOOD_PCT = 6.0
INTRO = "Look at each red dot until it moves"
BACKGROUND = QColor(24, 26, 32)
RED, ORANGE = QColor(235, 60, 50), QColor(255, 165, 40)
DOT_MIN, DOT_MAX = 8, 34  # logical px radius


def result_text(r: CalibrationResult) -> str:
    if not r.ok:
        return "Couldn't see your eyes well enough. Add light in front of you and try again."
    grade = "good" if r.error_pct <= GOOD_PCT else "usable but weak"
    return f"Accuracy: {r.error_pct:.1f}% of screen width ({grade})"


class CalibrationWindow(QWidget):
    finished = Signal(object)  # CalibrationResult

    def __init__(self, screen_w: int, screen_h: int, result_s: float = RESULT_S):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.session = CalibrationSession(screen_w, screen_h)
        self.result_s = result_s
        self.t0 = None
        self.step: CalibStep | None = None
        self.text = INTRO
        self.showing_result = False
        self.ended = False
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def show_full_screen(self) -> None:
        self.setGeometry(QGuiApplication.primaryScreen().geometry())
        self.showFullScreen()
        self.activateWindow()
        self.setFocus()

    def on_features(self, feat: np.ndarray | None, t: float) -> None:
        if self.showing_result or self.ended:
            return
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 < COUNTDOWN_S:
            self.update()
            return
        if self.step is None:
            self.session.start(t)
            self.text = ""
        self.step = self.session.update(t, feat)
        if self.step.phase == "done":
            result = self.session.result()
            self.text, self.showing_result = result_text(result), True
            QTimer.singleShot(int(self.result_s * 1000), lambda: self._end(result))
        self.update()

    def _end(self, result) -> None:
        if not self.ended:
            self.ended = True
            self.close()
            self.finished.emit(result)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:  # for a helper at the keyboard
            self._end(CalibrationResult(False, "cancelled", None, None, None, 0, 0))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        if self.step is not None and self.step.target is not None and not self.showing_result:
            x, y = to_logical(self.step.target, self.devicePixelRatioF())
            r = DOT_MIN + (DOT_MAX - DOT_MIN) * (1 - self.step.progress)
            p.setPen(Qt.NoPen)
            p.setBrush(RED if self.step.phase == "calib" else ORANGE)
            p.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QRectF(x - 3, y - 3, 6, 6))
        if self.text:
            font = QFont()
            font.setPixelSize(30)
            p.setFont(font)
            p.setPen(QColor(235, 235, 235))
            p.drawText(self.rect().adjusted(60, 0, -60, 0), Qt.AlignCenter | Qt.TextWordWrap, self.text)
        p.end()
