"""The click-through layer over the screen: gaze ring, snapped target, edge strip, messages.

Everything arrives in physical pixels and is converted to Qt logical pixels only here.
"""
import ctypes
from dataclasses import replace

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .controller import Mode, OverlayState
from .geometry import Rect
from .strip import StripButton, StripLayout

WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x8, 0x20, 0x80000, 0x8000000
EX_STYLE_FLAGS = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE
GWL_EXSTYLE = -20

RING_RADIUS, RING_WIDTH = 18, 3  # logical px
BLUE = QColor(40, 140, 255, 170)
AMBER = QColor(255, 180, 40, 230)
PANEL_FILL = QColor(20, 24, 32, 205)
HOT_FILL = QColor(40, 140, 255, 235)
TEXT = QColor(255, 255, 255)

BUTTON_LABELS = {StripButton.SCROLL_UP: "▲ Scroll up", StripButton.SCROLL_DOWN: "▼ Scroll down",
                 StripButton.RIGHT_CLICK: "Right click", StripButton.SELECT: "Select / drag",
                 StripButton.KEYBOARD: "Keyboard", StripButton.PAUSE: "Pause"}
MODE_BADGES = {Mode.RIGHT_ONCE: "Right click next", Mode.SELECT_START: "Select: blink at start",
               Mode.DRAGGING: "Dragging", Mode.SCROLLING: "Scrolling"}
PAUSED_TEXT = "Paused — close eyes 2 s to resume"


def to_logical(pt: tuple[float, float], dpr: float) -> tuple[float, float]:
    return (pt[0] / dpr, pt[1] / dpr)


def rect_to_logical(r: Rect, dpr: float) -> Rect:
    return Rect(r.x / dpr, r.y / dpr, r.w / dpr, r.h / dpr)


def _q(r: Rect) -> QRectF:
    return QRectF(r.x, r.y, r.w, r.h)


class Overlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.state: OverlayState | None = None
        self.layout_: StripLayout | None = None
        self.ring_visible = True

    def cover_primary(self) -> None:
        self.setGeometry(QGuiApplication.primaryScreen().geometry())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if QGuiApplication.platformName() == "windows":
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
            user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | EX_STYLE_FLAGS)

    def set_ring_visible(self, visible: bool) -> None:
        self.ring_visible = visible
        self.update()

    def show_state(self, state: OverlayState, layout: StripLayout) -> None:
        old, self.state, self.layout_ = self.state, state, layout
        # Repainting a full-screen layered window every frame is expensive: when only the
        # gaze moved, repaint just the ring and its badge.
        if old is not None and replace(old, gaze=state.gaze) == state:
            for gaze in (old.gaze, state.gaze):
                if gaze is not None:
                    x, y = to_logical(gaze, self.devicePixelRatioF())
                    d = RING_RADIUS + RING_WIDTH + 2
                    self.update(QRect(int(x - d), int(y - d), 2 * d + 1, 2 * d + 1))
                    if state.mode in MODE_BADGES:
                        self.update(QRect(int(x + 20), int(y + 20), 240, 42))
        else:
            self.update()

    def paintEvent(self, event) -> None:
        if self.state is None:
            return
        s, dpr = self.state, self.devicePixelRatioF()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if s.snapped is not None:
            self._snapped(p, s.snapped, dpr)
        if s.strip_visible:
            self._strip(p, s, dpr)
        elif not s.paused:
            self._edge_line(p, dpr)
        if s.gaze is not None and self.ring_visible and not s.paused:
            self._ring(p, to_logical(s.gaze, dpr))
        if s.mode in MODE_BADGES and s.gaze is not None:
            x, y = to_logical(s.gaze, dpr)
            self._label(p, MODE_BADGES[s.mode], QRectF(x + 24, y + 24, 230, 34), AMBER, QColor(20, 20, 20))
        banner = PAUSED_TEXT if s.paused else s.message
        if banner:
            w = self.width()
            self._label(p, banner, QRectF(w / 2 - 260, 30, 520, 52), PANEL_FILL, TEXT)
        p.end()

    def _ring(self, p, center) -> None:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(BLUE, RING_WIDTH))
        p.drawEllipse(QRectF(center[0] - RING_RADIUS, center[1] - RING_RADIUS, 2 * RING_RADIUS, 2 * RING_RADIUS))

    def _snapped(self, p, rect, dpr) -> None:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(AMBER, 3))
        p.drawRoundedRect(_q(rect_to_logical(rect, dpr)).adjusted(-3, -3, 3, 3), 6, 6)

    def _edge_line(self, p, dpr) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(BLUE)
        p.drawRect(_q(rect_to_logical(self.layout_.edge_line, dpr)))

    def _strip(self, p, s, dpr) -> None:
        font = QFont()
        h = self.layout_.buttons[StripButton.PAUSE].h / dpr
        font.setPixelSize(max(14, round(h * 0.2)))
        font.setBold(True)
        p.setFont(font)
        for button, rect in self.layout_.buttons.items():
            q = _q(rect_to_logical(rect, dpr))
            p.setPen(QPen(QColor(255, 255, 255, 90), 2))
            p.setBrush(HOT_FILL if button is s.hot_button else PANEL_FILL)
            p.drawRoundedRect(q, 14, 14)
            p.setPen(TEXT)
            p.drawText(q, Qt.AlignCenter | Qt.TextWordWrap, BUTTON_LABELS[button])

    def _label(self, p, text, rect, fill, color) -> None:
        font = QFont()
        font.setPixelSize(18)
        font.setBold(True)
        p.setFont(font)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(rect, 10, 10)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, text)
