"""Real mouse and keyboard input through Win32 SendInput, in physical pixels."""
import ctypes
from ctypes import wintypes
from typing import Callable, Literal, Protocol, runtime_checkable

VK_ESCAPE = 0x1B
WHEEL_DELTA = 120  # one wheel notch

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
SM_CXSCREEN, SM_CYSCREEN = 0, 1

_BUTTON_FLAGS = {"left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                 "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


@runtime_checkable
class Actuator(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def click(self, button: Literal["left", "right"]) -> None: ...
    def double_click(self) -> None: ...
    def button_down(self, button: Literal["left", "right"]) -> None: ...
    def button_up(self, button: Literal["left", "right"]) -> None: ...
    def wheel(self, notches: int) -> None: ...  # positive = up
    def key(self, vk: int) -> None: ...


def to_absolute(x: float, y: float, screen_w: int, screen_h: int) -> tuple[int, int]:
    """Physical pixels on the primary monitor -> SendInput's 0..65535 range."""
    ax = round(x * 65535 / (screen_w - 1))
    ay = round(y * 65535 / (screen_h - 1))
    return min(max(ax, 0), 65535), min(max(ay, 0), 65535)


def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    i = INPUT(type=INPUT_MOUSE)
    i.u.mi = MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)  # mouseData is a DWORD; wheel-down is negative
    return i


def _key(vk: int, flags: int) -> INPUT:
    i = INPUT(type=INPUT_KEYBOARD)
    i.u.ki = KEYBDINPUT(vk, 0, flags, 0, 0)
    return i


def _send_input(inputs: list[INPUT]) -> int:
    arr = (INPUT * len(inputs))(*inputs)
    return ctypes.windll.user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


class SendInputActuator:
    """`send` defaults to user32.SendInput; tests pass a recorder instead."""

    def __init__(self, screen_w: int, screen_h: int, send: Callable[[list[INPUT]], int] | None = None):
        self.screen_w, self.screen_h = screen_w, screen_h
        self.send = send or _send_input

    @staticmethod
    def _flags(button):
        if button not in _BUTTON_FLAGS:
            raise ValueError(f"unknown mouse button: {button!r}")
        return _BUTTON_FLAGS[button]

    def move(self, x: int, y: int) -> None:
        ax, ay = to_absolute(x, y, self.screen_w, self.screen_h)
        self.send([_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)])

    def click(self, button: Literal["left", "right"]) -> None:
        down, up = self._flags(button)
        self.send([_mouse(down), _mouse(up)])

    def double_click(self) -> None:
        down, up = self._flags("left")
        self.send([_mouse(down), _mouse(up), _mouse(down), _mouse(up)])

    def button_down(self, button: Literal["left", "right"]) -> None:
        self.send([_mouse(self._flags(button)[0])])

    def button_up(self, button: Literal["left", "right"]) -> None:
        self.send([_mouse(self._flags(button)[1])])

    def wheel(self, notches: int) -> None:
        self.send([_mouse(MOUSEEVENTF_WHEEL, data=notches * WHEEL_DELTA)])

    def key(self, vk: int) -> None:
        self.send([_key(vk, 0), _key(vk, KEYEVENTF_KEYUP)])


def set_dpi_awareness() -> None:
    """Per-monitor DPI aware (v2), so every coordinate we see and send is a physical pixel.
    Call before any window or Qt application exists."""
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
            return
    except AttributeError:  # Windows older than 10 1703
        pass
    user32.SetProcessDPIAware()


def primary_screen_size() -> tuple[int, int]:
    """Primary monitor in physical pixels (only true after set_dpi_awareness)."""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN)
