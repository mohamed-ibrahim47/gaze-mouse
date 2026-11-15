import ctypes
import sys

import pytest

from gazemouse import actions
from gazemouse.actions import (INPUT, VK_ESCAPE, Actuator, SendInputActuator, primary_screen_size,
                               set_dpi_awareness, to_absolute)

SCREENS = [(1366, 768), (2560, 1600), (3840, 2400)]


@pytest.mark.parametrize("w,h", SCREENS)  # Review Focus 3
def test_corners_map_to_full_range(w, h):
    assert to_absolute(0, 0, w, h) == (0, 0)
    assert to_absolute(w - 1, h - 1, w, h) == (65535, 65535)


@pytest.mark.parametrize("w,h", SCREENS)
def test_centre_maps_to_middle(w, h):
    x, y = to_absolute((w - 1) / 2, (h - 1) / 2, w, h)
    assert abs(x - 32768) <= 1 and abs(y - 32768) <= 1


def test_out_of_range_is_clamped():
    assert to_absolute(-50, 99999, 2560, 1600) == (0, 65535)


def test_input_struct_layout():
    # 64-bit Windows: 4-byte type + 4 padding + 32-byte union (MOUSEINPUT is the largest).
    assert ctypes.sizeof(INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


class Recorder:
    """Stands in for user32.SendInput: records what would have been sent."""

    def __init__(self):
        self.calls = []

    def __call__(self, inputs):
        self.calls.append([self._decode(i) for i in inputs])
        return len(inputs)

    @staticmethod
    def _decode(i):
        if i.type == actions.INPUT_MOUSE:
            m = i.u.mi
            return ("mouse", m.dx, m.dy, m.dwFlags, ctypes.c_int32(m.mouseData).value)
        k = i.u.ki
        return ("key", k.wVk, k.dwFlags)


@pytest.fixture
def act():
    rec = Recorder()
    return SendInputActuator(2560, 1600, send=rec), rec


def test_actuator_matches_protocol(act):
    assert isinstance(act[0], Actuator)


def test_move_is_absolute(act):
    a, rec = act
    a.move(2559, 0)
    [[(kind, dx, dy, flags, _)]] = rec.calls
    assert kind == "mouse" and (dx, dy) == (65535, 0)
    assert flags == actions.MOUSEEVENTF_MOVE | actions.MOUSEEVENTF_ABSOLUTE


def test_clicks_send_down_then_up_in_one_call(act):
    a, rec = act
    a.click("left")
    a.click("right")
    a.double_click()
    flags = [[e[3] for e in call] for call in rec.calls]
    L, Lu = actions.MOUSEEVENTF_LEFTDOWN, actions.MOUSEEVENTF_LEFTUP
    assert flags == [[L, Lu], [actions.MOUSEEVENTF_RIGHTDOWN, actions.MOUSEEVENTF_RIGHTUP], [L, Lu, L, Lu]]


def test_drag_buttons(act):
    a, rec = act
    a.button_down("left")
    a.button_up("left")
    assert [c[0][3] for c in rec.calls] == [actions.MOUSEEVENTF_LEFTDOWN, actions.MOUSEEVENTF_LEFTUP]


def test_wheel_sign_and_size(act):
    a, rec = act
    a.wheel(2)
    a.wheel(-1)
    assert [(c[0][3], c[0][4]) for c in rec.calls] == [(actions.MOUSEEVENTF_WHEEL, 240),
                                                       (actions.MOUSEEVENTF_WHEEL, -120)]


def test_key_press_and_release(act):
    a, rec = act
    a.key(VK_ESCAPE)
    assert rec.calls == [[("key", 0x1B, 0), ("key", 0x1B, actions.KEYEVENTF_KEYUP)]]


def test_unknown_button_rejected(act):
    with pytest.raises(ValueError):
        act[0].click("middle")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_primary_screen_size_is_physical_pixels():
    set_dpi_awareness()
    w, h = primary_screen_size()
    assert w >= 640 and h >= 480
