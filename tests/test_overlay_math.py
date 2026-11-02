import pytest
from PySide6.QtGui import QColor

from gazemouse.controller import Mode, OverlayState
from gazemouse.geometry import Rect
from gazemouse.overlay import EX_STYLE_FLAGS, Overlay, rect_to_logical, to_logical
from gazemouse.strip import StripButton, StripLayout


@pytest.mark.parametrize("dpr,expected", [(1.0, (2560, 1600)), (1.25, (2048, 1280)),
                                          (1.5, (2560 / 1.5, 1600 / 1.5)), (2.0, (1280, 800))])
def test_to_logical(dpr, expected):  # Review Focus 3
    assert to_logical((2560, 1600), dpr) == pytest.approx(expected)


def test_rect_to_logical():
    assert rect_to_logical(Rect(100, 200, 50, 20), 2.0) == Rect(50, 100, 25, 10)


def test_ex_style_is_click_through_and_never_takes_focus():
    WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x8, 0x20, 0x80000, 0x8000000
    assert EX_STYLE_FLAGS == WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE


W, H = 1280, 800
LAYOUT = StripLayout(W, H)


def state(**kw):
    base = dict(gaze=(400, 300), snapped=None, strip_visible=False, hot_button=None, mode=Mode.NORMAL,
                paused=False, message=None)
    return OverlayState(**{**base, **kw})


STATES = [
    state(),
    state(gaze=None),
    state(snapped=Rect(100, 100, 80, 30)),
    state(strip_visible=True),
    state(strip_visible=True, hot_button=StripButton.SCROLL_DOWN),
    *[state(mode=m) for m in Mode],
    state(mode=Mode.DRAGGING, gaze=None),
    state(paused=True),
    state(message="Keyboard comes in a later version"),
]


@pytest.fixture
def overlay(qapp):
    o = Overlay()
    o.resize(W, H)
    yield o
    o.close()


@pytest.mark.parametrize("s", STATES)
def test_every_state_paints(overlay, s):
    overlay.show_state(s, LAYOUT)
    assert not overlay.grab().isNull()


def painted(overlay, s, pt):
    overlay.show_state(s, LAYOUT)
    return overlay.grab().toImage().pixelColor(*pt)


def test_ring_visible_setting_hides_the_ring(overlay):
    on_ring = (400 + 18, 300)
    assert painted(overlay, state(), on_ring).alpha() > 0
    overlay.set_ring_visible(False)
    assert painted(overlay, state(), on_ring).alpha() == 0


def test_hot_button_is_drawn_differently(overlay):
    b = LAYOUT.buttons[StripButton.SCROLL_DOWN]
    pt = (round(b.x + 8), round(b.y + 8))
    cold = QColor(painted(overlay, state(strip_visible=True), pt))
    hot = painted(overlay, state(strip_visible=True, hot_button=StripButton.SCROLL_DOWN), pt)
    assert hot != cold


def test_hidden_strip_shows_only_the_edge_line(overlay):
    b = LAYOUT.buttons[StripButton.SCROLL_DOWN].center
    assert painted(overlay, state(), (round(b[0]), round(b[1]))).alpha() == 0
    assert painted(overlay, state(), (W - 2, H // 2)).alpha() > 0
