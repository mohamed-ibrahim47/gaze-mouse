import itertools

import pytest

from gazemouse.geometry import Rect
from gazemouse.strip import StripButton, StripLayout, StripReveal

W, H = 2560, 1600
DT = 1 / 60


def overlaps(a, b):
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h


def look(reveal, point, seconds, t0):
    """Hold the gaze on point for `seconds` at 60 Hz; returns (visible, next t)."""
    visible, n = False, round(seconds * 60)
    for i in range(n):
        visible = reveal.update(t0 + i * DT, point)
    return visible, t0 + n * DT


EDGE = (0.98 * W, 0.5 * H)
PAGE = (0.5 * W, 0.5 * H)


def revealed():
    r = StripReveal(StripLayout(W, H))
    visible, t = look(r, EDGE, 0.5, 0.0)
    assert visible
    return r, t


def test_rect_basics():
    r = Rect(10, 20, 100, 50)
    assert r.contains((10, 20)) and r.contains((110, 70)) and not r.contains((111, 70))
    assert r.center == (60, 45)
    assert r.distance_to((60, 45)) == 0 and r.distance_to((113, 74)) == pytest.approx(5)
    assert r.expanded(5) == Rect(5, 15, 110, 60)


def test_reveal_needs_0_3_s():
    r = StripReveal(StripLayout(W, H))
    visible, t = look(r, EDGE, 0.2, 0.0)
    assert not visible
    visible, _ = look(r, EDGE, 0.15, t)
    assert visible


def test_reading_near_edge_briefly_does_not_reveal():
    r = StripReveal(StripLayout(W, H))
    t = 0.0
    for _ in range(5):
        visible, t = look(r, EDGE, 0.25, t)
        assert not visible
        visible, t = look(r, PAGE, 0.5, t)
        assert not visible


def test_hides_after_looking_back_0_4_s():
    r, t = revealed()
    visible, t = look(r, PAGE, 0.3, t)
    assert visible
    visible, _ = look(r, PAGE, 0.15, t)
    assert not visible


def test_stays_visible_over_buttons():
    r, t = revealed()
    button = r.layout.buttons[StripButton.SCROLL_DOWN].center
    visible, _ = look(r, button, 3.0, t)
    assert visible


def test_glance_away_and_back_resets_hide_timer():
    r, t = revealed()
    button = r.layout.buttons[StripButton.PAUSE].center
    for _ in range(4):
        visible, t = look(r, PAGE, 0.3, t)
        visible, t = look(r, button, 0.2, t)
        assert visible


def test_no_face_freezes_timers():
    r = StripReveal(StripLayout(W, H))
    _, t = look(r, EDGE, 0.2, 0.0)
    visible, t = look(r, None, 2.0, t)
    assert not visible
    visible, _ = look(r, EDGE, 0.15, t)
    assert visible


def test_hide_resets():
    r, t = revealed()
    r.hide()
    assert not r.update(t, EDGE)


def test_gaze_far_off_screen_does_not_reveal():  # Review Focus 5
    r = StripReveal(StripLayout(W, H))
    visible, _ = look(r, (1.3 * W, 0.5 * H), 2.0, 0.0)
    assert not visible


@pytest.mark.parametrize("w,h", [(1366, 768), (2560, 1600), (3840, 2400)])  # Review Focus 3
def test_layout_fits_and_buttons_do_not_overlap(w, h):
    L = StripLayout(w, h)
    rects = list(L.buttons.values())
    assert list(L.buttons) == list(StripButton)  # top to bottom
    assert all(0 <= r.x and r.x + r.w <= w and 0 <= r.y and r.y + r.h <= h for r in rects)
    assert all(not overlaps(a, b) for a, b in itertools.combinations(rects, 2))
    assert all(a.y < b.y for a, b in zip(rects, rects[1:]))
    assert L.buttons[StripButton.SCROLL_UP].w == pytest.approx(0.12 * w)
    assert L.edge_line.w == pytest.approx(max(6, 0.006 * w)) and L.edge_line.h == h
    assert all(L.panel.contains(r.center) for r in rects)


def test_button_at_center_and_outside():
    L = StripLayout(W, H)
    for button, rect in L.buttons.items():
        assert L.button_at(rect.center) is button
    assert L.button_at(PAGE) is None
    gap_y = L.buttons[StripButton.SCROLL_UP].y + L.buttons[StripButton.SCROLL_UP].h + 0.007 * H
    assert L.button_at((L.buttons[StripButton.SCROLL_UP].center[0], gap_y)) is None
