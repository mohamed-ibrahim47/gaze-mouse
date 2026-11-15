import os
import time
from types import SimpleNamespace

import pytest

from gazemouse.geometry import Rect
from gazemouse.snap import BackgroundTargets, UiaTargetProvider, snap


def test_point_inside_rect_snaps_to_centre():
    r = snap((105, 105), [Rect(100, 100, 40, 20)], 100)
    assert r.point == (120, 110) and r.rect == Rect(100, 100, 40, 20)


def test_nested_rects_pick_the_smaller():
    big, small = Rect(0, 0, 400, 400), Rect(90, 90, 30, 30)
    assert snap((100, 100), [big, small], 100).rect == small


def test_nearest_within_radius_wins():
    near, far = Rect(530, 495, 10, 10), Rect(580, 495, 10, 10)  # 30 px and 80 px to the right
    r = snap((500, 500), [far, near], 100)
    assert r.rect == near and r.point == near.center


def test_nothing_within_radius_keeps_point():
    r = snap((500, 500), [Rect(650, 495, 10, 10)], 100)
    assert r.point == (500, 500) and r.rect is None


def test_empty_list_keeps_point():
    assert snap((1, 2), [], 100).rect is None and snap((1, 2), [], 100).point == (1, 2)


class FakeProvider:
    def __init__(self, rects=None, fail=False):
        self.rects, self.fail, self.calls = rects or [], fail, []

    def targets_near(self, point, radius):
        self.calls.append((point, radius))
        if self.fail:
            raise RuntimeError("COM error")
        return self.rects


def wait_for(cond, timeout=0.5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_background_targets_follow_the_point():
    rects = [Rect(1, 2, 3, 4)]
    p = FakeProvider(rects)
    bg = BackgroundTargets(p, radius=150, hz=50)
    try:
        assert bg.targets() == []
        bg.set_point((10, 20))
        assert wait_for(lambda: bg.targets() == rects)
        assert p.calls[-1] == ((10, 20), 150)
    finally:
        bg.stop()


def test_background_targets_survive_provider_errors():
    p = FakeProvider(fail=True)
    bg = BackgroundTargets(p, radius=150, hz=50)
    try:
        bg.set_point((10, 20))
        assert wait_for(lambda: len(p.calls) >= 3)
        assert bg.targets() == [] and bg.thread.is_alive()
    finally:
        bg.stop()
    assert not bg.thread.is_alive()


# --- UiaTargetProvider with fake controls (no real UI Automation) ---

def control(kind, rect, parent=None, pid=999):
    left, top, w, h = rect
    c = SimpleNamespace(ControlTypeName=kind, ProcessId=pid,
                        BoundingRectangle=SimpleNamespace(left=left, top=top, right=left + w, bottom=top + h))
    c.GetParentControl = lambda: parent
    return c


def provider(hit, screen=(2560, 1600)):
    return UiaTargetProvider(*screen, control_from_point=hit)


def test_walks_up_to_clickable_parent():
    button = control("ButtonControl", (100, 100, 80, 30))
    text = control("TextControl", (110, 105, 40, 20), parent=button)
    assert provider(lambda x, y: text).targets_near((120, 110), 60) == [Rect(100, 100, 80, 30)]


def test_gives_up_after_three_parents():
    button = control("ButtonControl", (0, 0, 80, 30))
    c = button
    for _ in range(4):
        c = control("PaneControl", (0, 0, 80, 30), parent=c)
    assert provider(lambda x, y: c).targets_near((10, 10), 60) == []


def test_skips_own_process_and_huge_rects():
    mine = control("ButtonControl", (0, 0, 80, 30), pid=os.getpid())
    huge = control("ListItemControl", (0, 0, 2000, 1500))
    assert provider(lambda x, y: mine).targets_near((10, 10), 60) == []
    assert provider(lambda x, y: huge).targets_near((10, 10), 60) == []


def test_dedupes_and_probes_nine_points():
    link = control("HyperlinkControl", (50, 50, 100, 20))
    probes = []

    def hit(x, y):
        probes.append((x, y))
        return link

    assert provider(hit).targets_near((100, 60), 40) == [Rect(50, 50, 100, 20)]
    assert len(probes) == 9 and (100, 60) in probes


def test_com_error_returns_what_was_found():
    link = control("HyperlinkControl", (50, 50, 100, 20))
    calls = []

    def hit(x, y):
        calls.append(1)
        if len(calls) > 1:
            raise OSError("COM error")
        return link

    assert provider(hit).targets_near((100, 60), 40) == [Rect(50, 50, 100, 20)]


# --- sticky snapping: a wobbling gaze must not flip the cursor between neighbours ---

from gazemouse.snap import SWITCH_S, StickySnap

S_A, S_B = Rect(1000, 500, 100, 40), Rect(1000, 560, 100, 40)  # stacked, 20 px apart
S_R = 150
IN_A, IN_B = (1050, 520), (1050, 580)
FAR = (200, 1200)


def test_sticky_first_target_locks_at_once():
    s = StickySnap(S_R)
    r = s.update(0.0, IN_A, [S_A, S_B])
    assert r.rect == S_A and r.point == S_A.center


def test_sticky_brief_wobble_onto_a_neighbour_keeps_the_lock():
    s = StickySnap(S_R)
    s.update(0.0, IN_A, [S_A, S_B])
    assert s.update(0.05, IN_B, [S_A, S_B]).rect == S_A
    assert s.update(0.10, IN_A, [S_A, S_B]).rect == S_A
    # the wobble's time doesn't add up across visits
    assert s.update(0.10 + SWITCH_S - 0.01, IN_B, [S_A, S_B]).rect == S_A


def test_sticky_steady_look_at_the_neighbour_switches():
    s = StickySnap(S_R)
    s.update(0.0, IN_A, [S_A, S_B])
    assert s.update(0.1, IN_B, [S_A, S_B]).rect == S_A
    assert s.update(0.1 + SWITCH_S, IN_B, [S_A, S_B]).rect == S_B


def test_sticky_look_far_away_lets_go_at_once():
    s = StickySnap(S_R)
    s.update(0.0, IN_A, [S_A, S_B])
    r = s.update(0.02, FAR, [S_A, S_B])
    assert r.rect is None and r.point == FAR


def test_sticky_just_outside_the_radius_holds_for_a_moment():
    s = StickySnap(S_R)
    s.update(0.0, IN_A, [S_A])
    edge = (S_A.x + S_A.w + S_R + 10, 520)  # plain snap says "nothing", but it's still close
    assert s.update(0.05, edge, [S_A]).rect == S_A
    assert s.update(0.05 + SWITCH_S, edge, [S_A]).rect is None


def test_sticky_reset_forgets_the_lock():
    s = StickySnap(S_R)
    s.update(0.0, IN_A, [S_A, S_B])
    s.reset()
    assert s.update(0.01, IN_B, [S_A, S_B]).rect == S_B
