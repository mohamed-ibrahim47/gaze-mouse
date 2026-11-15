from collections import namedtuple

import pytest

from gazemouse.blink import BlinkDetector, GazeHistory, PressTrigger, is_closed

HZ = 60
Emitted = namedtuple("Emitted", "kind aim_time emitted_at")


def feed(det, timeline):
    """Run [(duration_s, closed), ...] through the detector at 60 Hz (t = frame / 60)."""
    out, frame = [], 0
    for duration, closed in timeline:
        for _ in range(round(duration * HZ)):
            t = frame / HZ
            out += [Emitted(e.kind, e.aim_time, t) for e in det.update(t, closed)]
            frame += 1
    return out


def kinds(events):
    return [e.kind for e in events]


@pytest.fixture
def d():
    return BlinkDetector()


def test_detector_is_a_press_trigger(d):
    assert isinstance(d, PressTrigger)


def test_natural_blinks_never_click(d):  # Review Focus 2
    assert feed(d, [(0.5, False), (0.12, True)] * 20) == []


def test_long_blink_clicks_after_double_window(d):
    ev = feed(d, [(1, False), (0.5, True), (1, False)])
    assert kinds(ev) == ["click"]
    assert ev[0].aim_time == pytest.approx(1.0 - 0.2, abs=0.02)
    assert ev[0].emitted_at == pytest.approx(1.5 + 0.6, abs=0.02)


def test_two_long_blinks_make_one_double(d):
    ev = feed(d, [(1, False), (0.5, True), (0.3, False), (0.5, True), (1, False)])
    assert kinds(ev) == ["double"]
    assert ev[0].aim_time == pytest.approx(0.8, abs=0.02)   # aimed at the first closure
    assert ev[0].emitted_at == pytest.approx(2.38, abs=0.02)  # second one ended + open confirmed


def test_second_long_blink_after_window_is_two_clicks(d):
    assert kinds(feed(d, [(1, False), (0.5, True), (0.8, False), (0.5, True), (1, False)])) == ["click", "click"]


def test_natural_blink_inside_window_keeps_the_click(d):
    ev = feed(d, [(1, False), (0.5, True), (0.2, False), (0.12, True), (1, False)])
    assert kinds(ev) == ["click"]
    assert ev[0].emitted_at == pytest.approx(2.1, abs=0.02)


def test_pause_on_two_second_close(d):
    ev = feed(d, [(1, False), (2.5, True), (0.2, False)])
    assert kinds(ev) == ["pause"]
    assert ev[0].emitted_at == pytest.approx(3.58, abs=0.02)  # on reopening, once confirmed


def test_in_between_closure_ignored(d):
    assert feed(d, [(1, False), (1.7, True), (1, False)]) == []


def test_still_closed_emits_nothing_yet(d):
    assert feed(d, [(1, False), (0.5, True)]) == []


def test_cancel_forgets_the_closure(d):
    feed(d, [(1, False), (0.5, True)])
    d.cancel()
    assert feed(d, [(1, False)]) == []


def test_wink_and_squint_are_not_closed():  # Review Focus 2
    assert not is_closed(0.9, 0.3)
    assert not is_closed(0.45, 0.45)
    assert is_closed(0.6, 0.7)


def test_history_returns_sample_before_time():
    h = GazeHistory()
    h.push(1.0, (10, 10))
    h.push(1.1, (20, 20))
    assert h.at(1.05) == (10, 10)
    assert h.at(1.1) == (20, 20)
    assert h.at(0.5) is None


def test_history_forgets_old_samples():
    h = GazeHistory(keep_s=2.0)
    h.push(0.0, (1, 1))
    for i in range(1, 400):
        h.push(i / 60, (i, i))
    assert h.at(0.0) is None
    assert len(h.samples) <= 2.0 * 60 + 2


def test_one_open_frame_does_not_split_a_closure(d):  # review 4
    assert kinds(feed(d, [(1, False), (1.3, True), (1 / 60, False), (1.3, True), (0.3, False)])) == ["pause"]


def test_camera_hiccup_inside_a_blink_is_not_a_long_blink(d):  # review 4
    out = []
    for i in range(60):
        out += d.update(i / 60, False)
    out += d.update(1.0, True)          # a natural blink starts...
    for i in range(60):
        out += d.update(1.45 + i / 60, False)  # ...and the next frame comes 0.45 s later
    assert out == []


def test_closure_needs_several_closed_frames(d):  # review 4
    out = d.update(0.0, False) + d.update(1.0, True) + d.update(1.2, True)
    for i in range(60):
        out += d.update(1.45 + i / 60, False)  # 0.45 s long, but only two frames saw it
    assert out == []


def test_after_cancel_eyes_must_be_seen_open_first(d):  # review 4
    feed(d, [(1, False), (0.5, True)])
    d.cancel()
    assert feed(d, [(0.8, True), (1, False)]) == []


def test_stall_in_a_closure_discards_it_even_with_enough_frames(d):  # review 4
    out = d.update(0.0, False)
    for t in (1.0, 1.02, 1.04, 1.5):  # the frame after 1.04 arrives 0.46 s late
        out += d.update(t, True)
    for i in range(60):
        out += d.update(1.52 + i / 60, False)
    assert out == []


# --- LidGuard: freeze the cursor while the lids are on their way down or back up ---

from gazemouse.blink import HOLD_MAX_S, SETTLE_S, LidGuard

F = 1 / 30


def guard_feed(g, t, seconds, lids, closed=False):
    held = []
    for _ in range(round(seconds / F)):
        held.append(g.update(t, lids, closed))
        t += F
    return held, t


def test_guard_open_eyes_never_hold():
    held, _ = guard_feed(LidGuard(), 0.0, 2.0, 0.1)
    assert not any(held)


def test_guard_holds_from_the_first_lid_drop_until_after_reopening():
    g = LidGuard()
    _, t = guard_feed(g, 0.0, 1.0, 0.1)
    drop, t = guard_feed(g, t, 0.1, 0.45)             # lids coming down, not closed yet
    shut, t = guard_feed(g, t, 0.6, 0.9, closed=True)
    back, t = guard_feed(g, t, SETTLE_S - F, 0.1)     # just reopened: the iris reading isn't settled
    later, t = guard_feed(g, t + F, 0.5, 0.1)
    assert all(drop) and all(shut) and all(back) and not any(later)


def test_guard_looking_down_only_holds_briefly():
    g = LidGuard()
    _, t = guard_feed(g, 0.0, 1.0, 0.1)
    held, t = guard_feed(g, t, 2.0, 0.45)  # score up and staying up, lids never close
    assert held[0] and not any(held[round((HOLD_MAX_S + 2 * F) / F):])
    # the new level is normal now: a blink from here is caught again
    _, t = guard_feed(g, t, 0.1, 0.4)
    drop, _ = guard_feed(g, t, 0.1, 0.75)
    assert all(drop)


def test_guard_small_flicker_is_ignored():
    g = LidGuard()
    _, t = guard_feed(g, 0.0, 1.0, 0.1)
    held, _ = guard_feed(g, t, 0.3, 0.2)
    assert not any(held)


def test_guard_frozen_point_is_from_before_the_lids_moved():
    g = LidGuard()
    t = 0.0
    for _ in range(30):
        g.update(t, 0.1, False)
        g.remember(t, (500, 400))
        t += F
    g.update(t, 0.15, False)        # the drop has begun but is too small to notice yet
    g.remember(t, (500, 430))
    t += F
    assert g.update(t, 0.45, False)
    assert g.frozen == (500, 400)


def test_guard_no_history_means_nothing_to_freeze_on():
    g = LidGuard()
    assert g.update(0.0, 0.9, True) and g.frozen is None
