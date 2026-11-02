import pytest

from gazemouse.actions import VK_ESCAPE
from gazemouse.blink import BlinkDetector, PressTrigger
from gazemouse.controller import Controller, Mode
from gazemouse.geometry import Rect
from gazemouse.strip import StripButton, StripLayout

W, H = 2560, 1600
DT = 1 / 60
A = Rect(1000, 500, 100, 40)   # a snap target
P1, P2 = (300, 300), (600, 1200)  # page points far from any target
EDGE = (0.98 * W, 0.5 * H)
LAYOUT = StripLayout(W, H)


def btn(b):
    return LAYOUT.buttons[b].center


class FakeActuator:
    def __init__(self):
        self.calls = []

    def move(self, x, y):
        self.calls.append(("move", (x, y)))

    def click(self, button):
        self.calls.append(("click", (button,)))

    def double_click(self):
        self.calls.append(("double_click", ()))

    def button_down(self, button):
        self.calls.append(("button_down", (button,)))

    def button_up(self, button):
        self.calls.append(("button_up", (button,)))

    def wheel(self, notches):
        self.calls.append(("wheel", (notches,)))

    def key(self, vk):
        self.calls.append(("key", (vk,)))

    def named(self, name):
        return [c for c in self.calls if c[0] == name]

    def without_moves(self):
        return [c for c in self.calls if c[0] != "move"]

    def last_move(self):
        return self.named("move")[-1][1]


class Rig:
    def __init__(self):
        self.fake = FakeActuator()
        self.ctrl = Controller(W, H, self.fake, lambda: [A], 0.06 * W)
        self.frame = 0
        self.state = None

    def run(self, *script):
        """[(duration_s, gaze, closed), ...] at 60 Hz. closed=None with gaze=None is a lost face."""
        for duration, gaze, closed in script:
            for _ in range(round(duration * 60)):
                self.state = self.ctrl.step(self.frame / 60, gaze, closed)
                self.frame += 1
        return self.state

    def blink_on(self, point, wait=0.8):
        """Look at the point, long-blink, keep looking until the click has fired."""
        return self.run((0.4, point, False), (0.5, None, True), (wait, point, False))

    def open_strip(self):
        return self.run((0.5, EDGE, False))

    def press(self, button):
        self.open_strip()
        return self.blink_on(btn(button))


@pytest.fixture
def rig():
    return Rig()


def test_gaze_moves_cursor_to_snapped_target(rig):
    s = rig.run((0.2, (1040, 600), False))  # near A, not on it
    assert rig.fake.last_move() == A.center and s.snapped == A


def test_gaze_away_from_targets_moves_to_gaze(rig):
    s = rig.run((0.2, P1, False))
    assert rig.fake.last_move() == P1 and s.snapped is None


def test_small_jitter_does_not_flood_moves(rig):
    rig.run((0.5, P1, False))
    assert len(rig.fake.named("move")) == 1


def test_long_blink_clicks_where_user_looked_before_closing(rig):
    # Eyes drift down as the lids close, then the gaze disappears altogether.
    down = (A.center[0], A.center[1] + 200)
    rig.run((0.5, A.center, False), (0.05, down, False), (0.5, None, True), (0.8, P2, False))
    assert ("click", ("left",)) in rig.fake.calls
    i = rig.fake.calls.index(("click", ("left",)))
    assert rig.fake.calls[i - 1] == ("move", A.center)


def test_very_long_click_blink_still_finds_its_aim(rig):
    # 1.4 s closure + 0.6 s window + 0.2 s aim-back: older than GazeHistory's default 2 s.
    rig.run((0.5, A.center, False), (1.4, None, True), (0.8, P2, False))
    assert rig.fake.named("click") == [("click", ("left",))]
    i = rig.fake.calls.index(("click", ("left",)))
    assert rig.fake.calls[i - 1] == ("move", A.center)


def test_double_blink_double_clicks(rig):
    rig.run((0.5, A.center, False), (0.5, None, True), (0.3, A.center, False), (0.5, None, True),
            (0.8, A.center, False))
    assert rig.fake.named("double_click") == [("double_click", ())]
    assert rig.fake.named("click") == []


def test_slow_double_blink_still_finds_its_aim(rig):
    rig.run((0.5, A.center, False), (1.4, None, True), (0.5, A.center, False), (1.4, None, True),
            (0.8, P2, False))
    assert rig.fake.named("double_click") == [("double_click", ())]


def test_right_click_is_one_shot(rig):
    s = rig.press(StripButton.RIGHT_CLICK)
    assert s.mode is Mode.RIGHT_ONCE
    rig.run((0.6, P1, False))
    rig.blink_on(P1)
    assert rig.fake.named("click") == [("click", ("right",))]
    rig.blink_on(P1)
    assert rig.fake.named("click") == [("click", ("right",)), ("click", ("left",))]


def test_button_press_never_clicks_the_page(rig):
    rig.press(StripButton.RIGHT_CLICK)
    assert rig.fake.named("click") == []


def test_button_still_pressed_if_strip_hid_before_click_fired(rig):
    # Looks away right after the blink: the strip hides inside the 0.6 s wait.
    rig.open_strip()
    s = rig.run((0.4, btn(StripButton.RIGHT_CLICK), False), (0.5, None, True), (0.9, P1, False))
    assert not s.strip_visible and s.mode is Mode.RIGHT_ONCE
    assert rig.fake.named("click") == []


def test_blink_on_panel_gap_does_nothing(rig):
    rig.open_strip()
    b = LAYOUT.buttons[StripButton.SCROLL_UP]
    gap = (b.center[0], b.y + b.h + 0.007 * H)
    assert LAYOUT.button_at(gap) is None
    rig.blink_on(gap)
    assert rig.fake.without_moves() == []


def test_strip_gaze_leaves_cursor_alone(rig):
    rig.run((0.3, P1, False))
    rig.open_strip()
    moves = len(rig.fake.named("move"))
    s = rig.run((1.0, btn(StripButton.PAUSE), False))
    assert len(rig.fake.named("move")) == moves
    assert s.strip_visible and s.hot_button is StripButton.PAUSE


def test_select_drag_down_then_up(rig):
    s = rig.press(StripButton.SELECT)
    assert s.mode is Mode.SELECT_START
    rig.run((0.6, P1, False))
    s = rig.blink_on(P1)
    assert s.mode is Mode.DRAGGING
    i = rig.fake.calls.index(("button_down", ("left",)))
    assert rig.fake.calls[i - 1] == ("move", P1)
    rig.run((0.3, P2, False))
    assert rig.fake.last_move() == P2
    s = rig.blink_on(P2)
    assert s.mode is Mode.NORMAL
    i = rig.fake.calls.index(("button_up", ("left",)))
    assert rig.fake.calls[i - 1] == ("move", P2)


def test_select_does_not_snap(rig):
    # A link near the text must not drag the selection start away from it.
    near_a = (A.x - 50, A.center[1])
    rig.press(StripButton.SELECT)
    rig.run((0.6, near_a, False))
    assert rig.fake.last_move() == near_a
    rig.blink_on(near_a)
    i = rig.fake.calls.index(("button_down", ("left",)))
    assert rig.fake.calls[i - 1] == ("move", near_a)


def test_no_strip_while_dragging(rig):
    rig.press(StripButton.SELECT)
    rig.run((0.6, P1, False))
    rig.blink_on(P1)
    assert not rig.run((1.0, EDGE, False)).strip_visible
    assert rig.fake.last_move() == (round(EDGE[0]), round(EDGE[1]))


def test_scroll_runs_while_looking_and_stops_on_leave(rig):
    rig.run((0.5, P2, False))
    s = rig.press(StripButton.SCROLL_DOWN)
    assert s.mode is Mode.SCROLLING
    assert ("move", P2) in rig.fake.calls
    rig.fake.calls.clear()
    rig.run((1.0, btn(StripButton.SCROLL_DOWN), False))
    wheels = rig.fake.named("wheel")
    assert 9 <= len(wheels) <= 11 and set(wheels) == {("wheel", (-1,))}
    s = rig.run((0.2, P1, False))
    assert s.mode is Mode.NORMAL
    rig.fake.calls.clear()
    rig.run((0.5, P1, False))
    assert rig.fake.named("wheel") == []


def test_scroll_up_direction(rig):
    rig.press(StripButton.SCROLL_UP)
    rig.fake.calls.clear()
    rig.run((0.5, btn(StripButton.SCROLL_UP), False))
    assert set(rig.fake.named("wheel")) == {("wheel", (1,))}


def test_natural_blink_does_not_stop_scrolling(rig):
    rig.press(StripButton.SCROLL_DOWN)
    b = btn(StripButton.SCROLL_DOWN)
    s = rig.run((0.3, b, False), (0.12, None, True), (0.05, None, False), (0.3, b, False))
    assert s.mode is Mode.SCROLLING


def test_long_blink_stops_scrolling_without_clicking(rig):
    rig.press(StripButton.SCROLL_DOWN)
    b = btn(StripButton.SCROLL_DOWN)
    s = rig.run((0.3, b, False), (0.5, None, True), (0.8, b, False))
    assert s.mode is Mode.NORMAL and rig.fake.named("click") == []


def test_pause_blocks_everything_until_pause_again(rig):
    s = rig.run((0.5, P1, False), (2.2, None, True), (0.1, P1, False))
    assert s.paused and not s.strip_visible
    rig.fake.calls.clear()
    s = rig.run((1.0, P2, False), (1.0, EDGE, False))
    assert not s.strip_visible
    rig.blink_on(P2)
    assert rig.fake.calls == []
    s = rig.run((0.3, P2, False), (2.2, None, True), (0.1, P2, False))
    assert not s.paused and rig.fake.calls[-1] == ("move", P2)


def test_pause_button(rig):
    s = rig.press(StripButton.PAUSE)
    assert s.paused


def test_pause_mid_drag_releases_the_button(rig):
    rig.press(StripButton.SELECT)
    rig.run((0.6, P1, False))
    rig.blink_on(P1)
    s = rig.run((2.2, None, True), (0.1, P1, False))
    assert s.paused and s.mode is Mode.NORMAL
    assert rig.fake.calls[-2:] == [("key", (VK_ESCAPE,)), ("button_up", ("left",))]


def test_release_on_quit_lets_go_of_the_button(rig):
    rig.press(StripButton.SELECT)
    rig.run((0.6, P1, False))
    rig.blink_on(P1)
    rig.ctrl.release()
    assert rig.fake.calls[-1] == ("button_up", ("left",))
    rig.ctrl.release()
    assert len(rig.fake.named("button_up")) == 1


def test_keyboard_button_shows_a_message_for_a_while(rig):
    s = rig.press(StripButton.KEYBOARD)
    assert s.message == "Keyboard comes in a later version"
    assert rig.run((2.1, P1, False)).message is None


def test_face_lost_stops_scroll_immediately(rig):  # Review Focus 1
    rig.press(StripButton.SCROLL_DOWN)
    s = rig.run((1 / 60, None, None))
    assert s.mode is Mode.NORMAL
    rig.fake.calls.clear()
    rig.run((0.5, None, None))
    assert rig.fake.named("wheel") == []


def test_face_lost_during_drag_cancels_after_3_s(rig):  # Review Focus 1
    rig.press(StripButton.SELECT)
    rig.run((0.6, P1, False))
    rig.blink_on(P1)
    rig.fake.calls.clear()
    s = rig.run((2.9, None, None))
    assert s.mode is Mode.DRAGGING and rig.fake.calls == []
    s = rig.run((0.2, None, None))
    assert s.mode is Mode.NORMAL and s.message == "Drag cancelled: face lost"
    assert rig.fake.calls == [("key", (VK_ESCAPE,)), ("button_up", ("left",))]


def test_face_back_in_time_keeps_the_drag(rig):
    rig.press(StripButton.SELECT)
    rig.run((0.6, P1, False))
    rig.blink_on(P1)
    s = rig.run((2.0, None, None), (0.1, P2, False), (2.0, None, None), (0.1, P2, False))
    assert s.mode is Mode.DRAGGING


def test_face_lost_with_eyes_closed_is_not_a_click(rig):
    # The closure ends because the face vanished, not because the eyes opened.
    rig.run((0.5, A.center, False), (0.5, None, True), (1.5, None, None), (1.0, A.center, False))
    assert rig.fake.without_moves() == []


def test_off_screen_gaze_does_not_move_or_click(rig):  # Review Focus 5
    away = (1280, 1.3 * H)
    rig.run((0.5, away, False), (0.5, None, True), (1.0, away, False))
    assert rig.fake.calls == []


def test_blink_after_looking_at_keyboard_does_not_click_stale_point(rig):
    rig.run((0.5, A.center, False), (1.0, (1280, 1.3 * H), False))
    rig.fake.calls.clear()
    rig.run((0.5, None, True), (1.0, (1280, 1.3 * H), False))
    assert rig.fake.calls == []


def test_off_screen_gaze_does_not_reveal_strip(rig):  # Review Focus 5
    assert not rig.run((1.0, (1.3 * W, 800), False)).strip_visible


def test_gaze_just_outside_is_clamped(rig):
    s = rig.run((0.2, (-40, 1650), False))
    assert rig.fake.last_move() == (0, H - 1) and s.gaze == (0, H - 1)


def test_presses_are_pluggable():
    class OneClick:
        def __init__(self):
            self.done = False

        def update(self, t, closed):
            if t > 0.3 and not self.done:
                self.done = True
                return [type("E", (), {"kind": "click", "aim_time": 0.2})()]
            return []

        def cancel(self):
            pass

    assert isinstance(OneClick(), PressTrigger) and isinstance(BlinkDetector(), PressTrigger)
    fake = FakeActuator()
    c = Controller(W, H, fake, lambda: [], 100, presses=OneClick())
    for i in range(30):
        c.step(i / 60, P1, False)
    assert fake.named("click") == [("click", ("left",))]
