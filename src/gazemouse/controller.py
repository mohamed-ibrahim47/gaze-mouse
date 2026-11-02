"""The brain: turns gaze + eyes-closed into cursor moves, clicks, scrolling and an overlay state.

Pure logic with an injected Actuator, stepped once per camera frame.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .actions import VK_ESCAPE, Actuator
from .blink import BlinkDetector, BlinkEvent, GazeHistory, PressTrigger
from .geometry import Rect
from .snap import snap
from .strip import StripButton, StripLayout, StripReveal

OFF_SCREEN_MARGIN = 0.1  # gaze further outside than this fraction of the screen is "away"
MIN_MOVE_PX = 2
SCROLL_PERIOD_S = 0.1
SCROLL_KEEP_PAD = 0.03   # x W, around the pressed button
DRAG_FACE_LOST_S = 3.0
MESSAGE_S = 2.0
# A click fires up to 0.2 + 1.5 + 0.6 s after its aim; a slow double up to about 4 s after.
HISTORY_S = 5.0


class Mode(Enum):
    NORMAL = "normal"
    RIGHT_ONCE = "right_once"
    SELECT_START = "select_start"
    DRAGGING = "dragging"
    SCROLLING = "scrolling"


@dataclass
class OverlayState:
    gaze: tuple[float, float] | None
    snapped: Rect | None
    strip_visible: bool
    hot_button: StripButton | None
    mode: Mode
    paused: bool
    message: str | None


class Controller:
    def __init__(self, screen_w: int, screen_h: int, actuator: Actuator,
                 targets: Callable[[], list[Rect]], snap_radius: float,
                 presses: PressTrigger | None = None):
        self.w, self.h = screen_w, screen_h
        self.act, self.targets, self.snap_radius = actuator, targets, snap_radius
        self.presses = presses or BlinkDetector()
        self.layout = StripLayout(screen_w, screen_h)
        self.reveal = StripReveal(self.layout)
        self.history = GazeHistory(HISTORY_S)        # None samples mark "looking away"
        self.strip_history = GazeHistory(HISTORY_S)  # was the strip open at that moment?
        self.mode = Mode.NORMAL
        self.paused = False
        self.last_page_point = (screen_w / 2, screen_h / 2)
        self.last_move = None
        self.was_closed = False
        self.face_lost_since = None
        self.scroll_rect, self.scroll_dir, self.next_wheel = None, 0, 0.0
        self.message, self.message_until = None, 0.0
        self.t = 0.0

    # --- outside controls ---

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.release()
            self.reveal.hide()

    def release(self) -> None:
        """Back to NORMAL with nothing held. Also called on quit."""
        if self.mode is Mode.DRAGGING:
            self.act.key(VK_ESCAPE)
            self.act.button_up("left")
        self.mode = Mode.NORMAL

    # --- per frame ---

    def step(self, t: float, gaze, closed: bool | None) -> OverlayState:
        self.t = t
        face_lost = gaze is None and closed is None
        self.face_lost_since = (self.face_lost_since if self.face_lost_since is not None else t) if face_lost else None
        if closed is None and self.was_closed:
            self.presses.cancel()  # the eyes didn't reopen, we just can't see them
        self.was_closed = bool(closed)

        away = gaze is not None and self._is_away(gaze)
        point = None if gaze is None or away else self._clamp(gaze)
        if point is not None or away or face_lost:
            self.history.push(t, point)

        events = self.presses.update(t, bool(closed))
        if self.paused or self.mode is Mode.DRAGGING:
            self.reveal.hide()
            visible = False
        else:
            visible = self.reveal.update(t, point)
        self.strip_history.push(t, visible)

        for e in events:
            self._on_event(e)
        visible = self.reveal.visible and not self.paused

        snapped = None
        if not self.paused:
            if self.mode is Mode.DRAGGING:
                self._track_drag(t, point)
            elif self.mode is Mode.SCROLLING:
                self._track_scroll(t, point, away or face_lost)
            elif point is not None and not self._on_strip(point, visible):
                snapped = self._track_page(point)

        hot = self.layout.button_at(point) if visible and point is not None else None
        if t >= self.message_until:
            self.message = None
        return OverlayState(point, snapped, visible, hot, self.mode, self.paused, self.message)

    # --- gaze tracking ---

    def _is_away(self, gaze) -> bool:
        mx, my = OFF_SCREEN_MARGIN * self.w, OFF_SCREEN_MARGIN * self.h
        return not (-mx <= gaze[0] <= self.w + mx and -my <= gaze[1] <= self.h + my)

    def _clamp(self, gaze):
        return (min(max(gaze[0], 0), self.w - 1), min(max(gaze[1], 0), self.h - 1))

    def _on_strip(self, point, visible) -> bool:
        return visible and point[0] >= self.layout.panel.x

    def _move(self, point) -> None:
        x, y = round(point[0]), round(point[1])
        if self.last_move is None or max(abs(x - self.last_move[0]), abs(y - self.last_move[1])) >= MIN_MOVE_PX:
            self.act.move(x, y)
            self.last_move = (x, y)

    def _move_exact(self, point) -> None:
        """Before a press: the cursor must be there, whatever the last move was."""
        self.last_move = None
        self._move(point)

    def _aim(self, point):
        return snap(point, self.targets(), self.snap_radius)

    def _track_page(self, point) -> Rect | None:
        rect, aimed = None, point
        if self.mode is not Mode.SELECT_START:  # a nearby link must not pull a text selection away
            s = self._aim(point)
            rect, aimed = s.rect, s.point
        self._move(aimed)
        if not self.layout.edge_zone.contains(point):  # on its way to the strip isn't aiming at the page
            self.last_page_point = aimed
        return rect

    def _track_drag(self, t, point) -> None:
        if point is not None:
            self._move(point)
        if self.face_lost_since is not None and t - self.face_lost_since > DRAG_FACE_LOST_S:
            self.release()
            self._say("Drag cancelled: face lost")

    def _track_scroll(self, t, point, gone) -> None:
        if gone or (point is not None and not self.scroll_rect.contains(point)):
            self.mode = Mode.NORMAL
        elif point is not None and t >= self.next_wheel:  # None here is a blink: hold, don't stop
            self.act.wheel(self.scroll_dir)
            self.next_wheel = max(self.next_wheel + SCROLL_PERIOD_S, t)

    # --- presses ---

    def _on_event(self, e: BlinkEvent) -> None:
        if e.kind == "pause":
            self.toggle_pause()
            return
        if self.paused:
            return
        if self.mode is Mode.SCROLLING:  # a long blink ends the scroll; it isn't a click
            self.mode = Mode.NORMAL
            return
        aim = self.history.at(e.aim_time)
        if aim is None:  # looking away, or no face at the time
            return
        if self.mode is not Mode.DRAGGING and self.strip_history.at(e.aim_time):
            button = self.layout.button_at(aim)
            if button is not None:
                self._press_button(button)
                return
            if self.layout.panel.contains(aim):  # a gap in the panel: never click what's behind it
                return
        self._press_page(e.kind, aim)

    def _press_page(self, kind, aim) -> None:
        if self.mode is Mode.DRAGGING:
            self._move_exact(aim)
            self.act.button_up("left")
            self.mode = Mode.NORMAL
        elif self.mode is Mode.SELECT_START:
            self._move_exact(aim)
            self.act.button_down("left")
            self.mode = Mode.DRAGGING
        elif self.mode is Mode.RIGHT_ONCE:
            self._move_exact(self._aim(aim).point)
            self.act.click("right")
            self.mode = Mode.NORMAL
        elif kind == "double":
            self._move_exact(self._aim(aim).point)
            self.act.double_click()
        else:
            self._move_exact(self._aim(aim).point)
            self.act.click("left")

    def _press_button(self, button: StripButton) -> None:
        if button is StripButton.RIGHT_CLICK:
            self.mode = Mode.RIGHT_ONCE
        elif button is StripButton.SELECT:
            self.mode = Mode.SELECT_START
        elif button in (StripButton.SCROLL_UP, StripButton.SCROLL_DOWN):
            self._move_exact(self.last_page_point)
            self.scroll_dir = 1 if button is StripButton.SCROLL_UP else -1
            self.scroll_rect = self.layout.buttons[button].expanded(SCROLL_KEEP_PAD * self.w)
            self.next_wheel = self.t
            self.mode = Mode.SCROLLING
        elif button is StripButton.PAUSE:
            self.toggle_pause()
        elif button is StripButton.KEYBOARD:
            self._say("Keyboard comes in a later version")

    def _say(self, text: str) -> None:
        self.message, self.message_until = text, self.t + MESSAGE_S
