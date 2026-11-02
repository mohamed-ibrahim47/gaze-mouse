"""Eyes-closed signal -> click / double / pause events, and the recent gaze to aim them at."""
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# Spec 5.3. Module-level so Plan 2 can make them per-user.
LONG_MIN_S = 0.4       # shorter closures are natural blinks (~0.1-0.15 s)
LONG_MAX_S = 1.5       # 1.5..2.0 s is ignored: neither a click nor a pause
PAUSE_MIN_S = 2.0
DOUBLE_WINDOW_S = 0.6  # a second long blink must start this soon after the first reopens
AIM_BACK_S = 0.2       # aim at the gaze from just before the eyes started closing
OPEN_CONFIRM_S = 0.08  # one open frame inside a closure is noise; this long open ends it
MIN_CLOSED_FRAMES = 3  # fewer closed frames than this isn't evidence of a deliberate blink
MAX_FRAME_GAP_S = 0.25 # a camera stall inside a closure makes its length meaningless
EPS = 1e-9


def is_closed(blink_left: float, blink_right: float, threshold: float = 0.5) -> bool:
    """Both eyes closed. A wink or a squint (one eye, or both half shut) is not."""
    return blink_left >= threshold and blink_right >= threshold


@dataclass
class BlinkEvent:
    kind: Literal["click", "double", "pause"]
    aim_time: float


@runtime_checkable
class PressTrigger(Protocol):
    """Anything that turns a per-frame signal into press events (blinks now; winks or switches later)."""

    def update(self, t: float, closed: bool) -> list[BlinkEvent]: ...

    def cancel(self) -> None:
        """Forget a press in progress (the signal was lost mid-press)."""
        ...


class BlinkDetector:
    def __init__(self):
        self.closed_since = None  # start time of the current closure
        self.open_since = None    # first open frame after it, until OPEN_CONFIRM_S confirms the end
        self.closed_frames = 0
        self.stalled = False      # the camera stalled during this closure
        self.need_open = False    # after cancel(): the eyes must be seen open before the next closure
        self.last_t = None
        self.pending = None       # (aim_time, deadline) of a click waiting out the double window

    def update(self, t: float, closed: bool) -> list[BlinkEvent]:
        events = []
        if self.closed_since is not None and self.last_t is not None and t - self.last_t > MAX_FRAME_GAP_S:
            self.stalled = True
        self.last_t = t
        self._flush(t, events)
        if closed:
            if not self.need_open:
                if self.closed_since is None:
                    self.closed_since, self.closed_frames, self.stalled = t, 0, False
                self.closed_frames += 1
                self.open_since = None
        else:
            self.need_open = False
            if self.closed_since is not None:
                if self.open_since is None:
                    self.open_since = t
                if t - self.open_since >= OPEN_CONFIRM_S - EPS:
                    self._finish_closure(t, events)
        return events

    def _finish_closure(self, t, events):
        start, reopened = self.closed_since, self.open_since
        self.closed_since = self.open_since = None
        if not self.stalled and self.closed_frames >= MIN_CLOSED_FRAMES:
            self._closure_ended(start, reopened - start, reopened, events)
        self._flush(t, events)

    def cancel(self) -> None:
        self.closed_since = self.open_since = None
        self.need_open = True

    def _flush(self, t, events):
        """Emit the pending click once its window has passed, unless a closure is still running
        (it started inside the window and may turn out to be the second half of a double)."""
        if self.pending and self.closed_since is None and t >= self.pending[1] - EPS:
            events.append(BlinkEvent("click", self.pending[0]))
            self.pending = None

    def _closure_ended(self, start, d, t, events):
        is_long = LONG_MIN_S <= d < LONG_MAX_S
        if self.pending and is_long:
            events.append(BlinkEvent("double", self.pending[0]))
            self.pending = None
            return
        # A natural blink inside the window leaves the click pending; _flush fires it on time.
        # A close of 1.5 s or more is something else, so the first blink's click goes out now.
        if self.pending and d >= LONG_MAX_S:
            events.append(BlinkEvent("click", self.pending[0]))
            self.pending = None
        if is_long:
            self.pending = (start - AIM_BACK_S, t + DOUBLE_WINDOW_S)
        elif d >= PAUSE_MIN_S:
            events.append(BlinkEvent("pause", start - AIM_BACK_S))


class GazeHistory:
    """The last few seconds of gaze points, so a blink can click where the user was looking."""

    def __init__(self, keep_s: float = 2.0):
        self.keep_s = keep_s
        self.samples = deque()  # (t, point), oldest first

    def push(self, t: float, point: tuple[float, float]) -> None:
        self.samples.append((t, point))
        while self.samples and self.samples[0][0] < t - self.keep_s:
            self.samples.popleft()

    def has_gap(self, t0: float, t1: float) -> bool:
        """Was anything between these times a None sample (looking away, no face)?"""
        return any(p is None for t, p in self.samples if t0 - EPS <= t <= t1 + EPS)

    def at(self, t: float) -> tuple[float, float] | None:
        """Latest sample at or before t."""
        i = bisect_right([s[0] for s in self.samples], t + EPS)
        return self.samples[i - 1][1] if i else None
