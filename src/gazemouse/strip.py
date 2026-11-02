"""The hidden right-edge strip: where its buttons go, and when it slides out."""
from enum import Enum

from .geometry import Rect

# Spec 5.4, as fractions of the screen so every resolution gets the same layout.
EDGE_ZONE_X = 0.95         # gaze at x >= 0.95 W counts as "at the edge" (gaze error is ~5% of W)
EDGE_ZONE_OVERSHOOT = 0.05  # ...up to 5% past the edge; further out is looking off the screen
EDGE_LINE_MIN_PX, EDGE_LINE_W = 6, 0.006
BUTTON_W, MARGIN_R = 0.12, 0.02
BUTTONS_TOP, BUTTONS_BOTTOM, GAP = 0.08, 0.88, 0.015
PANEL_PAD = 0.02
REVEAL_S = 0.3
HIDE_S = 0.4
HIDE_MARGIN = 0.02         # gaze must be this far left of the panel for the hide timer to run
EPS = 1e-9


class StripButton(Enum):  # top to bottom
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    RIGHT_CLICK = "right_click"
    SELECT = "select"
    KEYBOARD = "keyboard"
    PAUSE = "pause"


class StripLayout:
    def __init__(self, screen_w: int, screen_h: int):
        w, h = screen_w, screen_h
        self.screen_w = w
        self.edge_zone = Rect(EDGE_ZONE_X * w, 0, (1 - EDGE_ZONE_X + EDGE_ZONE_OVERSHOOT) * w, h)
        line_w = max(EDGE_LINE_MIN_PX, EDGE_LINE_W * w)
        self.edge_line = Rect(w - line_w, 0, line_w, h)
        n = len(StripButton)
        bh = ((BUTTONS_BOTTOM - BUTTONS_TOP) - GAP * (n - 1)) * h / n
        bx = w - (MARGIN_R + BUTTON_W) * w
        self.buttons = {b: Rect(bx, BUTTONS_TOP * h + i * (bh + GAP * h), BUTTON_W * w, bh)
                        for i, b in enumerate(StripButton)}
        first, last = self.buttons[StripButton.SCROLL_UP], self.buttons[StripButton.PAUSE]
        self.panel = Rect(bx, first.y, BUTTON_W * w, last.y + last.h - first.y).expanded(PANEL_PAD * w)

    def button_at(self, pt) -> StripButton | None:
        return next((b for b, r in self.buttons.items() if r.contains(pt)), None)


class StripReveal:
    """Shows the strip after a deliberate look at the edge; hides it once the user looks back."""

    def __init__(self, layout: StripLayout):
        self.layout = layout
        self.last_t = None
        self.hide()

    def hide(self) -> None:
        self.visible = False
        self.in_zone_s = None  # how long the gaze has been in the edge zone (None: not there)
        self.away_s = None     # how long the gaze has been back on the page while visible

    def update(self, t: float, gaze: tuple[float, float] | None) -> bool:
        dt = t - self.last_t if self.last_t is not None else 0.0
        self.last_t = t
        if gaze is None:  # no face: timers freeze, nothing changes
            return self.visible
        L = self.layout
        if not self.visible:
            in_zone = L.edge_zone.contains(gaze)
            self.in_zone_s = (self.in_zone_s + dt if self.in_zone_s is not None else 0.0) if in_zone else None
            if in_zone and self.in_zone_s >= REVEAL_S - EPS:
                self.visible, self.away_s = True, None
        else:
            away = gaze[0] < L.panel.x - HIDE_MARGIN * L.screen_w
            self.away_s = (self.away_s + dt if self.away_s is not None else 0.0) if away else None
            if away and self.away_s >= HIDE_S - EPS:
                self.hide()
        return self.visible
