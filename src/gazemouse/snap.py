"""Snapping the gaze onto clickable things found through Windows UI Automation."""
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass

from .geometry import Rect

SNAP_RADIUS = 0.06  # x screen width: the spec's accuracy target
CLICKABLE = {f"{n}Control" for n in ("Button", "Hyperlink", "Edit", "ListItem", "MenuItem", "TabItem",
                                     "CheckBox", "RadioButton", "ComboBox", "TreeItem", "SplitButton",
                                     "DataItem")}
MAX_PARENTS = 3
MAX_AREA = 0.25  # of the screen: bigger "targets" are panes, not things to click


@dataclass
class SnapResult:
    point: tuple[float, float]
    rect: Rect | None


def snap(point, candidates: list[Rect], radius: float) -> SnapResult:
    """A target containing the point wins (the smallest if nested); else the nearest within radius."""
    inside = [r for r in candidates if r.contains(point)]
    if inside:
        best = min(inside, key=lambda r: r.w * r.h)
    else:
        near = [(r.distance_to(point), r) for r in candidates]
        near = [(d, r) for d, r in near if d <= radius]
        if not near:
            return SnapResult(tuple(point), None)
        best = min(near, key=lambda dr: dr[0])[1]
    return SnapResult(best.center, best)


def _uia_control_from_point(x, y):
    import uiautomation
    return uiautomation.ControlFromPoint(int(x), int(y))


class UiaTargetProvider:
    """Clickable rects around a point. Slow (COM), so it runs on BackgroundTargets' thread."""

    def __init__(self, screen_w: int, screen_h: int, control_from_point=None):
        self.max_area = MAX_AREA * screen_w * screen_h
        self.control_from_point = control_from_point or _uia_control_from_point
        self.own_pid = os.getpid()

    def thread_context(self):
        if self.control_from_point is not _uia_control_from_point:
            return nullcontext()
        import uiautomation
        return uiautomation.UIAutomationInitializerInThread()

    def targets_near(self, point, radius) -> list[Rect]:
        x, y, d = point[0], point[1], radius / 2
        probes = [(x + dx * d, y + dy * d) for dy in (0, -1, 1) for dx in (0, -1, 1)]
        found = []
        try:
            for px, py in probes:
                rect = self._clickable_rect(self.control_from_point(px, py))
                if rect is not None and rect not in found:
                    found.append(rect)
        except Exception:  # COM errors are routine (windows closing, elevated apps): keep what we have
            pass
        return found

    def _clickable_rect(self, c) -> Rect | None:
        for _ in range(MAX_PARENTS + 1):
            if c is None:
                return None
            if c.ControlTypeName in CLICKABLE:
                if c.ProcessId == self.own_pid:
                    return None
                b = c.BoundingRectangle
                rect = Rect(b.left, b.top, b.right - b.left, b.bottom - b.top)
                if rect.w <= 0 or rect.h <= 0 or rect.w * rect.h > self.max_area:
                    return None
                return rect
            c = c.GetParentControl()
        return None


class BackgroundTargets:
    """Keeps a fresh list of snap targets around the latest gaze point, off the GUI thread."""

    def __init__(self, provider, radius: float, hz: float = 10):
        self.provider, self.radius, self.period = provider, radius, 1.0 / hz
        self._lock = threading.Lock()
        self._point, self._targets = None, []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="snap-targets", daemon=True)
        self.thread.start()

    def set_point(self, pt) -> None:
        with self._lock:
            self._point = pt

    def targets(self) -> list[Rect]:
        with self._lock:
            return list(self._targets)

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)

    def _run(self):
        with getattr(self.provider, "thread_context", nullcontext)():
            while not self._stop.is_set():
                with self._lock:
                    pt = self._point
                if pt is not None:
                    try:
                        found = self.provider.targets_near(pt, self.radius)
                    except Exception:
                        found = []
                    with self._lock:
                        self._targets = found
                self._stop.wait(self.period)
