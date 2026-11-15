"""Screen rectangles, in physical pixels."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, pt) -> bool:
        return self.x <= pt[0] <= self.x + self.w and self.y <= pt[1] <= self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def distance_to(self, pt) -> float:
        """0 inside, else the distance to the nearest edge."""
        dx = max(self.x - pt[0], 0, pt[0] - (self.x + self.w))
        dy = max(self.y - pt[1], 0, pt[1] - (self.y + self.h))
        return math.hypot(dx, dy)

    def expanded(self, d: float) -> "Rect":
        return Rect(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)
