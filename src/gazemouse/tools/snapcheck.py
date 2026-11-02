"""Print the snap targets around the real mouse cursor once a second.

    python -m gazemouse.tools.snapcheck [--count N]
"""
import argparse
import ctypes
import time
from ctypes import wintypes

from ..actions import primary_screen_size, set_dpi_awareness
from ..snap import SNAP_RADIUS, UiaTargetProvider, snap


def cursor() -> tuple[int, int]:
    p = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, default=0, help="stop after N queries (0 = run until Ctrl+C)")
    args = ap.parse_args(argv)
    set_dpi_awareness()
    w, h = primary_screen_size()
    radius = SNAP_RADIUS * w
    provider = UiaTargetProvider(w, h)
    print(f"screen {w}x{h}, snap radius {radius:.0f} px. Hover over links and buttons; Ctrl+C to stop.")
    n = 0
    with provider.thread_context():
        while not args.count or n < args.count:
            pt = cursor()
            t0 = time.perf_counter()
            rects = provider.targets_near(pt, radius)
            ms = (time.perf_counter() - t0) * 1000
            s = snap(pt, rects, radius)
            print(f"cursor {pt}: {len(rects)} targets in {ms:.0f} ms; snaps to "
                  f"{tuple(round(v) for v in s.point) if s.rect else 'nothing'}")
            for r in rects:
                print(f"    {r.x:.0f},{r.y:.0f} {r.w:.0f}x{r.h:.0f}")
            n += 1
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
