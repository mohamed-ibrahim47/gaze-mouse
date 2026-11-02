"""Is this webcam good enough? Prints its size, speed and how often it sees your eyes.

    python -m gazemouse.tools.camcheck [--camera INDEX]
"""
import argparse

from ..camera import Camera, slow_or_small
from ..features import extract_features
from ..landmarks import MODEL_PATH, FaceTracker

FRAMES = 60


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        camera = Camera(args.camera)
    except RuntimeError as e:
        print(e)
        return 1
    tracker = FaceTracker(MODEL_PATH)
    print(f"Look at the screen for a few seconds ({FRAMES} frames)...")
    eyes, done = 0, 0
    try:
        while done < FRAMES:
            frame = camera.read()
            if frame is None:
                continue
            done += 1
            obs = tracker.process(*frame)
            if obs is not None and extract_features(obs) is not None:
                eyes += 1
        info = camera.info()
    finally:
        camera.close()
    print(f"camera: {info.width}x{info.height} at {info.fps:.0f} fps (including face tracking)")
    print(f"eyes seen in {eyes / FRAMES:.0%} of frames")
    for problem in slow_or_small(info):
        print("WARNING:", problem)
    if eyes / FRAMES < 0.9:
        print("WARNING: eyes seen in under 90% of frames. Add light in front of you and face the camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
