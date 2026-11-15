"""Download the MediaPipe face model into src/gazemouse/models/ (skipped if already there)."""
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
DEST = Path(__file__).resolve().parent.parent / "src" / "gazemouse" / "models" / "face_landmarker.task"


def main():
    if DEST.exists():
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(DEST)
    print(f"downloaded {DEST.stat().st_size} bytes to {DEST}")


if __name__ == "__main__":
    main()
