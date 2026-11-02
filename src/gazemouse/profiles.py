"""Per-user calibration profiles, saved as JSON under the app's home folder."""
import json
import os
import tempfile
from pathlib import Path

from .gaze_model import GazeModel

VERSION = 1
MAX_NAME = 64
# Windows device names can't be file names, whatever the extension.
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def gazemouse_home() -> Path:
    """GAZEMOUSE_HOME if set (dev box: keeps everything off C:), else %APPDATA%/GazeMouse."""
    env = os.environ.get("GAZEMOUSE_HOME")
    home = Path(env) if env else Path(os.environ["APPDATA"]) / "GazeMouse"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _valid_name(name: str) -> bool:
    # Letters in any script (Arabic names too), digits, space, _ and -. Nothing that can leave the folder.
    return (0 < len(name) <= MAX_NAME and name == name.strip()
            and all(c.isalnum() or c in " _-" for c in name) and name.upper() not in RESERVED)


def _path(name: str) -> Path:
    return gazemouse_home() / "profiles" / f"{name}.json"


def save_profile(name: str, model: GazeModel, meta: dict) -> Path:
    if not _valid_name(name):
        raise ValueError(f"invalid profile name: {name!r}")
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"version": VERSION, "model": model.to_dict(), "meta": meta}, indent=1)
    # Write a temp file and swap it in, so a crash mid-write never leaves a half profile.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_profile(name: str) -> tuple[GazeModel, dict] | None:
    """The saved model and meta, or None if missing or unusable (the caller recalibrates)."""
    if not _valid_name(name):
        return None
    try:
        data = json.loads(_path(name).read_text(encoding="utf-8"))
        if data.get("version") != VERSION:
            return None
        return GazeModel.from_dict(data["model"]), data["meta"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
