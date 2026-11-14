"""Frozen entry point for GazeMouse.exe (no console window).

A windowed exe has no stderr, so warnings go to gazemouse.log in the app's home folder.
"""
import sys

from gazemouse.profiles import gazemouse_home

if sys.stderr is None or sys.stdout is None:
    log = open(gazemouse_home() / "gazemouse.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log

from gazemouse.app import main  # noqa: E402

sys.exit(main())
