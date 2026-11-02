"""Wires it all together: camera -> gaze -> controller -> mouse + overlay, with a tray icon.

    python -m gazemouse [--profile NAME] [--calibrate] [--camera INDEX]
"""
import argparse
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .actions import SendInputActuator, primary_screen_size, set_dpi_awareness
from .blink import is_closed
from .calibration import CalibrationResult
from .calibration_ui import GOOD_PCT, RESULT_S, CalibrationWindow
from .camera import slow_or_small
from .controller import Controller, OverlayState
from .features import extract_features
from .filters import OneEuro
from .gaze_model import GazeModel
from .landmarks import MODEL_PATH
from .overlay import Overlay
from .profiles import load_profile, save_profile, valid_name
from .snap import SNAP_RADIUS, BackgroundTargets, UiaTargetProvider

MAX_ATTEMPTS = 3
MAX_PCT = 12.0   # worse than this, the cursor would click the wrong things: never take control
WATCHDOG_MS = 250
NO_FRAMES_S = 1.0
WARNING_S = 8.0
CAMERA_LOST = "Camera stopped. Trying to reconnect…"
TOO_WEAK_MESSAGE = "Calibration failed: accuracy was too low to use safely. Add light in front of you and try again."
NO_EYES_MESSAGE = "Calibration failed: the camera couldn't see your eyes. Add light in front of you and try again."


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="gazemouse", description="Control the computer with your eyes.")
    ap.add_argument("--profile", default="default", help="calibration profile name")
    ap.add_argument("--calibrate", action="store_true", help="calibrate even if a profile exists")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: first that works)")
    args = ap.parse_args(argv)
    if not valid_name(args.profile):
        ap.error(f"invalid profile name: {args.profile!r}")
    return args


def is_better(result: CalibrationResult, best: CalibrationResult | None) -> bool:
    return result.ok and result.error_pct <= MAX_PCT and (best is None or result.error_pct < best.error_pct)


def _tray_icon() -> QIcon:
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(40, 140, 255))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, 28, 28)
    p.setBrush(QColor(255, 255, 255))
    p.drawEllipse(10, 10, 12, 12)
    p.end()
    return QIcon(pix)


class GazeApp(QObject):
    """Tracker, actuator and provider can be replaced, so the whole loop runs in tests."""

    def __init__(self, args, screen_w: int, screen_h: int, tracker=None, actuator=None, provider=None,
                 result_s: float = RESULT_S):
        super().__init__()
        self.args, self.w, self.h, self.result_s = args, screen_w, screen_h, result_s
        self.interactive = tracker is None
        if tracker is None:
            from .tracker import TrackerThread
            tracker = TrackerThread(args.camera)
        if hasattr(tracker, "ready"):
            tracker.ready.connect(self.on_frame)
            tracker.camera_info.connect(self.on_camera_info)
            tracker.failed.connect(self.on_tracker_failed)
        self.tracker = tracker
        self.actuator = actuator or SendInputActuator(screen_w, screen_h)
        self.targets = BackgroundTargets(provider or UiaTargetProvider(screen_w, screen_h), SNAP_RADIUS * screen_w)
        self.overlay = Overlay()
        self.overlay.cover_primary()
        self.filter = OneEuro()
        self.model: GazeModel | None = None
        self.controller: Controller | None = None
        self.calib: CalibrationWindow | None = None
        self.state: OverlayState | None = None
        self.camera_info = None
        self.best, self.attempts = None, 0
        self.warning = ""
        self.exit_code = None
        self.last_frame = None
        self.timer = QTimer(self)  # the controller's timers only run on frames, so something must run without them
        self.timer.timeout.connect(lambda: self.watchdog(time.perf_counter()))
        self.tray = self._make_tray() if QSystemTrayIcon.isSystemTrayAvailable() else None

    def _make_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(_tray_icon())
        tray.setToolTip("Gaze Mouse")
        self.menu = QMenu()
        self.menu.addAction("Recalibrate", self.recalibrate)
        self.menu.addAction("Pause / Resume", self.toggle_pause)
        ring = QAction("Show gaze ring", self.menu, checkable=True, checked=True)
        ring.toggled.connect(self.overlay.set_ring_visible)
        self.menu.addAction(ring)
        self.menu.addSeparator()
        self.menu.addAction("Quit", self.quit)
        tray.setContextMenu(self.menu)
        tray.show()
        return tray

    def start(self) -> None:
        loaded = None if self.args.calibrate else load_profile(self.args.profile)
        if loaded is not None and loaded[1].get("screen") == [self.w, self.h]:  # gaze maps to this exact screen
            self.model = loaded[0]
            self._start_control()
        else:
            self.begin_calibration()
        self.tracker.start()
        if self.interactive:
            self.timer.start(WATCHDOG_MS)

    # --- calibration ---

    def begin_calibration(self) -> None:
        self.best, self.attempts = None, 0
        self._open_calibration()

    def _open_calibration(self) -> None:
        self.overlay.hide()
        self.calib = CalibrationWindow(self.w, self.h, self.result_s)
        self.calib.finished.connect(self.on_calibrated)
        self.calib.show_full_screen()

    def recalibrate(self) -> None:
        if self.calib is None:
            if self.controller is not None:
                self.controller.release()
            self.begin_calibration()

    def on_calibrated(self, result: CalibrationResult) -> None:
        if self.calib is not None:
            self.calib.close()
            self.calib = None
        self.attempts += 1
        if is_better(result, self.best):
            self.best = result
        cancelled = result.reason == "cancelled"
        good = result.ok and result.error_pct <= GOOD_PCT
        if not cancelled and not good and self.attempts < MAX_ATTEMPTS:
            self._open_calibration()
            return
        if self.best is not None:
            self._adopt(self.best)
        elif self.model is not None:
            self._start_control()  # recalibration didn't work out: keep the old model
        elif cancelled:
            self.quit(0)
        else:
            self._fail(TOO_WEAK_MESSAGE if result.ok else NO_EYES_MESSAGE)

    def _adopt(self, result: CalibrationResult) -> None:
        self.model = result.model
        self.warning = ""
        if result.error_pct > GOOD_PCT:
            self.warning = f"Accuracy is weak ({result.error_pct:.1f}% of screen width). Try better light."
        if not self._saved_is_better(result):
            camera = asdict(self.camera_info) if self.camera_info else None
            try:
                save_profile(self.args.profile, result.model,
                             {"error_pct": result.error_pct, "camera": camera, "screen": [self.w, self.h],
                              "date": datetime.now().isoformat(timespec="seconds")})
            except (OSError, ValueError) as e:  # the calibration is still good for this session
                self.warning = f"{self.warning} Couldn't save your calibration: {e}".strip()
        if self.tray is not None:
            self.tray.setToolTip(self.warning or f"Gaze Mouse: {result.error_pct:.1f}% accuracy")
        self._start_control()
        if self.warning:
            self.controller.say(self.warning, WARNING_S)

    def _saved_is_better(self, result: CalibrationResult) -> bool:
        """A weak result is fine for now, but shouldn't replace a better profile for next time."""
        saved = load_profile(self.args.profile)
        if result.error_pct <= GOOD_PCT or saved is None or saved[1].get("screen") != [self.w, self.h]:
            return False
        return saved[1].get("error_pct", float("inf")) <= result.error_pct

    # --- control ---

    def _start_control(self) -> None:
        self.controller = Controller(self.w, self.h, self.actuator, self.targets.targets, SNAP_RADIUS * self.w)
        self.filter.reset()
        self.overlay.show()

    def toggle_pause(self) -> None:
        if self.controller is not None:
            self.controller.toggle_pause()

    def on_frame(self) -> None:
        latest = self.tracker.take()
        if latest is not None:
            self.on_observed(*latest)

    def watchdog(self, now: float) -> None:
        """Frames stopped: run the lost-face path anyway, so a held button can't stay held."""
        if self.last_frame is None or now - self.last_frame <= NO_FRAMES_S:
            return
        if self.calib is not None:
            self.calib.on_features(None, now)
        elif self.controller is not None:
            state = self.controller.step(now, None, None)
            self.controller.say(CAMERA_LOST)
            self.state = replace(state, message=CAMERA_LOST)
            self.overlay.show_state(self.state, self.controller.layout)

    def on_observed(self, obs, t: float) -> None:
        self.last_frame = t
        feat = extract_features(obs) if obs is not None else None
        if self.calib is not None:
            self.calib.on_features(feat, t)
            return
        if self.controller is None:
            return
        gaze = None
        if obs is None:
            closed = None
            self.filter.reset()  # after a gap, start from where the eyes are, not a glide
        else:
            # Both must agree. Laughing or looking down raises the blink score with the lids still apart.
            closed = is_closed(obs.blink_left, obs.blink_right) and feat is None
            if feat is not None:
                gaze = tuple(float(v) for v in self.filter(self.model.predict(feat), t))
                self.targets.set_point(gaze)
        self.state = self.controller.step(t, gaze, closed)
        self.overlay.show_state(self.state, self.controller.layout)

    def on_camera_info(self, info) -> None:
        self.camera_info = info
        for problem in slow_or_small(info):
            print("Warning:", problem, file=sys.stderr)

    def on_tracker_failed(self, message: str) -> None:
        self._fail(message)

    def _fail(self, message: str) -> None:
        self.warning = message
        print(message, file=sys.stderr)
        if self.calib is not None:  # always on top: it would hide the dialog and block Esc
            self.calib.close()
            self.calib = None
        if self.interactive:
            self._show_error(message)
        self.quit(1)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(None, "Gaze Mouse", message)

    def quit(self, code: int = 0) -> None:
        if self.exit_code is not None:
            return
        self.exit_code = code
        self.timer.stop()
        if self.controller is not None:
            self.controller.release()
        if self.calib is not None:
            self.calib.close()
        self.targets.stop()
        self.tracker.stop()
        self.overlay.close()
        if self.tray is not None:
            self.tray.hide()
        QApplication.exit(code)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not MODEL_PATH.exists():
        print("Face model missing. Run: python scripts/fetch_model.py", file=sys.stderr)
        return 1
    set_dpi_awareness()  # before anything creates a window
    w, h = primary_screen_size()
    qt = QApplication(sys.argv[:1])
    qt.setQuitOnLastWindowClosed(False)
    app = GazeApp(args, w, h)
    app.start()
    code = qt.exec()
    return app.exit_code if app.exit_code is not None else code
