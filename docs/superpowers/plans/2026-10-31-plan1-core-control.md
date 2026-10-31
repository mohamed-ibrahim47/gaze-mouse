# Gaze Mouse — Plan 1: Core Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Windows app where the laptop webcam moves the real cursor by gaze, long blinks click, and a hidden right-edge strip gives scroll, right click, select/drag and pause.

**Architecture:** A tracker thread reads the camera and runs MediaPipe Face Landmarker. The GUI thread turns each observation into eye features, then a calibrated gaze point, then a smoothed point. A pure `Controller` state machine turns gaze + eyes-closed signals into actuator calls (SendInput) and an `OverlayState`, which a click-through PySide6 overlay draws. Snap targets come from Windows UI Automation on a background thread. Every decision-making unit is pure Python with injected fakes, so it is unit-tested on the dev box. Camera, overlay and SendInput are verified by hand on the laptop.

**Tech Stack:** Python 3.10+, mediapipe 1.2.0 (Tasks API), opencv-python 4.x, numpy, PySide6, uiautomation, pytest. Windows 10/11.

**Spec:** `docs/design.md` (approved 2026-10-31). Executors read it alongside this plan.

**Later plans (not in this one):** Plan 2 precision (2× zoom on ambiguous targets, head fine-aim, drift correction from clicks, quick 5-dot recal, head-movement research with licence check) · Plan 3 English + Arabic eye keyboard · Plan 4 release (camera-not-good-enough path, bilingual app UI, start at logon, installer, GitHub + SourceForge, code signing, privacy page).

## Global Constraints

- Repo: `D:\claude-apps\gaze-mouse` (own git repo). Commit identity per repo: `Mohamed Ibrahim <mohamed.ibrahim0907@gmail.com>`, never the work email. Push only when the user asks.
- Licence: GPL-3.0. Everything shipped must be redistributable under an open-source licence.
- **Nothing on C: on the dev box:** venv at `D:\claude-apps\gaze-mouse\.venv`; before pip/pytest export `TEMP='D:\dev\tmp' TMP='D:\dev\tmp' PIP_CACHE_DIR='D:\dev\pip-cache' GAZEMOUSE_HOME='D:\dev\gazemouse-home'`.
- Pinned deps: `mediapipe==1.2.0` (legacy `mp.solutions` does not exist), `opencv-python>=4.10,<5` (5.0 wheel cannot open cameras by index), `numpy`, `PySide6`, `uiautomation`; dev: `pytest`.
- Face model file `src/gazemouse/models/face_landmarker.task` is downloaded by `scripts/fetch_model.py` from `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` and git-ignored.
- All internal geometry is **physical pixels** of the primary monitor; the process is per-monitor-DPI aware. Convert to Qt logical pixels only when painting.
- Real-camera testing happens **on the laptop, not over Remote Desktop** (the forwarded camera is 640×480 and was shown to fail).
- Accuracy target: mean validation error ≤ **6% of screen width**.
- No dwell clicking anywhere. Looking alone never presses anything.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01FJt8oHjokS11Ct5XH6aW3L`

## Review Focus

1. **Face lost mid-action** (user turns away, leaves the frame) while scrolling or dragging → scrolling stops at once; a drag held > 3 s without a face is cancelled (Esc, then button up), never left with the button held. Test in Task 9.
2. **Natural blinks, squints and winks** (talking, laughing, one eye half-shut) → never click. Test in Task 5.
3. **Display scaling other than 100% and other resolutions** (other laptops: 125–200%, 1366×768 up to 3840×2400) → cursor, strip and overlay line up. Tests in Tasks 6, 7 and 10.
4. **Calibration where the face keeps getting lost** (glasses glare, dark room) → reports failure with a reason instead of fitting a model on a few dots. Test in Task 3.
5. **Gaze far outside the screen** (looking at the physical keyboard or away) → the cursor doesn't jump to an edge, the strip doesn't open, and blinks aimed off-screen don't click. Test in Task 9.

---

## File Structure

```
gaze-mouse/
  pyproject.toml            deps, pytest config
  LICENSE                   GPL-3.0 text
  README.md                 what it is, dev setup, how to run
  .gitignore                .venv, __pycache__, *.task, runs/
  docs/design.md            approved spec
  scripts/fetch_model.py    downloads face_landmarker.task
  src/gazemouse/
    __init__.py
    __main__.py             python -m gazemouse -> app.main()
    features.py             landmarks -> 8-value eye feature vector
    gaze_model.py           ridge regression features -> screen point
    filters.py              One Euro smoothing
    calibration.py          calibration session state machine (pure)
    profiles.py             save/load per-user calibration (JSON)
    camera.py               webcam open/read/probe
    landmarks.py            MediaPipe wrapper -> FaceObservation
    blink.py                eyes-closed -> click/double/pause events; GazeHistory
    geometry.py             Rect
    strip.py                edge-strip layout + reveal/hide timing
    actions.py              SendInput mouse/keyboard (Actuator)
    snap.py                 snap resolver + UIA target provider + background thread
    controller.py           the brain: modes, strip, clicks, scroll, pause
    overlay.py              click-through PySide6 overlay
    calibration_ui.py       full-screen calibration window
    tracker.py              QThread: camera + landmarks -> signal
    app.py                  wiring, tray, CLI
    tools/camcheck.py       prints camera info + face detection rate
    tools/snapcheck.py      prints snap targets around the mouse cursor
  tests/                    one test file per pure module
```

---

### Task 1: Repo scaffold + eye features

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `README.md`, `.gitignore`, `scripts/fetch_model.py`, `src/gazemouse/__init__.py`, `src/gazemouse/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `FEATURES: tuple[str, ...] = ("iris_x", "iris_y_corner", "iris_y_lid", "openness", "head_fx", "head_fy", "head_tx", "head_ty")`; `FaceObservation` dataclass (`points: np.ndarray` shape (478, 2) in camera pixels, `blink_left: float`, `blink_right: float`, `head: np.ndarray` shape (4, 4), `t: float` seconds); `extract_features(obs: FaceObservation, blink_ratio: float = 0.18) -> np.ndarray | None` returning shape (8,) in `FEATURES` order, `None` when either eye's openness < `blink_ratio`.
- Landmark indices (MediaPipe refined mesh): right eye iris 468, outer 33, inner 133, upper lid 159, lower lid 145; left eye iris 473, inner 362, outer 263, upper 386, lower 374.

- [ ] **Step 1: Scaffold.** `git init`, set the per-repo identity (Global Constraints), add `.gitignore`, the GPL-3.0 `LICENSE`, a short `README.md` (one paragraph from spec §1, dev setup commands from Global Constraints, `python scripts/fetch_model.py`, `python -m gazemouse`). `pyproject.toml`: src layout, package `gazemouse`, deps from Global Constraints, `[tool.pytest.ini_options] testpaths=["tests"]`, marker `model` ("needs face_landmarker.task"). Create the venv on D: and `pip install -e .[dev]`. Write `scripts/fetch_model.py` (urllib download to `src/gazemouse/models/face_landmarker.task`, skip if present) and run it.

- [ ] **Step 2: Write the failing tests** in `tests/test_features.py`. Build a synthetic `FaceObservation`: zeros (478, 2) with eye points set. Right eye (image-left): outer (100, 200), inner (160, 200), lids (130, 190)/(130, 210). Left eye (image-right): inner (240, 200), outer (300, 200), lids (270, 190)/(270, 210). `head = np.eye(4)`.

```python
def test_centered_iris_gives_half(): # irises at (130,200) and (270,200)
    f = extract_features(obs)
    assert f.shape == (8,)
    assert f[0] == pytest.approx(0.5, abs=1e-6)

def test_both_eyes_agree_on_horizontal(): # both irises +6 px toward image-right
    assert extract_features(shifted_right)[0] > 0.5
    # left eye x increases, right eye x decreases; the flip makes them agree

def test_vertical_sign_same_for_both_eyes(): # both irises +4 px down
    f = extract_features(shifted_down)
    assert f[1] == pytest.approx(4 / 60, abs=1e-6)  # regression: spike averaged opposite signs to ~0

def test_closed_eyes_return_none(): # lids 2 px apart -> openness 2/60 < 0.18
    assert extract_features(closed) is None
```

- [ ] **Step 3: Run** `.venv/Scripts/python -m pytest tests/test_features.py -v`. Expected: FAIL (module missing).

- [ ] **Step 4: Implement `extract_features` in `src/gazemouse/features.py`.** Per eye: corner axis inner→outer, `ux` unit vector, `uy` its perpendicular flipped so `uy[1] >= 0` (same "down" for both eyes); `x = (iris-inner)·ux / width`, `y_corner = (iris-inner)·uy / width`, `y_lid = (iris - lid_midpoint)·uy / width`, `openness = |upper-lower| / width`. Combine: `iris_x = (left.x + (1 - right.x)) / 2`, the other three are plain averages of both eyes. Head: `head[0,2], head[1,2], head[0,3], head[1,3]`.

- [ ] **Step 5: Run tests.** Expected: 4 PASS.

- [ ] **Step 6: Commit** `feat: scaffold repo and eye feature extraction`.

---

### Task 2: Gaze model + smoothing

**Files:**
- Create: `src/gazemouse/gaze_model.py`, `src/gazemouse/filters.py`
- Test: `tests/test_gaze_model.py`, `tests/test_filters.py`

**Interfaces:**
- Consumes: feature vectors of shape (8,) (Task 1).
- Produces: `class GazeModel` with `fit(feats: np.ndarray (N,8), targets: np.ndarray (N,2), lam: float = 1e-3) -> None` (raises `ValueError` if N < 10), `predict(feat: np.ndarray (8,)) -> np.ndarray (2,)`, `to_dict() -> dict`, `GazeModel.from_dict(d: dict) -> GazeModel`. `class OneEuro(min_cutoff: float = 0.6, beta: float = 0.004)` with `__call__(x: np.ndarray (2,), t: float) -> np.ndarray (2,)` and `reset()`.
- Model form (spike's best, "x + all three"): standardise columns 0–3 (iris_x, iris_y_corner, iris_y_lid, openness) with the training mean/std, add an intercept, ridge with `lam` on everything except the intercept. Head columns are not used.

- [ ] **Step 1: Write the failing tests.**

```python
def test_recovers_linear_mapping():  # 300 samples, targets linear in cols 0,1 + noise 0.002
    m = GazeModel(); m.fit(F, T)
    err = np.linalg.norm(np.array([m.predict(f) for f in F]) - T, axis=1).mean()
    assert err < 25  # px, on a 2560x1600 target range

def test_roundtrip_dict_same_predictions():
    m2 = GazeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(F[0]), m.predict(F[0]))

def test_too_few_samples():
    with pytest.raises(ValueError): GazeModel().fit(F[:5], T[:5])

def test_one_euro_still_input_settles():  # constant input -> output equals input
def test_one_euro_reduces_jitter():       # +-20 px noise around a point: output std < 0.5 * input std
def test_one_euro_follows_jump():         # step of 800 px: within 0.3 s output is > 600 px along
```

- [ ] **Step 2: Run** both test files. Expected: FAIL.
- [ ] **Step 3: Implement** both modules (One Euro as in the spike: derivative cutoff 1.0 Hz; cutoff = `min_cutoff + beta * |dx|`).
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: gaze regression model and One Euro filter`.

---

### Task 3: Calibration session

**Files:**
- Create: `src/gazemouse/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `GazeModel` (Task 2).
- Produces: `CalibStep` dataclass (`phase: Literal["calib", "valid", "done"]`, `target: tuple[int, int] | None`, `progress: float` 0..1 within the current dot); `CalibrationResult` dataclass (`ok: bool`, `reason: str | None` — `"face_lost"` or `None`, `model: GazeModel | None`, `mean_error_px: float | None`, `error_pct: float | None`, `calib_dots: int`, `valid_dots: int`); `class CalibrationSession(screen_w: int, screen_h: int, settle_s: float = 0.8, sample_s: float = 1.0)` with `start(t: float)`, `update(t: float, feat: np.ndarray | None) -> CalibStep`, `result() -> CalibrationResult` (valid once phase is `"done"`).
- Values: calibration dots at x ∈ {0.06, 0.35, 0.65, 0.94}·W × y ∈ {0.08, 0.36, 0.64, 0.92}·H, row by row (16); validation dots at x ∈ {0.22, 0.78} × y ∈ {0.25, 0.75} plus centre (5). Per dot, samples are taken only when `settle_s < elapsed ≤ settle_s + sample_s`. A dot counts if it has ≥ 5 samples; keep samples within 0.05 of the dot's median in (iris_x, iris_y_corner). Fit on all kept calibration samples; validation uses the mean of each dot's kept samples. `ok` requires ≥ 12 calibration dots and ≥ 4 validation dots. `error_pct = mean_error_px / screen_w * 100`.

- [ ] **Step 1: Write the failing tests.** Helper `run(session, feat_for_target)` advances t in 1/30 s steps, calling `update` with `feat_for_target(step.target)` until `phase == "done"`.

```python
def test_clean_data_calibrates_accurately():
    r = run(CalibrationSession(2560, 1600), synthetic_linear_features)
    assert r.ok and r.calib_dots == 16 and r.valid_dots == 5
    assert r.error_pct < 1.0

def test_settle_period_ignored():
    # garbage features during the first 0.8 s of each dot, clean afterwards
    assert run(..., garbage_then_clean).error_pct < 1.0

def test_face_lost_reports_failure():  # Review Focus 4
    r = run(CalibrationSession(2560, 1600), lambda target: None)
    assert not r.ok and r.reason == "face_lost" and r.model is None

def test_mostly_lost_face_fails():  # features only on 8 of 16 calibration dots
    assert run(..., half_missing).reason == "face_lost"

def test_dot_positions_scale_with_screen():
    s = CalibrationSession(1366, 768); s.start(0.0)
    assert s.update(0.0, None).target == (round(0.06 * 1366), round(0.08 * 768))
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement `CalibrationSession`.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: calibration session state machine`.

---

### Task 4: Profiles

**Files:**
- Create: `src/gazemouse/profiles.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `GazeModel.to_dict/from_dict` (Task 2).
- Produces: `gazemouse_home() -> Path` (env `GAZEMOUSE_HOME`, else `%APPDATA%/GazeMouse`; created on demand); `save_profile(name: str, model: GazeModel, meta: dict) -> Path` (writes `profiles/<name>.json` with `{"version": 1, "model": ..., "meta": ...}`); `load_profile(name: str) -> tuple[GazeModel, dict] | None`.

- [ ] **Step 1: Write the failing tests** (use `monkeypatch.setenv("GAZEMOUSE_HOME", str(tmp_path))`): roundtrip gives identical predictions and meta; missing profile → `None`; a file containing `{not json` → `None` (no exception); unknown `version` → `None`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Write to a temp file and then `os.replace` it, so a crash mid-write can't corrupt the profile.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: per-user calibration profiles`.

---

### Task 5: Blink events + gaze history

**Files:**
- Create: `src/gazemouse/blink.py`
- Test: `tests/test_blink.py`

**Interfaces:**
- Produces: `is_closed(blink_left: float, blink_right: float, threshold: float = 0.5) -> bool` (both eyes ≥ threshold); `BlinkEvent` dataclass (`kind: Literal["click", "double", "pause"]`, `aim_time: float`); `class PressTrigger(Protocol)` with `update(t: float, closed: bool) -> list[BlinkEvent]` (spec §6 "pluggable press triggers": Plan 2+ can add wink or hardware-switch triggers behind it); `class BlinkDetector` implementing `PressTrigger`; `class GazeHistory(keep_s: float = 2.0)` with `push(t: float, point: tuple[float, float]) -> None` and `at(t: float) -> tuple[float, float] | None` (latest sample at or before `t`).
- Values (spec §5.3): closure < 0.4 s ignored (natural ~0.1–0.15 s); 0.4 ≤ d < 1.5 s is a long blink; 1.5 ≤ d < 2.0 ignored; d ≥ 2.0 s gives `"pause"` on reopen. A single long blink emits `"click"` 0.6 s after reopening, unless a second long blink *starts* within those 0.6 s, which emits one `"double"` when the second ends (and no `"click"`). `aim_time = start_of_first_closure - 0.2`. Constants are module-level names so Plan 2 can make them per-user.

- [ ] **Step 1: Write the failing tests.** Helper `feed(det, timeline)` takes `[(duration_s, closed), ...]` at 60 Hz and returns a list of `Emitted(kind, aim_time, emitted_at)` namedtuples, one per event, where `emitted_at` is the `t` of the `update` call that returned it. `kinds(...)` maps that list to the kinds.

```python
def test_natural_blinks_never_click():  # Review Focus 2
    assert feed(d, [(0.5, False), (0.12, True)] * 20) == []

def test_long_blink_clicks_after_double_window():
    ev = feed(d, [(1, False), (0.5, True), (1, False)])
    assert [e.kind for e in ev] == ["click"]
    assert ev[0].aim_time == pytest.approx(1.0 - 0.2, abs=0.02)
    assert ev[0].emitted_at == pytest.approx(1.5 + 0.6, abs=0.02)

def test_two_long_blinks_make_one_double():
    assert kinds(feed(d, [(1, False), (0.5, True), (0.3, False), (0.5, True), (1, False)])) == ["double"]

def test_pause_on_two_second_close():
    assert kinds(feed(d, [(1, False), (2.5, True), (0.2, False)])) == ["pause"]

def test_in_between_closure_ignored():
    assert feed(d, [(1, False), (1.7, True), (1, False)]) == []

def test_wink_and_squint_are_not_closed():  # Review Focus 2
    assert not is_closed(0.9, 0.3)
    assert not is_closed(0.45, 0.45)
    assert is_closed(0.6, 0.7)

def test_history_returns_sample_before_time():
    h.push(1.0, (10, 10)); h.push(1.1, (20, 20))
    assert h.at(1.05) == (10, 10) and h.at(0.5) is None
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: blink event detector and gaze history`.

---

### Task 6: Geometry + edge strip

**Files:**
- Create: `src/gazemouse/geometry.py`, `src/gazemouse/strip.py`
- Test: `tests/test_strip.py`

**Interfaces:**
- Produces: `Rect(x: float, y: float, w: float, h: float)` frozen dataclass with `contains(pt) -> bool`, `center -> tuple[float, float]`, `distance_to(pt) -> float` (0 inside), `expanded(d: float) -> Rect`. `class StripButton(Enum)`: `SCROLL_UP, SCROLL_DOWN, RIGHT_CLICK, SELECT, KEYBOARD, PAUSE` (top to bottom). `class StripLayout(screen_w: int, screen_h: int)` with `edge_zone: Rect`, `edge_line: Rect`, `buttons: dict[StripButton, Rect]`, `panel: Rect`, `button_at(pt) -> StripButton | None`. `class StripReveal(layout: StripLayout)` with `update(t: float, gaze: tuple[float, float] | None) -> bool` (visible) and `hide()`.
- Values: `edge_line` = rightmost `max(6, 0.006·W)` px, full height. `edge_zone` = x ≥ 0.95·W (wider than the drawn line, because gaze error is ~5% of width). Buttons: width 0.12·W, right margin 0.02·W, six stacked between y = 0.08·H and 0.88·H with gap 0.015·H, equal heights. `panel` = union of the buttons expanded by 0.02·W. Reveal when gaze stays in `edge_zone` for ≥ 0.3 s. While visible, stay visible while gaze is inside `panel` or `edge_zone`; hide after gaze has been at x < `panel.x - 0.02·W` for ≥ 0.4 s. `gaze=None` freezes the timers.

- [ ] **Step 1: Write the failing tests.**

```python
def test_reveal_needs_0_3_s(): # gaze in edge zone: 0.2 s -> False, 0.35 s -> True
def test_reading_near_edge_briefly_does_not_reveal(): # 0.25 s in zone, back to page, again 0.25 s -> False
def test_hides_after_looking_back_0_4_s(): # visible; gaze on page 0.3 s -> True, 0.45 s -> False
def test_stays_visible_over_buttons(): # visible; 3 s over SCROLL_DOWN -> True
@pytest.mark.parametrize("w,h", [(1366, 768), (2560, 1600), (3840, 2400)])  # Review Focus 3
def test_layout_fits_and_buttons_do_not_overlap(w, h):
    L = StripLayout(w, h)
    rects = list(L.buttons.values())
    assert all(0 <= r.x and r.x + r.w <= w and 0 <= r.y and r.y + r.h <= h for r in rects)
    assert all(not overlaps(a, b) for a, b in itertools.combinations(rects, 2))
    assert L.buttons[StripButton.SCROLL_UP].w == pytest.approx(0.12 * w)
def test_button_at_center_and_outside():
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: edge strip layout and reveal timing`.

---

### Task 7: Actions (SendInput)

**Files:**
- Create: `src/gazemouse/actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Produces: `class Actuator(Protocol)`: `move(x: int, y: int)`, `click(button: Literal["left", "right"])`, `double_click()`, `button_down(button)`, `button_up(button)`, `wheel(notches: int)` (positive = up, 120 per notch), `key(vk: int)`. `VK_ESCAPE = 0x1B`. `to_absolute(x: float, y: float, screen_w: int, screen_h: int) -> tuple[int, int]` mapping physical px to SendInput's 0..65535 range: `round(x * 65535 / (screen_w - 1))`, clamped. `class SendInputActuator(screen_w: int, screen_h: int)` implementing `Actuator` with ctypes `SendInput` (`MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE`, LEFT/RIGHT DOWN/UP, `MOUSEEVENTF_WHEEL`, `KEYBDINPUT` down+up). `set_dpi_awareness() -> None` calling `SetProcessDpiAwarenessContext(-4)` (per-monitor v2) with a fallback to `SetProcessDPIAware`; `primary_screen_size() -> tuple[int, int]` (physical, via `GetSystemMetrics` after awareness is set).

- [ ] **Step 1: Write the failing tests** for `to_absolute`: corners map to (0,0) and (65535,65535) at 1366×768, 2560×1600 and 3840×2400 (Review Focus 3); centre maps to ≈ 32768; out-of-range input is clamped. Also a test that `SendInputActuator` builds the `INPUT` struct with the right `sizeof` (40 on 64-bit), so a wrong struct layout fails here instead of silently on the laptop.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS. **Manual:** `python -c "from gazemouse.actions import *; set_dpi_awareness(); w,h=primary_screen_size(); a=SendInputActuator(w,h); a.move(w//2,h//2)"` puts the cursor at the screen centre.
- [ ] **Step 5: Commit** `feat: SendInput actuator with DPI-correct coordinates`.

---

### Task 8: Snap targets

**Files:**
- Create: `src/gazemouse/snap.py`, `src/gazemouse/tools/snapcheck.py`
- Test: `tests/test_snap.py`

**Interfaces:**
- Consumes: `Rect` (Task 6).
- Produces: `SnapResult` dataclass (`point: tuple[float, float]`, `rect: Rect | None`); `snap(point, candidates: list[Rect], radius: float) -> SnapResult` (a rect containing the point wins, the smallest if several; otherwise the rect with the least `distance_to` ≤ radius; the result point is the rect centre; no candidate → original point, `rect=None`). `class UiaTargetProvider` with `targets_near(point, radius) -> list[Rect]`. `class BackgroundTargets(provider, hz: float = 10)` (daemon thread) with `set_point(pt)`, `targets() -> list[Rect]` (latest, thread-safe), `stop()`.
- UIA details: hit-test 9 points (the point plus 8 around it at radius/2) with `uiautomation.ControlFromPoint`. From each control walk up at most 3 parents to the first control whose `ControlTypeName` is one of Button, Hyperlink, Edit, ListItem, MenuItem, TabItem, CheckBox, RadioButton, ComboBox, TreeItem, SplitButton, DataItem. Skip controls owned by our own process id and rects larger than 25% of the screen area. Dedupe by rect. Every UIA call is wrapped in try/except: a COM error returns the targets found so far. Call `uiautomation.UIAutomationInitializerInThread` in the thread.
- Snap radius used by the app: **0.06 · screen_w** (spec accuracy target).

- [ ] **Step 1: Write the failing tests** for `snap`: point inside a rect → its centre; inside two nested rects → the smaller one; two rects 30 px and 80 px away with radius 100 → the 30 px one; nearest at 150 px with radius 100 → unchanged point, `rect is None`; empty list → unchanged. `BackgroundTargets` with a fake provider: after `set_point`, `targets()` returns the fake's rects within 0.5 s; a provider that raises keeps the thread alive and returns `[]`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, plus `tools/snapcheck.py`, which prints the targets around the real mouse cursor once a second.
- [ ] **Step 4: Run.** Expected: PASS. **Manual (dev box is fine):** open Chrome on a page, run `python -m gazemouse.tools.snapcheck`, hover over links and buttons, and check that rects are printed. Note in the commit message how long a query takes.
- [ ] **Step 5: Commit** `feat: snap resolver and UI Automation targets`.

---

### Task 9: Controller

**Files:**
- Create: `src/gazemouse/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: `PressTrigger`, `BlinkDetector`, `BlinkEvent`, `GazeHistory` (Task 5); `StripLayout`, `StripReveal`, `StripButton`, `Rect` (Task 6); `Actuator`, `VK_ESCAPE` (Task 7); `snap` (Task 8).
- Produces: `class Mode(Enum)`: `NORMAL, RIGHT_ONCE, SELECT_START, DRAGGING, SCROLLING`. `OverlayState` dataclass (`gaze: tuple[float, float] | None`, `snapped: Rect | None`, `strip_visible: bool`, `hot_button: StripButton | None`, `mode: Mode`, `paused: bool`, `message: str | None`). `class Controller(screen_w: int, screen_h: int, actuator: Actuator, targets: Callable[[], list[Rect]], snap_radius: float, presses: PressTrigger | None = None)` (default `BlinkDetector()`) with `step(t: float, gaze: tuple[float, float] | None, closed: bool | None) -> OverlayState` and `toggle_pause() -> None`.
- Behaviour (spec §5.3–5.4 and §6):
  - **Off-screen:** a gaze point more than 0.1·W / 0.1·H outside the screen counts as "away": no cursor move, no strip timer, and a blink aimed there does nothing. Otherwise clamp it to the screen.
  - **NORMAL:** if the strip is hidden and gaze is on the page, snap the gaze and `move` to the snapped point (only when it moved ≥ 2 px, to avoid flooding). Remember it as `last_page_point`. While gaze is on the strip, the cursor stays put.
  - **Blink events** (from `BlinkDetector` fed with `closed`): the aim point is `GazeHistory.at(event.aim_time)`. If it lands on a visible strip button, press that button. Otherwise act on the page at the snapped aim point: `click` → left (or right in `RIGHT_ONCE`, then back to NORMAL); `double` → `double_click`; in `SELECT_START` → `move` + `button_down("left")` → `DRAGGING`; in `DRAGGING` → `move` + `button_up("left")` → NORMAL. `pause` → `toggle_pause()`.
  - **Buttons:** RIGHT_CLICK → `RIGHT_ONCE`; SELECT → `SELECT_START`; SCROLL_UP/DOWN → `SCROLLING` with direction; PAUSE → `toggle_pause()`; KEYBOARD → `message="Keyboard comes in a later version"` for 2 s (Plan 3).
  - **DRAGGING:** the cursor follows raw clamped gaze, no snapping.
  - **SCROLLING:** first `move` to `last_page_point`, then `wheel(+1 or -1)` every 0.1 s while gaze stays inside the pressed button's rect expanded by 0.03·W. Leaving it, or a `None` gaze, returns to NORMAL immediately.
  - **Face lost** (`gaze is None` and `closed is None`): scrolling stops at once. `DRAGGING` with no face for > 3 s → `key(VK_ESCAPE)`, `button_up("left")`, NORMAL, `message="Drag cancelled: face lost"`.
  - **Paused:** no moves, clicks, wheel or strip. Blink events other than `pause` are ignored, and the strip is hidden.

- [ ] **Step 1: Write the failing tests** with `FakeActuator` (records `(name, args)`), a fixed target list, screen 2560×1600, and a helper `run(ctrl, script)` stepping at 60 Hz through `[(duration, gaze, closed), ...]`.

```python
def test_gaze_moves_cursor_to_snapped_target():
def test_long_blink_clicks_where_user_looked_before_closing():
    # look at button A; during the blink gaze drifts down 200 px / becomes None
    assert ("click", ("left",)) in fake.calls and last_move(fake) == A.center
def test_right_click_is_one_shot():  # reveal strip, blink on RIGHT_CLICK, blink on page -> right; next -> left
def test_select_drag_down_then_up():  # SELECT, blink at P1 -> button_down at P1, gaze to P2 moves, blink -> up at P2
def test_scroll_runs_while_looking_and_stops_on_leave():
    # blink on SCROLL_DOWN, look at it 1 s -> about 10 wheel(-1); cursor parked at last_page_point; look away -> no more
def test_double_blink_double_clicks():
def test_pause_blocks_everything_until_pause_again():
def test_face_lost_stops_scroll_immediately():        # Review Focus 1
def test_face_lost_during_drag_cancels_after_3_s():   # Review Focus 1: key(ESC) then button_up, not before 3 s
def test_off_screen_gaze_does_not_move_or_click():    # Review Focus 5: gaze (1280, 1.3*1600) + long blink -> no calls
def test_off_screen_gaze_does_not_reveal_strip():     # Review Focus 5: gaze (1.3*2560, 800) for 1 s -> strip hidden
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement `Controller`.**
- [ ] **Step 4: Run** `pytest -v` (whole suite). Expected: all PASS.
- [ ] **Step 5: Commit** `feat: controller state machine for clicks, strip, scroll, drag, pause`.

---

### Task 10: Camera, landmarks, overlay and calibration window

**Files:**
- Create: `src/gazemouse/camera.py`, `src/gazemouse/landmarks.py`, `src/gazemouse/tracker.py`, `src/gazemouse/overlay.py`, `src/gazemouse/calibration_ui.py`, `src/gazemouse/tools/camcheck.py`
- Test: `tests/test_landmarks.py`, `tests/test_overlay_math.py`

**Interfaces:**
- Consumes: `FaceObservation` (Task 1), `CalibrationSession`/`CalibStep`/`CalibrationResult` (Task 3), `OverlayState`/`Mode` (Task 9), `StripLayout` (Task 6).
- Produces:
  - `CameraInfo(width: int, height: int, fps: float)`; `class Camera(index: int | None = None)` that tries backends MSMF, DSHOW, ANY × indexes 0–2 until a read succeeds, asks for 1280×720 and reopens at the default size if a read then fails (the RDP camera broke on 720p), with `read() -> tuple[np.ndarray, float] | None` (BGR frame, `time.perf_counter()`), `info() -> CameraInfo` (fps measured over the first 30 frames), `close()`.
  - `class FaceTracker(model_path: Path)` (VIDEO mode, 1 face, `output_face_blendshapes=True`, `output_facial_transformation_matrixes=True`) with `process(frame_bgr: np.ndarray, t: float) -> FaceObservation | None`. Blink values come from blendshapes `eyeBlinkLeft`/`eyeBlinkRight`; timestamps passed to MediaPipe must strictly increase (ms).
  - `class TrackerThread(QThread)` with signal `observed = Signal(object, float)` (`FaceObservation | None`, t) and `camera_info = Signal(object)`.
  - `to_logical(pt: tuple[float, float], dpr: float) -> tuple[float, float]` and `rect_to_logical(r: Rect, dpr: float) -> Rect` in `overlay.py`. `class Overlay(QWidget)` with `show_state(state: OverlayState, layout: StripLayout)` and `set_ring_visible(visible: bool)`. Frameless, `Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput`, `WA_TranslucentBackground`, covers the primary screen; after `show()` OR in Win32 ex styles `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE`. Draws: gaze ring (radius 18 logical px, 3 px, semi-transparent blue); snapped-rect outline; hidden strip as a glowing `edge_line`; visible strip as translucent rounded buttons labelled "▲ Scroll up", "▼ Scroll down", "Right click", "Select / drag", "Keyboard", "Pause", with `hot_button` highlighted; a small mode badge near the cursor when mode ≠ NORMAL ("Right click next", "Select: blink at start", "Dragging", "Scrolling"); "Paused — close eyes 2 s to resume" when paused; `message` as a toast.
  - `class CalibrationWindow(QWidget)` with signal `finished = Signal(object)` (`CalibrationResult`) and slot `on_features(feat: np.ndarray | None, t: float)`. Full screen, 3-second countdown ("Look at each red dot until it moves"), then drives `CalibrationSession` and draws the dot shrinking with `progress` (red for calibration, orange for validation). Result text: ok → "Accuracy: {error_pct:.1f}% of screen width" plus "good" (≤ 6) or "usable but weak" (> 6); failure → "Couldn't see your eyes well enough. Add light in front of you and try again." Shown for 5 s, then emits `finished`.
- [ ] **Step 1: Write the failing tests.** `test_overlay_math.py`: `to_logical((2560, 1600), 1.25) == (2048, 1280)`, and the same at dpr 1.0, 1.5 and 2.0 (Review Focus 3). `test_landmarks.py` (marked `model`, skipped if the model file is missing): a black 1280×720 frame → `None`; timestamps given out of order don't raise (the tracker bumps them by 1 ms).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the six modules. `tools/camcheck.py` prints `CameraInfo`, the share of frames with a face and eyes open over 60 frames, and a warning when below 1280×720 or 25 fps.
- [ ] **Step 4: Run** the tests. Expected: PASS. **Manual on the laptop:** `python -m gazemouse.tools.camcheck` shows ≥ 1280×720, ≥ 25 fps and ≥ 90% face frames.
- [ ] **Step 5: Commit** `feat: camera, face tracker, overlay and calibration window`.

---

### Task 11: App wiring

**Files:**
- Create: `src/gazemouse/app.py`, `src/gazemouse/__main__.py`
- Test: `tests/test_app_args.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `parse_args(argv: list[str]) -> argparse.Namespace` with `--profile NAME` (default `"default"`), `--calibrate` (force), `--camera INDEX`; `main(argv: list[str] | None = None) -> int`.
- Flow: `set_dpi_awareness()` before `QApplication`; `primary_screen_size()`; start `TrackerThread`. Load the profile, or run `CalibrationWindow` (also when `--calibrate` is given). After a failed or weak (> 6%) result, recalibrate automatically up to 2 more times, then continue with the best model and set the tray tooltip to the accuracy warning. Save successful results with `meta={"error_pct", "camera": CameraInfo, "screen": [w, h], "date"}`. Per observation, on the GUI thread: `extract_features` → `GazeModel.predict` → `OneEuro` → `BackgroundTargets.set_point` → `Controller.step(t, gaze, is_closed(...))` → `Overlay.show_state`. `obs is None` → `step(t, None, None)`; eyes closed (features `None` but obs present) → `step(t, None, True)`. Tray icon (`QSystemTrayIcon`) with menu: Recalibrate, Pause/Resume, Show/Hide gaze ring (spec §6; `Overlay.set_ring_visible(bool)`), Quit. Quit stops the threads and releases the camera.

- [ ] **Step 1: Write the failing test** for `parse_args` defaults and flags.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: all PASS. **Manual (dev box over RDP, smoke only):** `python -m gazemouse --calibrate` opens the calibration window and doesn't crash. Accuracy over RDP is meaningless.
- [ ] **Step 5: Commit** `feat: wire app with tray, calibration flow and controller loop`.

---

### Task 12: Laptop acceptance run

**Files:**
- Create: `docs/test-log.md`

- [ ] **Step 1: Set up the laptop.** Python 3.10+, clone the repo, venv, `pip install -e .`, `python scripts/fetch_model.py`.
- [ ] **Step 2: Parity check.** `python -m gazemouse --calibrate`. Record the `error_pct`. Expected: ≤ 6% (the browser spike measured 5.3% live on the same laptop). If it's worse, record `camcheck` output and stop here. Plan 2 starts with that gap.
- [ ] **Step 3: Task checklist**, eyes and blinks only, recording pass/fail and notes for each: click the Chrome icon on the taskbar; click the address bar; (helper types a search, since there's no keyboard until Plan 3); click a result; scroll down and up with the strip; right-click a link and pick "Open in new tab"; select a paragraph with Select/drag; play and pause a YouTube video; pause control with a 2 s eye close and resume.
- [ ] **Step 4: Stability.** 10 minutes of normal reading. Count accidental clicks (target < 1) and note whether accuracy drifted (target: still ≤ 6% on a final recalibration check).
- [ ] **Step 5: Commit** `docs: laptop acceptance results for plan 1`.
