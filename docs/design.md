# Gaze Mouse: control a Windows laptop with only your eyes

**Date:** 2026-10-31 · **Status:** approved 2026-10-31 · **Working name:** Gaze Mouse (my call; rename freely)

---

## 1. Intent and the why

In 2018, for a graduation project, the user and a friend built a proof of concept for paralysed people:
a chair driven by EEG (alpha/beta/theta waves through an Arduino), plus software to use a computer with
the eyes. The software part failed in its original form: every iris-tracking algorithm they tried
(Daugman, gradient eye-centre, Hough circles; papers in `D:\claude-apps\eeg\`) could not find the iris
reliably in a front webcam image. They fell back to **glasses holding a camera in front of the eye** and
a PyQt app with **6 big buttons** ("Pain", "Food", "Movies"…). The user was never happy with that UI.
The idea they never tried was a calibration step mapping iris position to cursor position.

This project finishes what beat them: **no glasses, no special UI, only the laptop's built-in camera**,
controlling the whole of Windows (any app, any website) with eye movement plus blinks.

**Who it's for:** people with paralysis or ALS who cannot use a mouse or keyboard.
- ALS: eye muscles usually last longest; face and head control often fade. Head often barely moves.
- Spinal-cord paralysis: face usually works fully.

**Why now:** pretrained face models (MediaPipe Face Landmarker, 478 points incl. iris, real time on a
CPU) solve the exact part that failed in 2018. A spike on 2026-10-31 proved it on the user's laptop
(section 4).

**Related idea:** `app-ideas/low-vision-assist-app.md` needs the same "where on screen is the user
looking" engine. Keep the gaze engine a separate module so that app can reuse it later.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Move the real Windows cursor by gaze, accurately enough to use normal apps (browser, email, YouTube,
   File Explorer, Office) without a special launcher UI.
2. Every mouse action reachable with eyes + blinks only: left, double, right click, select/drag, scroll.
3. Type text with the eyes in **English and Arabic** (user's call).
4. Short per-user calibration, saved per user, quick to redo.
5. Fully on-device and offline. No camera frames leave the machine.
6. **Open source, published on SourceForge** (user's call), installable by non-technical people.
7. Works on **other people's Windows laptops**, not only the user's own (user's call).

**Non-goals (v1)**
- The EEG chair / Arduino side.
- macOS, Linux, Android, phones.
- Supporting special eye-tracker hardware (Tobii etc.). Webcam only.
- Voice control.
- Being a full AAC (speech-generating) system.
- Multiple monitors (primary monitor only in v1).
- Admin windows, UAC prompts, lock/sign-in screen (see section 6).

**Anti-goals (failure even if it technically works)**
- Clicking things the user only *looked at* (the "Midas touch" problem). Accidental clicks are the worst bug.
- Requiring the user to stare / hold gaze to click. The user explicitly rejected dwell clicking as too hard.
- Needing a helper to recalibrate every few minutes.
- A big always-on-screen panel of buttons that takes over the desktop (the 2018 UI).
- Needing glasses, a head mount, or any extra hardware.

## 3. Constraints

- **Windows 10/11 desktop app.** A browser page can't move the system cursor or click in other apps.
- **Python** (the user's 2018 language; MediaPipe, OpenCV, PySide6 all available). Python 3.10+ on the
  dev box at `C:\Python310`; **all venvs, caches, temp files and downloads on D:** (C: is full), e.g.
  `PIP_CACHE_DIR=D:\dev\pip-cache`, `TEMP=D:\dev\tmp`.
- **Camera:** built-in laptop webcam, used directly. **Not over Remote Desktop.** Forwarded RDP cameras
  arrive at 640×480 with ~0.6 s lag, which proved too coarse (section 4). The user develops on this PC
  over RDP, so testing has to happen on the laptop itself.
- **Packages proven in the spike:** `mediapipe` 1.2.0 (Tasks API only; legacy `mp.solutions` is gone),
  `opencv-python` **4.x** (the 5.0 wheel would not open cameras by index), `numpy`.
- Personal side project: `D:\claude-apps\`, own git repo, personal GitHub identity, push only when asked.
- **Everything shipped must be redistributable under an open-source licence.** That rules out model
  weights or datasets with non-commercial / research-only terms. ETH-XGaze (CC BY-NC-SA) and
  MPIIGaze-trained weights are out unless their licence allows it. MediaPipe (Apache-2.0) and PySide6
  (LGPL, dynamically linked) are fine. Word lists for suggestions need compatible licences too.
- **Other laptops:** webcams vary (many are 640×480 or 720p at 30 fps), as do display scaling
  (100–200%), resolution and CPU. The app must cope or say clearly why it can't.

## 4. What the spike proved (`D:\claude-apps\eeg\gaze-spike\`, throwaway)

| Run | Camera | Mean error on 5 held-out dots |
|---|---|---|
| Python over RDP, run 2 | 640×480 @ ~23 fps, ~0.6 s lag | 855 px (horizontal fine, vertical broken by a bug) |
| Python over RDP, run 3 (bug fixed) | 640×480 | ~400–500 px: too few pixels per eye movement |
| **Browser on the laptop** | **1280×720 @ 77–95 fps** | **108 px live, 75 px best offline model** on a 2048×1280 screen |

**Units:** the browser figures are CSS pixels. The laptop runs 2560×1600 physical at 125% scaling, so
in physical pixels (what a native app sees) the live error is ~135 px and the best model ~94 px. Targets
in this document are stated as **% of screen width** to avoid the ambiguity: live ≈ 5.3%, best ≈ 3.7%.

Laptop run details (`runs/run-20261031-211344.json`, replay with `analyze.py`):
- Iris horizontal vs target X: correlation −0.99. Iris vertical (from eye corners) vs Y: +0.96.
- Best features: iris x, iris y relative to the **eye-corner line**, iris y relative to the **lid
  midline**, and **lid openness**, combined linearly (ridge, small λ): 75 px (x 61, y 29).
- Adding raw head-pose numbers as regression inputs made it **worse** (176–264 px).
- Iris settles on a new dot ~0.4 s after it appears (just the eye's own jump).

**Limits of that evidence:** one person, one session, head held still (the user still nodded toward
dots: head height correlated −0.98 with Y), 5 validation dots, no drift test over 10+ minutes, no
glasses, one lighting condition. Drift over time is the biggest unknown.

The spike code is reference only. Don't build on it; rewrite cleanly.

## 5. How it works (decided with the user)

### 5.1 Gaze engine
- **Pretrained model + per-user calibration** (user's call). MediaPipe Face Landmarker finds iris and
  face landmarks every frame. A small per-user model maps eye features to screen position.
- **Calibration:** follow dots on screen (spike used 16 dots in a 4×4 grid + 5 validation dots, ~40 s).
  Show the validation error at the end in plain words.

### 5.2 Precision: all three tricks (user wanted all of them)
1. **Snap:** the cursor locks onto the nearest clickable element (button, link, text box, list item)
   near the gaze point, using **Windows UI Automation** to get element rectangles from any app.
2. **Zoom:** for crowded targets or apps that expose nothing to UI Automation (games, canvases), a
   magnified view (~2×; the spike's ~100 px error suggests 2× is enough) lets the user pick precisely.
3. **Head fine-aim:** small head movements nudge the cursor the last few pixels. **Optional setting,
   off by default** (many ALS users can't move their head reliably).

### 5.3 Pressing: blinks, no dwell (user's call)
- **Long blink** (about 0.5 s, both eyes) = **left click** on the current (snapped) target.
- **Double long blink** = double click.
- **Close eyes ~2 s** = pause / resume all control (watch a video, rest).
- Natural blinks (~0.1–0.15 s) must never click. The long-blink threshold is tuned per user.
- **Gaze freezes at blink start:** while eyes close the iris reading slides down, so the click goes to
  where the user looked just before the eyes began closing (keep ~200 ms of gaze history).
- Rejected by the user: dwell clicking, open-mouth gesture. Eyebrow gesture dropped (the strip below
  replaces it). Blinks are the only face movement required.

### 5.4 The hidden edge strip: option C (user's choice; mockup in `.superpowers/brainstorm/`)
- Normally only a **thin glowing line on the right screen edge**.
- Looking at the line slides out a see-through column of large buttons:
  **Scroll up · Scroll down · Right click · Select/drag · Keyboard · Pause**. Looking back at the page
  hides it.
- Buttons are pressed with a long blink like anything else. Each button is about 2× the measured gaze error.
- **Scroll:** long blink on a scroll button starts scrolling; it continues while the eyes stay on the
  button; looking away stops it. Scrolls the window under the last page position the user was aiming at.
- **Right click / Select-drag are one-shot modes:** they apply to the next long blink, then revert to
  left click. Select/drag: blink at the start point (mouse down), cursor follows gaze, blink at the end
  point (mouse up).

## 6. My calls (not discussed; the user should check these)

- **Strip reveal:** opens after looking at the edge line ~0.3 s, so reading text near the right edge
  doesn't keep popping it open. Looking alone never *presses* anything.
- **Zoom trigger:** when several snap targets are within the gaze error radius (ambiguous), the first
  long blink opens the 2× zoom of that area instead of clicking; a second long blink picks inside it.
  With no ambiguity, a long blink clicks straight away.
- **Keyboard:** our own big-key on-screen keyboard (Windows' built-in keyboard has keys smaller than the
  gaze error), with word suggestions, typed with gaze + long blink. Keys snap like any other target. It
  must not steal focus from the app being typed into.
  - **English and Arabic layouts** with a big switch key. The Arabic layout and suggestion bar are
    right-to-left. Offline word-frequency lists per language for suggestions.
  - Characters are sent as Unicode (`SendInput` with `KEYEVENTF_UNICODE`), so typing works whatever
    keyboard layout Windows has active.
- **App UI language:** the app's own screens (setup, calibration, settings, strip labels) in English and
  Arabic, following the Windows display language, switchable in settings.
- **Licence:** GPL-3.0, so improvements to an accessibility tool stay open. The 2018 reference tracker
  (PyGaze webcam-eyetracker) was GPL-3.0 too. MIT is the alternative if wider reuse matters more.
- **Publishing:** source on GitHub (personal account, like the user's other repos); releases (installer +
  portable zip) on **SourceForge**, which can mirror the GitHub releases. Windows installer via Inno
  Setup or similar. Unsigned builds trigger SmartScreen warnings; apply for free open-source code signing
  (e.g. SignPath Foundation) before the first public release. A simple project page + privacy note
  ("no data leaves your computer"), like the user's other showcase pages.
- **"Camera not good enough" path (other laptops):** setup measures camera resolution/fps and then the
  calibration's validation error. If the error exceeds the target (~6% of screen width), say so plainly,
  explain why (camera too low-resolution or dark), and suggest fixes: more light, face the light, or a
  cheap 1080p USB webcam. Never silently run with bad accuracy.
- **Display handling:** per-monitor DPI aware; all internal geometry in physical pixels. **v1 supports
  the primary monitor only.**
- **Minimum hardware target:** a typical 4-year-old laptop CPU running MediaPipe on CPU at ≥ 25 fps;
  GPU delegate optional.
- **Platform/stack:** Python + PySide6 overlay (the user already uses PySide6 in `downloader/`),
  MediaPipe Tasks, OpenCV 4.x for capture, `SendInput` via ctypes (or a small library) for mouse/keys,
  the `uiautomation` package (or comtypes UIA) for snap targets.
- **Overlay windows** (gaze ring, strip, zoom lens, keyboard) are always-on-top, click-through and
  never take focus (`WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE`) so real clicks
  land in the real app.
- **Gaze ring visible** (small, see-through), with a toggle to hide it. Users need feedback on where the
  system thinks they're looking.
- **Fix drift from clicks:** every confirmed click on a snapped target is a free calibration sample
  (target centre vs predicted gaze). Use these to keep correcting a small offset in the background.
  Offer a quick 5-dot recalibration when error grows.
- **Profiles:** one calibration profile per user, saved locally; re-used on start, with a quick check.
- **Privacy:** no frames stored. Optional debug logging of features (like the spike's JSON) off by default.
- **Name:** "Gaze Mouse" as a working name.
- **Windows security boundaries are out of scope for v1:** no control of elevated (admin) windows, UAC
  prompts, or the lock/sign-in screen. These need a signed `uiAccess=true` build installed under Program
  Files, which is a later milestone. v1 starts automatically at user logon (Task Scheduler / Startup
  entry) so nobody has to launch it by hand. The success tasks avoid admin prompts.
- **Pluggable press triggers:** the input state machine takes "press" events from a trigger interface.
  v1 ships long blink (threshold adjustable, tuned at setup). The interface must allow adding later a
  one-eye wink, a slower adaptive blink, or a hardware switch without redesign.
- **Camera check:** at startup, measure the camera's resolution/fps; below 1280×720 or ~25 fps, warn
  that accuracy will be poor (the spike showed 640×480 fails).
- **Dev/test loop:** code lives in a git repo; the laptop gets Python 3.10+ and pulls the repo to run
  real tests (the desktop over RDP can only run unit/replay tests). An early milestone confirms that the
  Python pipeline on the laptop matches the browser spike's accuracy, since only the browser was measured there.

## 7. Left to the builder

- **Head-movement handling** (the main technical risk). Raw head pose as regression inputs hurt in the
  spike. Options: geometric normalisation of eye features by head pose; a pretrained appearance-based
  gaze model with calibration correcting only a small personal offset; or both. **Check every model's
  weight and training-data licence first** (see constraints): many popular gaze models (e.g. L2CS-Net
  weights, ETH-XGaze-trained ones) carry non-commercial dataset terms and can't ship in an
  open-source release. Choose by replaying logged runs (extend `analyze.py`'s approach), including deliberate
  head-movement sessions.
- Gaze model form, smoothing filter (spike used a One Euro filter), snap radius, exact thresholds.
- Blink detection method (MediaPipe blendshapes `eyeBlinkLeft/Right` vs lid-openness ratio) and the
  per-user tuning step.
- UI Automation performance (querying element trees fast enough; caching).
- Calibration screen details, settings screen, tray icon, start-with-Windows.
- Packaging (PyInstaller or similar) so it runs on a laptop without a dev setup.
- Test strategy: logged-feature replay tests for the gaze pipeline; unit tests for the blink state
  machine, snap resolver and action modes; manual on-laptop sessions for the rest.

## 8. Success looks like (my guess, for the user to confirm)

- After one calibration, the user completes, eyes only, without help: open Chrome, search for something,
  open a result, scroll it, play a YouTube video, pause it, type and send a short email.
- Gaze error stays under ~6% of screen width (~150 physical px on the user's 2560-wide laptop) over a
  15-minute session without full recalibration.
- Fewer than 1 accidental click per 10 minutes of normal reading.
- Eventually: tested by a real person with ALS or paralysis. None has been identified yet.

## 9. Open risks

1. Drift and head movement over long sessions (untested).
2. Glasses (reflections), poor or backlit lighting, darker or hooded eyes: untested.
3. Long-blink reliability for ALS users with weak eyelids.
4. Apps that expose nothing to UI Automation fall back to zoom only.
5. Fatigue: blink-clicking hundreds of times a day.
