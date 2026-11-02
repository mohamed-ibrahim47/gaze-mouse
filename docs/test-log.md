# Laptop acceptance test: Plan 1

Run this on the laptop itself, not over Remote Desktop. Fill in the **Result** column as you go.

## 1. Set up

Needs Python 3.10 or newer and Git.

```powershell
git clone -b plan1-core-control https://github.com/mohamed-ibrahim47/gaze-mouse.git
cd gaze-mouse
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python scripts\fetch_model.py
.venv\Scripts\python -m pytest -q
```

Expected: all tests pass.

## 2. Camera check

Sit normally, with light in front of you, and look at the screen.

```powershell
.venv\Scripts\python -m gazemouse.tools.camcheck
```

| Check | Target | Result |
|---|---|---|
| Camera size | at least 1280×720 | |
| Speed (with face tracking) | at least 25 fps | |
| Eyes seen | at least 90% of frames | |

If this fails, write down the output and stop. The later steps depend on it.

## 3. Calibrate

```powershell
.venv\Scripts\python -m gazemouse --calibrate
```

Follow each dot with your eyes and keep your head still. It takes about 40 seconds. Press **Esc** to cancel.

The app recalibrates by itself, up to 3 tries, when the result is weak or fails.

| Check | Target | Result |
|---|---|---|
| Accuracy shown at the end | 6% or less | |
| Gaze ring follows your eyes | roughly, with little shake | |
| Ring reaches all four corners and the bottom row | yes | |

To quit, right-click the tray icon (near the clock) and choose **Quit**. If the cursor is hard to control,
**close your eyes for 2 seconds** to pause, then use the real mouse.

## 4. Blinks and normal use

Use eyes and blinks only. A helper may type the search, because the keyboard arrives in Plan 3.

| Task | Pass? | Notes |
|---|---|---|
| Long blink (about half a second) clicks; the click comes about 0.6 s after your eyes open | | |
| Normal blinking and talking cause no clicks | | |
| One-eye wink and laughing cause no clicks | | |
| **Looking at the taskbar or the bottom of the screen does not count as eyes closed** | | |
| Click the Chrome icon on the taskbar | | |
| Click the address bar | | |
| Click a search result | | |
| Look at the right edge for a moment: the strip slides out | | |
| Scroll down and up with the strip | | |
| Normal blinks while scrolling don't stop it | | |
| Right-click a link, then choose "Open in new tab" | | |
| Select a paragraph with Select / drag | | |
| Play and pause a YouTube video | | |
| Two long blinks double-click a desktop icon | | |
| Close eyes for 2 s to pause; again to resume | | |
| Tray menu: Recalibrate, Pause/Resume, gaze ring on/off, Quit | | |
| Restart without `--calibrate`: it starts straight away from the saved profile | | |

## 5. Ten minutes of reading

Read something normally for 10 minutes.

| Check | Target | Result |
|---|---|---|
| Accidental clicks | fewer than 1 | |
| Accuracy on a fresh calibration afterwards | still 6% or less | |
| CPU in Task Manager | note it | |

## 6. Other laptops (when you can)

Try one other screen size or display scaling (100%, 150%). The ring and strip should still line up with
where you're looking.

## Results

Date:

Laptop and screen:

Biggest problems, worst first:

1.
2.
3.
