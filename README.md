# Gaze Mouse

Control a Windows laptop with only your eyes, using the built-in webcam. No glasses, no eye-tracker
hardware, no special launcher screen.

Gaze Mouse is for people with paralysis or ALS who can't use a mouse or keyboard. Your eyes move the
real cursor, which snaps onto buttons and links. A long blink clicks. Looking at the right edge of the
screen slides out a strip of large buttons for scrolling, right click, select/drag, the keyboard and
pause. Everything runs on your computer. No camera images leave it.

> **Status:** early development. Plan 1 (core control) is being built. Not ready for daily use yet.

## How it works

1. **Calibration** (about 40 seconds): follow a dot around the screen with your eyes. The app learns
   how *your* eyes map to *your* screen, then tells you how accurate it is.
2. **Aiming:** a pretrained face model ([MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker))
   finds your irises in each camera frame. Your calibration turns that into a screen position.
3. **Pressing:** close both eyes for about half a second.

| Action | How |
|---|---|
| Left click | Long blink (~0.5 s) |
| Double click | Two long blinks |
| Scroll, right click, select/drag, keyboard | Look at the right screen edge, then long-blink a button |
| Pause / resume | Close your eyes for 2 seconds |

## Background

This project started in 2018 as part of a university graduation project: an EEG-controlled wheelchair
for paralysed people, with software for using a computer with the eyes. Back then, finding the iris in
a webcam image didn't work well enough, so the project ended up with a camera mounted on glasses and a
6-button screen. Modern face models make the plain laptop webcam good enough. In an early test on a
laptop's 720p camera, gaze error was about 5% of the screen width.

## Requirements

- Windows 10 or 11
- A webcam with at least 1280×720 at 25 fps (most recent laptops; a cheap 1080p USB webcam also works)
- Good light on your face

## Development setup

```bash
git clone https://github.com/mohamed-ibrahim47/gaze-mouse.git
cd gaze-mouse
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip   # old pip can't do editable installs
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python scripts\fetch_model.py      # downloads the face model (~4 MB)
.venv\Scripts\python -m pytest                   # unit tests
.venv\Scripts\python -m gazemouse --calibrate    # run the app
```

Test with the laptop's own camera. A camera forwarded over Remote Desktop is too low-resolution to track
the eyes.

## Docs

- [Design](docs/design.md): goals, decisions and why
- [Plan 1: core control](docs/superpowers/plans/2026-10-31-plan1-core-control.md)

## Licence

[GPL-3.0](LICENSE). Improvements to this accessibility tool stay open for everyone.
