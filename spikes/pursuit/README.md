# Pursuit calibration spike (throwaway)

A test bench, not part of the app. It checks whether calibrating with a gliding dot, plus a step where
you hold your eyes on a dot while moving your head, makes the gaze point survive head movement.
Findings are in [docs/algorithms.md](../../docs/algorithms.md), section 5.

## The session (about 3 minutes, Esc quits)

1. 16 calibration dots, head still (the app's current calibration).
2. A dot glides around the screen for 30 s; follow it with your eyes, head still.
3. The same glide; let your head turn with it.
4. A still dot at three places; keep your eyes on it while you nod and turn your head.
5. 9 test dots, head still.
6. 9 test dots, head free.

Every model learns from parts 1 to 4 and is scored on parts 5 and 6. The result table is shown on screen
and saved to `runs/pursuit-result-<time>.txt` next to the exe, with the raw frames in
`runs/pursuit-<time>.npz`. Then a green ring (current model) and a blue ring (number keys pick the
model) follow your gaze live.

## Models

| Name | What it learns from | Inputs |
|---|---|---|
| Dots (current) | part 1 | 4 eye features, in pixels (the app today) |
| Dots+glides, head (px) | parts 1-3 | eye features + head yaw and pitch, in pixels |
| Angles: ... | the parts named | eye features + head yaw and pitch, fitted as the angle from the eyes to the gaze point, then turned into pixels using where the head is in 3D |

The gliding dot's target is where the dot was `lag` seconds earlier. The lag (camera and processing
delay, 250 ms at 30 frames/s) is measured from part 2.

## Run and build (Git Bash, from the repo root)

```bash
export PYTHONPATH=src TEMP='D:\dev\tmp' TMP='D:\dev\tmp'
.venv/Scripts/python spikes/pursuit/pursuit_spike.py                 # live session
.venv/Scripts/python spikes/pursuit/pursuit.py runs/pursuit-<time>.npz  # re-score a saved run
.venv/Scripts/python spikes/pursuit/headless_pursuit.py <face photo> <shot.png>  # offscreen flow check
.venv/Scripts/python -m PyInstaller spikes/pursuit/pursuit.spec --distpath <out> --workpath D:/dev/tmp/build
```

Runs describe a person's eyes: `runs/` is git-ignored, keep it that way.

## Licence

`wetlib/` is unmodified code from [WebEyeTrack](https://github.com/RedForestAI/WebEyeTrack) (MIT,
`wetlib/LICENSE`), used only for the 3D head position. WebEyeTrack's eye network and weights are not
used or included.
