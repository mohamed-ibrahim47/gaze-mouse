# Gaze algorithms: what failed in 2018, what we use now, and what could replace it

**Date:** 2026-11-14 · Background for `design.md` sections 1, 4 and 7.

## 1. The 2018 algorithms and why they failed

The 2018 failure reasons were never written down. The explanations below come from how each algorithm
works, not from a record of what was seen then. The papers are matched by title and were not re-read.

| Algorithm (paper) | How it finds the iris or pupil | Why it fails on a laptop webcam |
|---|---|---|
| **Daugman's integro-differential operator** ("Iris localization using Daugman's algorithm", "How Iris Recognition Works") | Tries many circles and keeps the one where brightness changes most sharply along the edge | Built for iris *recognition*: a close-up infrared photo where the iris is 100+ pixels wide and fully visible. In a webcam frame the iris is 10 to 20 pixels wide and the lids cover its top and bottom, so there is no full circle to find |
| **Hough circle transform** (the old OpenCV scripts) | Edge pixels vote for circle centres | Same problem. With so few edge pixels the votes are noise. It also needs hand-tuned thresholds that break when the light changes |
| **Eye centre by gradients**, Timm and Barth ("Accurate Eye Centre Localisation by Means of Gradients") | The centre is the point that most image gradients point away from | The best of the classic methods, and it does work at low resolution. But it needs a good eye crop first (Haar cascades were jittery), and eyebrows, lashes, glasses and reflections pull the centre away |
| **Isophote / isocentric patterns**, Valenti and Gevers ("Accurate Eye Center Location through Invariant Isocentric Patterns") | Curves of equal brightness vote for their centre | Same class as the gradient method, with the same weak points: shadows, lids and reflections |

**The deeper reason is the same for all four.** They only answer "where is the pupil centre in the
image", and they answer it to about 1 to 2 pixels. Across the whole screen the iris moves only about
10 to 15 pixels in a 640×480 frame, so 1 to 2 pixels of noise is 10 to 20% of the screen. None of
them gives a stable reference (eye corners, lids), so head movement looks like eye movement. None
includes a calibration step that maps eye position to screen position.

That is why the glasses-mounted camera worked in 2018: close to the eye, the iris is hundreds of
pixels wide and the head cannot move relative to the camera.

## 2. What Gaze Mouse uses now

1. **MediaPipe Face Landmarker** (pretrained, Apache-2.0) gives 478 face points, including iris centres.
2. **Four hand-made features** (`features.py`): iris x along the eye-corner line, iris y below the
   corner line, iris y below the lid midline, lid openness. Both eyes averaged, scaled by eye width.
3. **Ridge regression** (`gaze_model.py`): a linear map from those 4 numbers to screen x, y, fitted
   per user from a 16-dot calibration.
4. **One Euro filter** to smooth the cursor.

Why this works where 2018 did not: MediaPipe is a neural network trained on many faces, so it places
the iris centre to a fraction of a pixel and gives the eye corners and lids as a reference in the same
pass. A 720p camera doubles the pixels. The calibration supplies the missing mapping. The spike
confirmed the pixel argument: at 640×480 over RDP the new pipeline failed too.

Measured on the laptop (one person, one session): about 5.3% of screen width live, 3.7% best offline.
Published webcam trackers report 2.4° to 4°, so this is in the normal range.

## 3. Its weakness

The features describe where the iris sits inside the eye socket. That equals gaze direction only
while the head is still. The spike already showed the crack: head height correlated −0.98 with target
Y, so the model partly learned the nod, not the eye. Adding raw head pose as regression inputs made
accuracy worse, so the model uses only the four eye features and head movement is uncorrected.

## 4. Options to replace or extend it

| # | Approach | Handles head movement | Published accuracy | Licence | Effort |
|---|---|---|---|---|---|
| A | Current features + ridge | No | 5.3% of screen width (our spike) | Clean | Done |
| B | Same features, normalised by head pose and distance before the regression | Partly | Not measured | Clean | Low |
| C | 3D geometric model: eyeball centre + iris gives a gaze ray, intersected with the screen plane; calibration fits a small personal offset | Yes, by construction | Noisy at webcam resolution; no solid number found | Clean | Medium |
| D | Small appearance-based CNN + few-shot personal calibration (WebEyeTrack, 2025) | Yes (takes head pose as input) | 2.32 cm on GazeCapture with 9 calibration samples; 670 KB model | Code is MIT. **Licence of the weights' training data not checked yet** | Medium to high |
| E | Google's model (Valliappan et al. 2020) | Yes | About 0.5 cm on phones after calibration | **Model not released** | High (retrain) |
| F | iTracker / GazeCapture (2016) | Yes | 1.3 to 1.7 cm on phones | Dataset believed research-only; verify | High |

**Recommendation:** keep A for Plan 1. If the laptop acceptance run shows drift from head movement,
try B first, then D, with C as the fallback if D's weights cannot ship under GPL-3.0. Choose by
replaying logged runs, as `design.md` section 7 says, not by paper numbers.

## Sources

- [WebEyeTrack (arXiv 2508.19544)](https://arxiv.org/abs/2508.19544)
- [Eye Tracking for Everyone: iTracker / GazeCapture (arXiv 1606.05814)](https://arxiv.org/abs/1606.05814)
- [Valliappan et al. 2020, smartphone eye tracking, Nature Communications](https://pmc.ncbi.nlm.nih.gov/articles/PMC7486382)
- [Webcam appearance-based gaze accuracy, 2.4° (Behavior Research Methods 2023)](https://link.springer.com/10.3758/s13428-023-02190-6)
