"""MediaPipe Face Landmarker: camera frame -> FaceObservation."""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .features import FaceObservation

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


class FaceTracker:
    def __init__(self, model_path: Path = MODEL_PATH):
        opts = vision.FaceLandmarkerOptions(
            # Bytes, not a path: MediaPipe can't open paths with non-ASCII characters on Windows.
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> FaceObservation | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ms = max(int(t * 1000), self.last_ms + 1)  # MediaPipe needs strictly increasing timestamps
        self.last_ms = ms
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        points = np.array([(q.x * w, q.y * h) for q in result.face_landmarks[0]])
        blink = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return FaceObservation(points, blink["eyeBlinkLeft"], blink["eyeBlinkRight"],
                               np.array(result.facial_transformation_matrixes[0]), t)
