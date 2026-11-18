"""Camera frame -> one Sample: the eye features plus the 3D head position (face origin, cm).

The face-origin geometry is a vectorised copy of WebEyeTrack's (MIT, see wetlib/LICENSE). Its eye
network and weights are not used here."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from gazemouse.features import FaceObservation, extract_features
from gazemouse.landmarks import MODEL_PATH

from wetlib.constants import (LEFT_EYE_HORIZONTAL_LANDMARKS, LEFTMOST_LANDMARK,
                              RIGHT_EYE_HORIZONTAL_LANDMARKS, RIGHTMOST_LANDMARK)
from wetlib.model_based import estimate_face_width, get_head_vector
from wetlib.utilities import (MAX_STEP_CM, create_perspective_matrix, estimate_camera_intrinsics,
                              euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles)

WIDTH_FRAMES = 30  # open-eye frames used to settle the face width (the reference uses only the first)


@dataclass
class Sample:
    t: float
    feat: np.ndarray | None      # gazemouse's eye features; None while the eyes are closed
    head_vec: np.ndarray | None  # (3,)
    origin: np.ndarray | None    # (3,) point between the eyes, cm from the camera
    lm: np.ndarray               # (478, 3) normalised face points, for offline experiments
    rt: np.ndarray               # (4, 4) head transform
    ms: dict                     # how long each stage took

    @property
    def usable(self) -> bool:
        return self.feat is not None and self.origin is not None


def _project(pts, K):
    cam = pts @ K.T
    z = np.where(np.abs(cam[:, 2]) < 1e-6, 1e-6, cam[:, 2])
    return (cam[:, :2] / z[:, None]).astype(np.int32)  # the reference rounds to whole pixels here


def face_origin(lm: np.ndarray, rt: np.ndarray, w: int, h: int, face_width_cm: float) -> np.ndarray:
    """WebEyeTrack.compute_face_origin_3d (face_reconstruction + estimate_gaze_origins), vectorised:
    the reference loops over 478 points in Python and takes ~130 ms a frame. lm is (478, 3), normalised.
    check_geometry.py proves this returns the same numbers."""
    K = estimate_camera_intrinsics(np.zeros((h, w, 3)))
    inv_p = np.linalg.inv(create_perspective_matrix(aspect_ratio=w / h))
    ndc = np.column_stack([2 * lm[:, 0] - 1, 1 - 2 * lm[:, 1], -np.ones(len(lm)), np.ones(len(lm))])
    world = ndc @ inv_p.T
    mesh = np.column_stack([-world[:, 0] / world[:, 3], world[:, 1] / world[:, 3], lm[:, 2]])
    mesh = (mesh - mesh[4]) * np.array([-1, -1, 1])
    mesh = mesh / np.linalg.norm(mesh[LEFTMOST_LANDMARK] - mesh[RIGHTMOST_LANDMARK]) * face_width_cm

    pitch, yaw, roll = rotation_matrix_to_euler_angles(np.linalg.inv(rt[:3, :3]))
    face_r = euler_angles_to_rotation_matrix(-yaw, pitch, roll)
    mesh = mesh @ np.linalg.inv(face_r).T
    face_r = face_r / np.linalg.norm(face_r, axis=0).mean()

    z0 = 60
    detected = lm[:, :2] * np.array([w, h])
    centre = detected.mean(axis=0)
    shift = detected[4] - _project(mesh @ face_r.T + np.array([0, 0, z0], np.float32), K)[4]
    first = np.array([shift[0] * z0 / K[0, 0], shift[1] * z0 / K[1, 1], z0], np.float32)
    trans = first.copy()
    for _ in range(10):
        proj = _project(mesh @ face_r.T + trans, K)
        v = detected - proj
        away = np.einsum("ij,ij->i", v, centre - proj) >= 0
        dist = np.linalg.norm(v, axis=1)
        delta = 0.1 * np.where(away, dist, -dist).sum() / len(proj)
        new_z = trans[2] + max(-MAX_STEP_CM, min(MAX_STEP_CM, delta))
        if abs(new_z - trans[2]) < 0.25:
            break
        trans = np.array([first[0] * new_z / z0, first[1] * new_z / z0, new_z], np.float32)
    face = mesh @ face_r.T + trans
    return (face[LEFT_EYE_HORIZONTAL_LANDMARKS].mean(axis=0) + face[RIGHT_EYE_HORIZONTAL_LANDMARKS].mean(axis=0)) / 2


class WetTracker:
    def __init__(self, model_path: Path = MODEL_PATH):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.last_ms = -1
        self.widths = []
        self.face_width_cm = None

    def process(self, frame_bgr: np.ndarray, t: float) -> Sample | None:
        tick = cv2.getTickCount
        t0 = tick()
        ms = max(int(t * 1000), self.last_ms + 1)
        self.last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lm = np.array([(q.x, q.y, q.z) for q in result.face_landmarks[0]])
        rt = np.array(result.facial_transformation_matrixes[0])
        blink = {c.category_name: c.score for c in result.face_blendshapes[0]}
        px = lm[:, :2] * np.array([w, h])
        feat = extract_features(FaceObservation(px, blink["eyeBlinkLeft"], blink["eyeBlinkRight"], rt, t))
        t1 = tick()
        head_vec = origin = None
        try:
            head_vec = np.asarray(get_head_vector(rt), float)
            if self.face_width_cm is None and feat is not None:
                self.widths.append(estimate_face_width(lm[:, :2], rt))
                if len(self.widths) >= WIDTH_FRAMES:
                    self.face_width_cm = float(np.median(self.widths))
            if self.face_width_cm is not None:
                origin = np.asarray(face_origin(lm, rt, w, h, self.face_width_cm), float)
        except Exception:  # face half out of the frame
            origin = None
        f = 1000 / cv2.getTickFrequency()
        return Sample(t, feat, head_vec, origin, lm.astype(np.float32), rt,
                      {"landmarks": (t1 - t0) * f, "geometry": (tick() - t1) * f})
