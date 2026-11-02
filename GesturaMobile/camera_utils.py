"""camera_utils.py - camera, MediaPipe hand tracker and drawing helpers (needs cv2 + mediapipe)."""
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import BASE_DIR, frame_features

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def open_camera(index=0, width=1280, height=720):
    cap = None
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        for alt in (1, 2):
            cap = cv2.VideoCapture(alt)
            if cap.isOpened():
                break
    if not cap.isOpened():
        raise RuntimeError("Could not open any camera. Check camera permissions / index (--camera).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class HandTracker:
    """MediaPipe HandLandmarker in VIDEO mode (temporal tracking, smoother landmarks)."""

    def __init__(self, model_path=None, num_hands=2):
        model_path = model_path or os.path.join(BASE_DIR, "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"'{model_path}' not found (hand_landmarker.task must sit next to the scripts).")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, bgr_frame):
        """bgr_frame must already be mirrored. Returns (result, 132-d feature vector)."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)
        return result, frame_features(result.hand_landmarks)


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for hand in result.hand_landmarks or []:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if i in (4, 8, 12, 16, 20):
                cv2.circle(frame, p, 7, (0, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 4, (255, 0, 128), -1, cv2.LINE_AA)


def put_text(frame, text, org, scale=0.8, color=(255, 255, 255), thickness=2):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
