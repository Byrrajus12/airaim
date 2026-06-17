"""Webcam capture + MediaPipe hand landmark extraction.

Opens the webcam, runs MediaPipe hand tracking, draws the landmark overlay,
and reports the detected gesture state.

This build of mediapipe ships only the Tasks API (mp.tasks), not the legacy
mp.solutions module, so we use HandLandmarker in VIDEO mode and draw the
overlay ourselves.
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import gestures

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandTracker:
    """Thin wrapper over MediaPipe Tasks HandLandmarker for a single hand."""

    def __init__(self, max_hands=1, det_conf=0.7, track_conf=0.5):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"hand landmark model not found at {MODEL_PATH}"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms):
        """Run detection on a BGR frame.

        Returns (landmarks, result) where landmarks is a list of 21
        (x, y, z) normalized tuples for the first hand, or None if no hand.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            landmarks = [(p.x, p.y, p.z) for p in hand]
        return landmarks, result

    def draw(self, frame_bgr, result):
        if not result.hand_landmarks:
            return
        h, w = frame_bgr.shape[:2]
        for hand in result.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            for c in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[c.start], pts[c.end], (0, 200, 0), 2)
            for (x, y) in pts:
                cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)

    def close(self):
        self.landmarker.close()


def run_sandbox(camera_index=0):
    """Show the hand overlay and print the gesture state whenever it changes."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera_index}")
        return

    tracker = HandTracker()
    last_state = None
    prev_t = time.time()
    start = prev_t
    fps = 0.0

    print("AirAim sandbox running. Press 'q' in the window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: failed to read frame")
                break

            # Mirror so movement feels natural (like a front-facing camera).
            frame = cv2.flip(frame, 1)

            timestamp_ms = int((time.time() - start) * 1000)
            landmarks, result = tracker.process(frame, timestamp_ms)
            state = gestures.classify(landmarks, last_state == "pinch") if landmarks else "unknown"

            if state != last_state:
                print(f"gesture: {state}")
                last_state = state

            tracker.draw(frame, result)

            # FPS (smoothed) for a quick stability read.
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            color = (0, 255, 0) if landmarks else (0, 0, 255)
            cv2.putText(frame, f"{state}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(frame, f"{fps:4.1f} fps", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("AirAim - sandbox", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
