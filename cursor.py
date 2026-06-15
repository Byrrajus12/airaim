"""Fake-cursor sandbox.

Moves a circle around a blank window using the frame-to-frame motion of the
index fingertip. Nothing touches the real OS cursor here. Movement only
happens while the hand is an open palm; any other gesture pauses it, so you
can reposition your hand (a fist clutch) without dragging the circle.
"""

import math
import time

import cv2
import numpy as np

import gestures
from tracker import HandTracker

# --- Tuning knobs (tweak these live) ---
EMA_ALPHA = 0.3      # delta smoothing, 0..1. Higher = snappier, lower = smoother.
DEADZONE_PX = 2.0    # ignore fingertip movement smaller than this (camera pixels)
SENSITIVITY = 2.5    # scales fingertip motion into circle motion
TRANSITION_SKIP_FRAMES = 3   # frames to ignore right after the hand opens, so
                             # the uncurl from a fist/pinch isn't read as motion
NO_HAND_GRACE_FRAMES = 10    # max frames to keep coasting after the hand is lost
VELOCITY_DECAY = 0.7         # per-frame shrink of coasting velocity while lost

WINDOW_W, WINDOW_H = 960, 540
CIRCLE_RADIUS = 12


def _move(cx, cy, dx, dy):
    """Shift the circle and keep it inside the window."""
    cx = max(CIRCLE_RADIUS, min(WINDOW_W - CIRCLE_RADIUS, cx + dx))
    cy = max(CIRCLE_RADIUS, min(WINDOW_H - CIRCLE_RADIUS, cy + dy))
    return cx, cy


def run_cursor_sandbox(camera_index=0):
    """Drive a circle on a blank window with the index fingertip."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera_index}")
        return

    tracker = HandTracker()
    cx, cy = WINDOW_W / 2, WINDOW_H / 2   # circle position
    prev_tip = None                       # last fingertip position (camera px)
    prev_state = "unknown"
    skip_frames = 0                       # countdown while ignoring uncurl motion
    no_hand_frames = 0                    # consecutive frames with no hand seen
    ema_dx = ema_dy = 0.0
    start = time.time()

    print("AirAim fake cursor running. Press 'q' in the window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: failed to read frame")
                break

            # Mirror so movement feels natural (like a front-facing camera).
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            timestamp_ms = int((time.time() - start) * 1000)
            landmarks, _ = tracker.process(frame, timestamp_ms)
            state = gestures.classify(landmarks) if landmarks else "unknown"

            tip = None
            if landmarks:
                lx, ly, _ = landmarks[gestures.INDEX_TIP]
                tip = (lx * w, ly * h)

            # The moment the hand opens, hold off for a few frames: the finger
            # is still uncurling, and that motion isn't a real cursor move.
            if state == "open_palm" and prev_state != "open_palm":
                skip_frames = TRANSITION_SKIP_FRAMES

            active = (state == "open_palm" and tip is not None
                      and prev_tip is not None and skip_frames == 0)

            if active:
                no_hand_frames = 0
                dx = tip[0] - prev_tip[0]
                dy = tip[1] - prev_tip[1]

                # Deadzone: drop tiny jitter so a still hand doesn't drift.
                if abs(dx) < DEADZONE_PX:
                    dx = 0.0
                if abs(dy) < DEADZONE_PX:
                    dy = 0.0

                # Smooth the delta, then scale it.
                ema_dx = EMA_ALPHA * dx + (1 - EMA_ALPHA) * ema_dx
                ema_dy = EMA_ALPHA * dy + (1 - EMA_ALPHA) * ema_dy
                cx, cy = _move(cx, cy, ema_dx * SENSITIVITY, ema_dy * SENSITIVITY)
            elif tip is None:
                # Hand lost: keep coasting on the last velocity, decaying it
                # each frame, so a brief dropout glides to a stop instead of
                # freezing. Stops once it's slow enough or we've waited too long.
                no_hand_frames += 1
                ema_dx *= VELOCITY_DECAY
                ema_dy *= VELOCITY_DECAY
                if (no_hand_frames <= NO_HAND_GRACE_FRAMES
                        and math.hypot(ema_dx, ema_dy) >= DEADZONE_PX):
                    cx, cy = _move(cx, cy, ema_dx * SENSITIVITY, ema_dy * SENSITIVITY)
                else:
                    ema_dx = ema_dy = 0.0
            else:
                # Hand present but paused (clutch or transition): forget
                # momentum so it doesn't lurch when tracking resumes.
                no_hand_frames = 0
                ema_dx = ema_dy = 0.0
                if skip_frames > 0:
                    skip_frames -= 1

            # Always remember where the fingertip is, so the next active frame
            # measures movement from here. This is what makes the clutch work:
            # you can move your hand while paused without it counting.
            prev_tip = tip
            prev_state = state

            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), np.uint8)
            active = state == "open_palm"
            color = (0, 255, 0) if active else (0, 0, 255)
            cv2.circle(canvas, (int(cx), int(cy)), CIRCLE_RADIUS, color, -1)

            label = state if landmarks else "no hand"
            cv2.putText(canvas, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow("AirAim - fake cursor", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
