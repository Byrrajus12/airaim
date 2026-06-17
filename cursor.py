"""Cursor control from hand motion.

Two modes share the same motion math (CursorMotion); only the output differs:

- run_cursor_sandbox: moves a circle on a blank window, nothing touches the OS
- run_live: drives the real OS cursor with pydirectinput, pinch = left click

Movement comes from the index fingertip's frame-to-frame delta, smoothed with
an EMA and gated by a deadzone and sensitivity. It only runs on an open palm;
other gestures pause it, so you can reposition your hand (a fist clutch)
without dragging the cursor. A brief hand dropout coasts to a stop instead of
freezing.
"""

import math
import time

import cv2
import numpy as np
import pydirectinput

import gestures
from tracker import HandTracker

# --- Tuning knobs (tweak these live) ---
EMA_ALPHA = 0.3      # delta smoothing, 0..1. Higher = snappier, lower = smoother.
DEADZONE_PX = 2.0    # ignore fingertip movement smaller than this (camera pixels)
# Acceleration: slow hand = precision, fast hand = big traversals. The
# multiplier ramps smoothly from LOW to HIGH as speed climbs to ACCEL_MAX_SPEED.
SENSITIVITY_LOW = 3.0   # multiplier for slow, precise movement
SENSITIVITY_HIGH = 3.0   # multiplier at full speed for fast traversal
ACCEL_MAX_SPEED = 10.0   # camera px/frame where the multiplier reaches HIGH
TRANSITION_SKIP_FRAMES = 3   # frames to ignore right after the hand opens, so
                             # the uncurl from a fist/pinch isn't read as motion
NO_HAND_GRACE_FRAMES = 10    # max frames to keep coasting after the hand is lost
VELOCITY_DECAY = 0.7         # per-frame shrink of coasting velocity while lost

WINDOW_W, WINDOW_H = 960, 540
CIRCLE_RADIUS = 12


def _accelerate(dx, dy):
    """Scale a smoothed delta by speed: gentle when slow, punchy when fast."""
    t = min(math.hypot(dx, dy) / ACCEL_MAX_SPEED, 1.0)
    mult = SENSITIVITY_LOW + t * (SENSITIVITY_HIGH - SENSITIVITY_LOW)
    return dx * mult, dy * mult


class CursorMotion:
    """Turns the per-frame hand state into a smoothed cursor delta.

    Holds the smoothing, clutch, transition-skip and coast state so every
    mode behaves identically. Call step() once per frame; it returns the
    (dx, dy) to apply this frame, already scaled by sensitivity.
    """

    def __init__(self):
        self.prev_tip = None       # last fingertip position (camera px)
        self.prev_state = "unknown"
        self.skip_frames = 0       # countdown while ignoring uncurl motion
        self.no_hand_frames = 0    # consecutive frames with no hand seen
        self.ema_dx = 0.0
        self.ema_dy = 0.0

    def step(self, state, tip):
        # The moment the hand opens, hold off for a few frames: the finger is
        # still uncurling, and that motion isn't a real cursor move.
        if state == "open_palm" and self.prev_state != "open_palm":
            self.skip_frames = TRANSITION_SKIP_FRAMES

        active = (state == "open_palm" and tip is not None
                  and self.prev_tip is not None and self.skip_frames == 0)

        out_dx = out_dy = 0.0
        if active:
            self.no_hand_frames = 0
            dx = tip[0] - self.prev_tip[0]
            dy = tip[1] - self.prev_tip[1]

            # Deadzone: drop tiny jitter so a still hand doesn't drift.
            if abs(dx) < DEADZONE_PX:
                dx = 0.0
            if abs(dy) < DEADZONE_PX:
                dy = 0.0

            # Smooth the delta, then scale it through the acceleration curve.
            self.ema_dx = EMA_ALPHA * dx + (1 - EMA_ALPHA) * self.ema_dx
            self.ema_dy = EMA_ALPHA * dy + (1 - EMA_ALPHA) * self.ema_dy
            out_dx, out_dy = _accelerate(self.ema_dx, self.ema_dy)
        elif tip is None:
            # Hand lost: keep coasting on the last velocity, decaying it each
            # frame, so a brief dropout glides to a stop instead of freezing.
            # Stops once it's slow enough or we've waited too long.
            self.no_hand_frames += 1
            self.ema_dx *= VELOCITY_DECAY
            self.ema_dy *= VELOCITY_DECAY
            if (self.no_hand_frames <= NO_HAND_GRACE_FRAMES
                    and math.hypot(self.ema_dx, self.ema_dy) >= DEADZONE_PX):
                out_dx, out_dy = _accelerate(self.ema_dx, self.ema_dy)
            else:
                self.ema_dx = self.ema_dy = 0.0
        else:
            # Hand present but paused (clutch or transition): forget momentum
            # so it doesn't lurch when tracking resumes.
            self.no_hand_frames = 0
            self.ema_dx = self.ema_dy = 0.0
            if self.skip_frames > 0:
                self.skip_frames -= 1

        # Always remember where the fingertip is, so the next active frame
        # measures movement from here. This is what makes the clutch work:
        # you can move your hand while paused without it counting.
        self.prev_tip = tip
        self.prev_state = state
        return out_dx, out_dy


def _fingertip(landmarks, w, h):
    """Index fingertip in camera pixels, or None if there's no hand."""
    if not landmarks:
        return None
    lx, ly, _ = landmarks[gestures.INDEX_TIP]
    return (lx * w, ly * h)


def _move_circle(cx, cy, dx, dy):
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
    motion = CursorMotion()
    cx, cy = WINDOW_W / 2, WINDOW_H / 2
    last_state = "unknown"
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
            state = gestures.classify(landmarks, last_state == "pinch") if landmarks else "unknown"
            last_state = state

            dx, dy = motion.step(state, _fingertip(landmarks, w, h))
            cx, cy = _move_circle(cx, cy, dx, dy)

            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), np.uint8)
            color = (0, 255, 0) if state == "open_palm" else (0, 0, 255)
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


def run_live(camera_index=0):
    """Drive the real OS cursor; pinch to left click. Keeps the webcam open."""
    # pydirectinput sleeps after every call by default, which would throttle
    # the loop; we pace it ourselves off the webcam instead.
    pydirectinput.PAUSE = 0
    # Hitting a screen corner is normal cursor usage here, not a runaway script.
    pydirectinput.FAILSAFE = False

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera_index}")
        return

    tracker = HandTracker()
    motion = CursorMotion()
    last_state = "unknown"
    start = time.time()

    print("AirAim live. Cursor is hand-controlled. Press 'q' in the window to quit.")
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
            landmarks, result = tracker.process(frame, timestamp_ms)
            state = gestures.classify(landmarks, last_state == "pinch") if landmarks else "unknown"

            dx, dy = motion.step(state, _fingertip(landmarks, w, h))
            idx, idy = round(dx), round(dy)
            if idx or idy:
                # relative=True sends raw relative movement (no absolute
                # coordinate math), which behaves on multi-monitor setups and
                # is what games read for raw mouse input.
                pydirectinput.moveRel(idx, idy, relative=True)

            # Click once on the rising edge of a pinch, not every frame it holds.
            if state == "pinch" and last_state != "pinch":
                pydirectinput.click()
            last_state = state

            tracker.draw(frame, result)
            color = (0, 255, 0) if state == "open_palm" else (0, 0, 255)
            label = state if landmarks else "no hand"
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow("AirAim - live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
