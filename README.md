# AirAim

A webcam-based mouse. Your hand is the cursor. Move your hand to move the cursor (relative movement, like a real mouse), pinch to click, make a fist to "clutch" (recenter your hand without moving the cursor).

Work in progress. Goal: a smooth enough demo to control [Aim Labs](https://aimlabs.com/) Gridshot using just hand gestures via webcam.

## Status
Core input pipeline working — hand tracking, relative cursor movement, smooth acceleration, pinch to click, clutch gesture. Currently tuning against Aim Lab.

## Gestures
- **Open palm** — move your hand to move the cursor.
- **Pinch** (thumb to index) — left click.
- **Fist** — clutch: pauses the cursor so you can reposition your hand without moving it.

## How it works
The webcam feed goes to MediaPipe, which returns 21 hand landmarks per frame. Those landmarks drive two things: a gesture classifier (open palm / pinch / fist) and the cursor itself. Rather than mapping hand position straight to screen coordinates, we track the frame-to-frame movement of the index fingertip and feed that delta — smoothed with an exponential moving average and shaped by a speed-based acceleration curve — into `pydirectinput` as relative mouse movement. Relative deltas are what make the clutch possible: a fist simply stops the deltas from being sent, so you can recenter your hand mid-motion the same way you'd lift and reposition a real mouse.

## Tuning
The feel lives in a handful of constants — movement knobs in `cursor.py`, gesture thresholds in `gestures.py`:
- **SENSITIVITY_LOW** — multiplier for slow movement; lower means finer aim.
- **SENSITIVITY_HIGH** — multiplier at full speed; higher means faster full-screen traversal.
- **ACCEL_MAX_SPEED** — hand speed at which the multiplier reaches its high end.
- **EMA_ALPHA** — smoothing strength; lower is smoother, higher is snappier.
- **DEADZONE_PX** — minimum movement that registers, so a still hand doesn't drift.
- **PINCH_THRESHOLD** — how close thumb and index must be to count as a pinch.

## Setup
```bash
python -m venv .venv
# activate the venv, then:
pip install -r requirements.txt
```
