# AirAim

A webcam-based mouse. Your hand is the cursor. Move your hand to move the cursor (relative movement, like a real mouse), pinch to click, make a fist to "clutch" (recenter your hand without moving the cursor).

Work in progress. Goal: a smooth enough demo to control [Aim Labs](https://aimlabs.com/) Gridshot using just hand gestures via webcam.

## Status
Core pipeline complete and tuned against Aim Lab Gridshot.

## Gestures
- **Open palm** — move your hand to move the cursor.
- **Pinch** (thumb to index) — left click.
- **Fist** — clutch: pauses the cursor so you can reposition your hand without moving it.

## How it works
Webcam frames run through MediaPipe's hand landmark model, which tracks 21 points on the hand in real time. Those landmarks feed a gesture classifier that resolves one of three states — open palm (move), pinch (click), fist (clutch); the pinch test uses hysteresis, with separate thresholds for entering and exiting, so hovering near the boundary doesn't flicker between states.

Cursor movement is delta-based rather than an absolute position mapping: each frame measures how much the hand moved, not where it is, which is what makes it feel like a mouse instead of a touchscreen. That delta runs through a smooth acceleration curve — slow hand movement maps to precise, low-sensitivity output and fast movement to high-sensitivity traversal, so you can aim carefully and reposition quickly without touching a setting — and an exponential moving average smooths out per-frame jitter. A fist clutches: it pauses cursor output without losing position, so you can reposition your hand freely, the same as lifting a mouse off the pad. The result goes out through `pydirectinput` as raw relative mouse movement, which is what games expect for camera control.

## Tuning
The feel lives in a handful of constants — movement knobs in `cursor.py`, gesture thresholds in `gestures.py`:
- **SENSITIVITY_LOW** — multiplier for slow movement; lower means finer aim.
- **SENSITIVITY_HIGH** — multiplier at full speed; higher means faster full-screen traversal.
- **ACCEL_MAX_SPEED** — hand speed at which the multiplier reaches its high end.
- **EMA_ALPHA** — smoothing strength; lower is smoother, higher is snappier.
- **DEADZONE_PX** — minimum movement that registers, so a still hand doesn't drift.
- **PINCH_ON_THRESHOLD / PINCH_OFF_THRESHOLD** — how close thumb and index must be to start a pinch, and how far apart to end it; the gap stops flicker.

## Setup
```bash
python -m venv .venv
# activate the venv, then:
pip install -r requirements.txt
```
