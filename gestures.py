"""Gesture classification from MediaPipe hand landmarks.

Classifies a single hand as open_palm / pinch / fist / unknown.

Landmarks are expected as a sequence of 21 (x, y, z) tuples in MediaPipe's
normalized image coordinates (x, y in 0..1, origin top-left). Only x and y
are used here.

Thresholds live at the top as plain constants so they're easy to tune live.
"""

import math

# --- MediaPipe Hand landmark indices (subset we care about) ---
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# (pip_joint, tip) pairs for the four non-thumb fingers
FINGERS = [
    (INDEX_PIP, INDEX_TIP),
    (MIDDLE_PIP, MIDDLE_TIP),
    (RING_PIP, RING_TIP),
    (PINKY_PIP, PINKY_TIP),
]

# --- Tunable thresholds (normalized to hand size, so scale-invariant) ---
# Pinch uses hysteresis on the thumb-tip <-> index-tip distance: enter pinch
# only when closer than ON, exit only when farther than OFF. The gap between
# them is a dead band that stops flicker when hovering near the boundary.
PINCH_ON_THRESHOLD = 0.25
PINCH_OFF_THRESHOLD = 0.35
# A finger counts as "extended" when its tip is this much farther from the
# wrist than its PIP joint (ratio of distances, >1 means extended).
EXTEND_RATIO = 1.15


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hand_scale(lm):
    """A rotation-invariant size estimate: wrist -> middle-finger MCP."""
    return _dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6


def _finger_extended(lm, pip_idx, tip_idx):
    """True if the finger is extended (tip reaches past its PIP, away from wrist).

    Uses wrist-relative distances rather than raw y so it tolerates a tilted
    or rotated hand instead of assuming an upright palm.
    """
    wrist = lm[WRIST]
    return _dist(wrist, lm[tip_idx]) > _dist(wrist, lm[pip_idx]) * EXTEND_RATIO


def is_pinch(lm, in_pinch=False):
    """Hysteretic pinch test: the threshold loosens once already pinching."""
    scale = _hand_scale(lm)
    threshold = PINCH_OFF_THRESHOLD if in_pinch else PINCH_ON_THRESHOLD
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / scale < threshold


def classify(lm, in_pinch=False):
    """Return the gesture state string for one hand's landmarks.

    Pass the previous frame's pinch state as in_pinch so the pinch boundary
    has hysteresis and doesn't flicker.
    """
    if lm is None or len(lm) < 21:
        return "unknown"

    extended = [_finger_extended(lm, pip, tip) for pip, tip in FINGERS]
    num_extended = sum(extended)

    # Fist first: no fingers extended. Checked before pinch because in a real
    # closed fist the thumb often rests against the index, which would
    # otherwise read as a pinch.
    if num_extended == 0:
        return "fist"

    # Pinch: thumb and index pad together while at least one finger is out.
    if is_pinch(lm, in_pinch):
        return "pinch"

    # Open palm: all four fingers extended.
    if num_extended == 4:
        return "open_palm"

    return "unknown"
