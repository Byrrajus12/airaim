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
# Pinch: thumb-tip <-> index-tip distance, divided by hand scale.
PINCH_THRESHOLD = 0.45
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


def is_pinch(lm):
    scale = _hand_scale(lm)
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / scale < PINCH_THRESHOLD


def classify(lm):
    """Return the gesture state string for one hand's landmarks."""
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
    if is_pinch(lm):
        return "pinch"

    # Open palm: all four fingers extended.
    if num_extended == 4:
        return "open_palm"

    return "unknown"
