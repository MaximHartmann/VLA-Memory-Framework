"""The gripper's state from the measured finger width.

The finger width is what the robot measures (m, both fingers together: the sum of the two finger joints). On the reference
Franka hand with 4 cm cubes it tells three states apart:

    open             at or above 6 cm (fully open is 8 cm)
    closed_object    below 6 cm, at or above 3.4 cm: the fingers rest on an object (on a cube about 3.7 cm)
    closed_nothing   below 3.4 cm: the fingers closed on nothing (they stop at about 3.2 cm)

The retrieval (retrieve.py) keeps the open and the closed moments apart and shows a gripper closed on nothing only moments
that release; the memory sentence (sentence.py) reads "closed at the start" from the same 6 cm threshold.
"""
from __future__ import annotations

OPEN_WIDTH = 0.06        # m: the gripper is open at or above
EMPTY_WIDTH = 0.034      # m: the gripper is closed on nothing below
STATES = ("open", "closed_object", "closed_nothing")


def gripper_state(width: float) -> str:
    """open / closed_object / closed_nothing from the measured finger width (m, both fingers together)."""
    if width >= OPEN_WIDTH:
        return "open"
    if width >= EMPTY_WIDTH:
        return "closed_object"
    return "closed_nothing"
