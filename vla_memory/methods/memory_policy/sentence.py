"""The memory sentence of a stored moment: what the robot did during the moment's next HORIZON actions, in a few words.

    memory: down 6 cm, back 1 cm
    memory: release at step 1, up 2 cm
    memory: forward 2 cm, gripper stays closed
    memory: hold still

The policy was trained on exactly these sentences, so the rule must not change without relabelling the training data.

Movement: the displacement of the hand's TCP from the current joints to the last joint target of the chunk (forward
kinematics, kinematics.py), along the world axes of the bench: z up/down, y left/right, x forward/back. Whole centimetres;
a component under 1 cm is left out; the largest component first.

Gripper: closed at the START is read from the MEASURED finger width (below 6 cm, gripper.py): the policy changes its command
only at chunk boundaries, so the first command of a chunk never shows its own re-opening. Closed at the END is read from the
last command (closed at CLOSED_COMMAND or more; the commands run from 0 to 0.6). Then:

    closed at the start, open at the end      "release at step k", first   (k: the first open command, counted from 1)
    open at the start, closed at the end      "close at step k", last      (k: the first closed command)
    closed at the start and at the end        "gripper stays closed", last

A sentence without any words is "hold still". Every sentence begins with PREFIX.
"""
from __future__ import annotations

import numpy as np

from .gripper import OPEN_WIDTH
from .kinematics import FrankaKinematics

HORIZON = 8                 # actions per policy call (the bench executes 8 actions of every chunk)
CLOSED_COMMAND = 0.3        # a gripper command at or above it closes the gripper
PREFIX = "memory: "
AXES = ((2, "up", "down"), (1, "left", "right"), (0, "forward", "back"))   # world axis: words for + and -
FRANKA = FrankaKinematics()


def movement_words(q_now, chunk, arm=None) -> list[str]:
    """The TCP's displacement from the joints q_now to the chunk's last joint target, as words, largest first.
    arm: anything with tcp_position(joints) -> (x, y, z) in m (default: the Franka arm of the bench)."""
    if arm is None:
        arm = FRANKA
    chunk = np.asarray(chunk)
    shift = 100 * (arm.tcp_position(chunk[-1, :7]) - arm.tcp_position(q_now))   # cm
    parts = []
    for axis, positive, negative in AXES:
        size = abs(shift[axis])
        if size >= 1.0:
            direction = positive if shift[axis] > 0 else negative
            parts.append((size, f"{direction} {size:.0f} cm"))
    # largest first; equal sizes are ordered by their words, as in the training labels
    return [words for _, words in sorted(parts, reverse=True)]


def first_step(mask) -> int:
    """The step (counted from 1) of the first True in mask."""
    return int(np.argmax(mask)) + 1


def sentence(q_now, chunk, width_now: float, arm=None) -> str:
    """The memory sentence of one moment: q_now, its 7 joint angles; chunk, its next HORIZON actions (rows: 7 joint
    targets and the gripper command); width_now, the measured finger width at the moment (m, both fingers)."""
    chunk = np.asarray(chunk)
    words = movement_words(q_now, chunk, arm)
    closed_now = width_now < OPEN_WIDTH
    closed = chunk[:, 7] >= CLOSED_COMMAND
    if closed_now and not closed[-1]:
        words.insert(0, f"release at step {first_step(~closed)}")
    elif not closed_now and closed[-1]:
        words.append(f"close at step {first_step(closed)}")
    elif closed[-1]:
        words.append("gripper stays closed")
    if not words:
        return PREFIX + "hold still"
    return PREFIX + ", ".join(words)


def is_release(text: str) -> bool:
    """Whether a memory sentence begins with a release (the entries a gripper closed on nothing is shown)."""
    return text.startswith(PREFIX + "release")
