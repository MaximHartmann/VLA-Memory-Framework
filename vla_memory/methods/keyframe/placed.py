"""The later "placed" keyframe, an option of the keyframe extension (--keyframe-placed-later).

By default every keyframe is taken at the call at which the writer signals its event. With this option the keyframe of the
placing event (event 2, "placed red" in the reference task) waits for the hand: it is taken at the first call from the signal
on at which the fingertips are at least `height` above a cube resting on the table, and at the latest at the `calls`-th call
after the signal. Until then the previous keyframe stays in the slot. A later event that arrives while it waits is taken at
once, and the waiting one is dropped.

Why: the simulator's place event fires when the hand is 6 cm above the cube, while the training keyframes of "placed red"
show the hand about 10 cm up, where the demonstrations' labels change; a slow retreat then gives a keyframe lower than any
seen in training (docs/methods/keyframe/README.md).

The fingertip height comes from the call's joint state (observation/state, the 7 arm joints first) and the arm's forward
kinematics (the memory policy's FrankaKinematics: the reference bench's Franka arm); a cube resting on the reference table
has its centre at 0.76 m.
"""
from __future__ import annotations

import math

import numpy as np

from ..memory_policy.kinematics import FrankaKinematics

PLACED_EVENT = 2          # the second event of the reference task: placed red
HEIGHT = 0.10             # m: the fingertips above the placed cube
CALLS = 3                 # the keyframe is taken at the latest at the third call after the signal
CUBE_REST_Z = 0.76        # m: the centre of a 4 cm cube on the reference table (table top 0.74 m)
STATE_KEYS = ("observation/state", "observation/joint_position")   # the red/blue layout, the droid layout


class PlacedLater:
    """The waiting rule of the placing event's keyframe. kinematics: any object with tcp_position(joints) -> (x, y, z)."""

    def __init__(self, height: float = HEIGHT, calls: int = CALLS, event: int = PLACED_EVENT, kinematics=None,
                 rest_z: float = CUBE_REST_Z):
        if not math.isfinite(height) or calls < 0:
            raise ValueError(f"placed-later height {height} / calls {calls}")
        self.height = float(height)
        self.calls = int(calls)
        self.event = int(event)
        self.kinematics = kinematics if kinematics is not None else FrankaKinematics()
        self.rest_z = float(rest_z)
        self.reset()

    def reset(self) -> None:
        """No placing event is waiting (a new episode, or its keyframe was taken)."""
        self.pending = None       # the placing event was signalled, its keyframe not taken yet: {"signal_step", "calls"}

    def config(self) -> dict:
        """The event, the height and the latest call, for the metadata."""
        return {"event": self.event, "height": self.height, "calls": self.calls}

    def fingertip_height(self, observation: dict) -> float | None:
        """The fingertips' height above a resting cube (m), from the call's joint state; None without a usable one."""
        joints = None
        for key in STATE_KEYS:
            if observation.get(key) is not None:
                joints = np.asarray(observation[key], float).reshape(-1)
                break
        if joints is None or joints.size < 7 or not np.all(np.isfinite(joints[:7])):
            return None
        return float(self.kinematics.tcp_position(joints[:7])[2]) - self.rest_z

    def wait(self, step, height: float | None) -> str:
        """One call with the placing event signalled and its keyframe not taken: "height" (take it now, the hand is high
        enough), "fallback" (take it now, the last call to wait for) or "waiting"."""
        if self.pending is None:
            self.pending = {"signal_step": step, "calls": 0}
        else:
            self.pending["calls"] += 1
        if height is not None and height >= self.height:
            return "height"
        if self.pending["calls"] >= self.calls:
            return "fallback"
        return "waiting"
