"""The later "placed" keyframe, an option of the keyframe extension (--keyframe-placed-later).

By default every keyframe is taken at the call at which the writer signals its event. With this option the keyframe of a
placing event waits for the hand: it is taken at the first call from the signal on at which the fingertips are at least
`height` above an object resting on the table, and at the latest at the `calls`-th call after the signal. Until then the
previous keyframe stays in the slot. A later event that arrives while it waits is taken at once, and the waiting one is
dropped. A placing event is the completion of a skill with one of the task's placing verbs (TaskSpec.placed_verbs: the verbs
of its set_down rules unless keyframe.placed_verbs says otherwise; "placed red" in the reference task, the second event).

Why: the simulator's place event fires when the hand is 6 cm above the cube, while the training keyframes of "placed red"
show the hand about 10 cm up, where the demonstrations' labels change; a slow retreat then gives a keyframe lower than any
seen in training (docs/methods/keyframe/README.md).

The fingertip height comes from the call's joint state (observation/state, the 7 arm joints first) and the arm's forward
kinematics (the memory policy's FrankaKinematics: the reference bench's Franka arm); an object resting on the table has its
centre at keyframe.rest_z of the task file (the reference 4 cm cube on the reference table: 0.76 m).
"""
from __future__ import annotations

import math

import numpy as np

from ..memory_policy.kinematics import FrankaKinematics

PLACED_EVENT = 2          # the placing event of a rule built without a task or placing verbs: the second, as in the reference task
HEIGHT = 0.10             # m: the fingertips above the placed cube
CALLS = 3                 # the keyframe is taken at the latest at the third call after the signal
CUBE_REST_Z = 0.76        # m: the centre of a 4 cm cube on the reference table (table top 0.74 m); the task file's keyframe.rest_z
STATE_KEYS = ("observation/state", "observation/joint_position")   # the red/blue layout, the droid layout


class PlacedLater:
    """The waiting rule of the placing events' keyframes. verbs: the verbs of the placing skills (the event is then every
    event whose skill has one of them); without verbs the placing event is the `event`-th. kinematics: any object with
    tcp_position(joints) -> (x, y, z)."""

    def __init__(self, height: float = HEIGHT, calls: int = CALLS, event: int = PLACED_EVENT, kinematics=None,
                 rest_z: float = CUBE_REST_Z, verbs=None):
        if not math.isfinite(height) or calls < 0:
            raise ValueError(f"placed-later height {height} / calls {calls}")
        self.height = float(height)
        self.calls = int(calls)
        self.event = int(event)
        self.verbs = None if verbs is None else tuple(verbs)
        self.kinematics = kinematics if kinematics is not None else FrankaKinematics()
        self.rest_z = float(rest_z)
        self.reset()

    @classmethod
    def for_task(cls, task, height: float = HEIGHT, calls: int = CALLS, **options) -> "PlacedLater":
        """The rule of a task: its placing verbs (TaskSpec.placed_verbs) and the resting height of its objects (keyframe.rest_z).
        The event of the config is the first placing event of the task order."""
        verbs = task.placed_verbs()
        event = next((number for number, skill in enumerate(task.skills, 1) if task.parse_skill(skill)[0] in verbs), PLACED_EVENT)
        return cls(height, calls, event=event, rest_z=(task.keyframe or {}).get("rest_z", CUBE_REST_Z), verbs=verbs, **options)

    def is_placing(self, events: int, writer, task) -> bool:
        """Whether the event with this number (the count of completed skills) is a placing event: its skill has one of the
        verbs (given verbs and a task), else it is the `event`-th."""
        if self.verbs is None or task is None:
            return events == self.event
        return task.parse_skill(writer.completed()[events - 1])[0] in self.verbs

    def reset(self) -> None:
        """No placing event is waiting (a new episode, or its keyframe was taken)."""
        self.pending = None       # the placing event was signalled, its keyframe not taken yet: {"signal_step", "calls"}

    def config(self) -> dict:
        """The (first) placing event, the height and the latest call, for the metadata."""
        return {"event": self.event, "height": self.height, "calls": self.calls}

    def fingertip_height(self, observation: dict) -> float | None:
        """The fingertips' height above a resting object (m), from the call's joint state; None without a usable one."""
        joints = None
        for key in STATE_KEYS:
            if observation.get(key) is not None:
                joints = np.asarray(observation[key], float).reshape(-1)
                break
        if joints is None or joints.size < 7 or not np.all(np.isfinite(joints[:7])):
            return None
        return float(self.kinematics.tcp_position(joints[:7])[2]) - self.rest_z

    def wait(self, step, height: float | None) -> str:
        """One call with a placing event signalled and its keyframe not taken: "height" (take it now, the hand is high
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
