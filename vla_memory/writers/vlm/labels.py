"""Labelling recorded demonstrations for the exemplar store.

The reference demos store a waypoint `phase` per frame (the scripted demo generator's segment index). From the
phase and the position within the phase run, PhaseLabeller derives per frame: the state of every object (on the
table, held at the table, held lifted), the gripper state (from the recorded closedness), whether the hand is high
above the table, and the skill index that was active. The phase table comes from the task file (`demo_phases`).

Any other source of labels works too: a labeller is just a function (episode_dict, frame_index) -> label dict with
the task's object fields, 'gripper', 'gripper_high_above_table' and 'subgoal'.
"""
from __future__ import annotations

import pathlib

import numpy as np


def parse_ranges(spec: dict) -> dict:
    """{"0-3": 0, "4-6": 1} -> {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, ...}"""
    out = {}
    for key, value in spec.items():
        lo, _, hi = str(key).partition("-")
        lo = int(lo)
        hi = int(hi or lo)
        for phase in range(lo, hi + 1):
            out[phase] = int(value)
    return out


def parse_episode_ids(text: str) -> list[int]:
    """The --episodes option of the scripts, "0-19,25,30-31", as the list of episode ids [0, 1, ..., 19, 25, 30, 31]."""
    ids = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        ids += list(range(int(lo), int(hi or lo) + 1))
    return ids


def _object_state(phase: int, fraction: float, object_phases: dict) -> str:
    """The state of one object in `phase`, `fraction` of the way through the phase run, from the object's own close, lift,
    lower and open phases; on the table in every other phase."""
    if phase == object_phases["close"]:
        if fraction > 0.66:
            return "held_at_table"
        return "on_table"
    if phase == object_phases["lift"]:
        if fraction > 0.15:
            return "held_lifted"
        return "held_at_table"
    if phase == object_phases["lower"]:
        if fraction < 0.6:
            return "held_lifted"
        return "held_at_table"
    if phase == object_phases["open"]:
        if fraction < 0.33:
            return "held_at_table"
        return "on_table"
    return "on_table"


class PhaseLabeller:
    """Labels of one frame of a recorded demo from its waypoint phase (module docstring); the phase table is the task's
    `demo_phases`, the gripper state comes from the recorded closedness."""

    def __init__(self, task, closed_threshold: float | None = None, state_gripper_index: int = 7):
        phases = task.demo_phases
        self.task = task
        self.phase_to_skill = parse_ranges(phases["phase_to_skill"])
        self.per_object = {o: dict(v) for o, v in phases["per_object"].items()}
        self.high = set(phases.get("high_phases", []))
        self.low = set(phases.get("low_phases", []))
        self.rising = set(phases.get("rising_phases", []))
        if closed_threshold is None:
            closed_threshold = task.proprio.get("closed_threshold", 0.3)
        self.closed = closed_threshold
        self.gidx = state_gripper_index   # column of the recorded state vector holding the gripper closedness

    def __call__(self, ep: dict, i: int) -> dict:
        phases = np.asarray(ep["phase"]).astype(int)
        phase = int(phases[i])
        run = np.flatnonzero(phases == phase)
        fraction = (i - run[0]) / max(len(run) - 1, 1)
        closedness = float(np.asarray(ep["state"])[i, self.gidx])
        out = {self.task.field(o): _object_state(phase, fraction, p) for o, p in self.per_object.items()}
        if closedness > self.closed:
            gripper = "closed"
        else:
            gripper = "open"
        out.update(gripper=gripper, gripper_high_above_table=self._hand_is_high(phase, fraction),
                   subgoal=self.phase_to_skill[phase], phase=phase)
        return out

    def _hand_is_high(self, phase: int, fraction: float) -> bool:
        """Whether the hand is high above the table: fixed for the high and the low phases; a rising phase becomes high
        halfway through, every other (descending) phase stops being high halfway through."""
        if phase in self.high:
            return True
        if phase in self.low:
            return False
        if phase in self.rising:
            return bool(fraction > 0.5)
        return bool(fraction < 0.5)


def load_npz_episodes(demo_dir, episode_ids, camera_names=("exterior", "wrist")):
    """Generator over the reference demo format: episode_NNNN.npz with camera stacks, 'state' and 'phase'."""
    for ep in episode_ids:
        z = np.load(pathlib.Path(demo_dir) / f"episode_{ep:04d}.npz")
        d = {c: z[c] for c in camera_names}
        d.update(state=z["state"], phase=z["phase"], episode=int(ep))
        if "instruction" in z.files:
            d["instruction"] = str(z["instruction"])
        yield d
