"""Labelling recorded demonstrations for the exemplar store.

The reference demos store a waypoint `phase` per frame (the scripted demo generator's segment index). From the
phase and the position within the phase run, PhaseLabeller derives per frame: the state of every object (resting, held at
its resting height, held lifted: the task's state_roles, in the reference task on_table, held_at_table and held_lifted),
the gripper state (from the recorded closedness), whether the hand is high above the table, and the skill index that was
active. The phase table and the shares of a phase run at which an object changes state come from the task file
(`demo_phases`).

Any other source of labels works too: a labeller is just a function (episode_dict, frame_index) -> label dict with
the task's object fields, 'gripper', 'gripper_high_above_table' and 'subgoal'.
"""
from __future__ import annotations

import pathlib

import numpy as np


STATE_SHARES = {"close": 0.66, "lift": 0.15, "lower": 0.6, "open": 0.33}   # as measured on the reference demonstrations


def parse_ranges(spec: dict) -> dict:
    """{"0-3": 0, "4-6": 1} -> {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, ...}"""
    by_phase = {}
    for key, value in spec.items():
        first, _, last = str(key).partition("-")
        first = int(first)
        last = int(last or first)
        for phase in range(first, last + 1):
            by_phase[phase] = int(value)
    return by_phase


def parse_episode_ids(text: str) -> list[int]:
    """The --episodes option of the scripts, "0-19,25,30-31", as the list of episode ids [0, 1, ..., 19, 25, 30, 31]."""
    ids = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        ids += list(range(int(first), int(last or first) + 1))
    return ids


def _object_state(phase: int, fraction: float, object_phases: dict, states: tuple, shares: dict) -> str:
    """The state of one object in `phase`, `fraction` of the way through the phase run, from the object's own close, lift,
    lower and open phases: resting, held at its resting height, or held lifted (`states`: the task's names of the three);
    resting in every other phase. `shares`: the share of the phase run at which the state changes."""
    rest, held_low, held_high = states
    if phase == object_phases["close"]:
        if fraction > shares["close"]:
            return held_low
        return rest
    if phase == object_phases["lift"]:
        if fraction > shares["lift"]:
            return held_high
        return held_low
    if phase == object_phases["lower"]:
        if fraction < shares["lower"]:
            return held_high
        return held_low
    if phase == object_phases["open"]:
        if fraction < shares["open"]:
            return held_low
        return rest
    return rest


class PhaseLabeller:
    """Labels of one frame of a recorded demo from its waypoint phase (module docstring); the phase table is the task's
    `demo_phases`, the gripper state comes from the recorded closedness (the column state_gripper_index of the state)."""

    def __init__(self, task, closed_threshold: float | None = None, state_gripper_index: int = 7):
        phases = task.demo_phases
        self.task = task
        self.phase_to_skill = parse_ranges(phases["phase_to_skill"])
        self.per_object = {object_name: dict(object_phases) for object_name, object_phases in phases["per_object"].items()}
        self.high = set(phases.get("high_phases", []))
        self.low = set(phases.get("low_phases", []))
        self.rising = set(phases.get("rising_phases", []))
        if closed_threshold is None:
            closed_threshold = task.proprio.get("closed_threshold", 0.3)
        self.closed_threshold = closed_threshold
        self.states = tuple(task.state_role(role) for role in ("rest", "held_low", "held_high"))   # the task's names of the object states
        self.shares = {**STATE_SHARES, **phases.get("state_shares", {})}
        self.gripper_column = state_gripper_index   # column of the recorded state vector holding the gripper closedness

    def __call__(self, episode: dict, frame: int) -> dict:
        """The labels of one frame: the object states, the gripper, the hand height, the skill index and the phase."""
        phases = np.asarray(episode["phase"]).astype(int)
        phase = int(phases[frame])
        run = np.flatnonzero(phases == phase)
        fraction = (frame - run[0]) / max(len(run) - 1, 1)
        closedness = float(np.asarray(episode["state"])[frame, self.gripper_column])
        labels = {self.task.field(object_name): _object_state(phase, fraction, object_phases, self.states, self.shares)
                  for object_name, object_phases in self.per_object.items()}
        if closedness > self.closed_threshold:
            gripper = "closed"
        else:
            gripper = "open"
        labels.update(gripper=gripper, gripper_high_above_table=self._hand_is_high(phase, fraction),
                      subgoal=self.phase_to_skill[phase], phase=phase)
        return labels

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
    for episode_id in episode_ids:
        archive = np.load(pathlib.Path(demo_dir) / f"episode_{episode_id:04d}.npz")
        episode = {camera: archive[camera] for camera in camera_names}
        episode.update(state=archive["state"], phase=archive["phase"], episode=int(episode_id))
        if "instruction" in archive.files:
            episode["instruction"] = str(archive["instruction"])
        yield episode
