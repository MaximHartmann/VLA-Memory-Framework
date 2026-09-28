"""Labelling recorded demonstrations for the exemplar store.

The reference demos store a waypoint `phase` per frame (the scripted demo generator's segment index). From the
phase and the position within the phase run, PhaseLabeller derives per frame: the state of every object (on the
table, held at the table, held lifted), the gripper state (from the recorded closedness), whether the hand is high
above the table, and the skill index that was active. The phase table comes from the task file (`demo_phases`).

Any other source of labels works too: a labeller is just a function (episode_dict, frame_index) -> label dict with
the task's object fields, 'gripper', 'gripper_high_above_table' and 'subgoal'.
"""
from __future__ import annotations

import numpy as np


def parse_ranges(spec: dict) -> dict:
    """{"0-3": 0, "4-6": 1} -> {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, ...}"""
    out = {}
    for k, v in spec.items():
        lo, _, hi = str(k).partition("-"); lo = int(lo); hi = int(hi or lo)
        for p in range(lo, hi + 1):
            out[p] = int(v)
    return out


class PhaseLabeller:
    def __init__(self, task, closed_threshold: float | None = None, state_gripper_index: int = 7):
        d = task.demo_phases
        self.task = task
        self.phase_to_skill = parse_ranges(d["phase_to_skill"])
        self.per_object = {o: dict(v) for o, v in d["per_object"].items()}
        self.high = set(d.get("high_phases", [])); self.low = set(d.get("low_phases", [])); self.rising = set(d.get("rising_phases", []))
        self.closed = closed_threshold if closed_threshold is not None else task.proprio.get("closed_threshold", 0.3)
        self.gidx = state_gripper_index   # column of the recorded state vector holding the gripper closedness

    def __call__(self, ep: dict, i: int) -> dict:
        phase = np.asarray(ep["phase"]).astype(int); ph = int(phase[i])
        run = np.flatnonzero(phase == ph); frac = (i - run[0]) / max(len(run) - 1, 1)
        closedness = float(np.asarray(ep["state"])[i, self.gidx])

        def obj_state(p):
            if ph == p["close"]: return "held_at_table" if frac > 0.66 else "on_table"
            if ph == p["lift"]: return "held_lifted" if frac > 0.15 else "held_at_table"
            if ph == p["lower"]: return "held_lifted" if frac < 0.6 else "held_at_table"
            if ph == p["open"]: return "held_at_table" if frac < 0.33 else "on_table"
            return "on_table"

        if ph in self.high: high = True
        elif ph in self.low: high = False
        elif ph in self.rising: high = frac > 0.5
        else: high = frac < 0.5
        out = {self.task.field(o): obj_state(p) for o, p in self.per_object.items()}
        out.update(gripper="closed" if closedness > self.closed else "open", gripper_high_above_table=bool(high),
                   subgoal=self.phase_to_skill[ph], phase=ph)
        return out


def load_npz_episodes(demo_dir, episode_ids, camera_names=("exterior", "wrist")):
    """Generator over the reference demo format: episode_NNNN.npz with camera stacks, 'state' and 'phase'."""
    import pathlib
    for ep in episode_ids:
        z = np.load(pathlib.Path(demo_dir) / f"episode_{ep:04d}.npz")
        d = {c: z[c] for c in camera_names}
        d.update(state=z["state"], phase=z["phase"], episode=int(ep))
        if "instruction" in z.files:
            d["instruction"] = str(z["instruction"])
        yield d
