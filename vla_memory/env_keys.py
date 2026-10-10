"""The environment's side of the contract: the memory/ control keys a simulator or robot sends with every policy call
(docs/ENVIRONMENT_CONTRACT.md, section 3), built from its raw signals, so that a new environment does not rebuild the per-step
buffers by hand.

    keys = ControlKeys(task)                    # one per environment lane; the task names its extra signals (task.signals)
    keys.start_episode("seed7")                 # the next call carries memory/new_episode
    for step in range(max_steps):
        keys.record(finger_width=w, finger_effort=e, tip_xyz=p,           # EVERY control step: what the robot measures now,
                    signals={"contact_force": f})                          # before the step's action
        if call_due:                                                       # a policy call at this step
            obs = keys.observation(robot_keys, instruction, step, exterior_raw=ext, wrist_raw=wri,
                                   gripper_closedness=c, hand_height=h)    # in simulation also oracle_prompt=...
            reply = client.infer(obs)

record() buffers one row per control step. call() returns the keys of the policy call at `step` and starts the next buffer;
observation() is the policy's own keys, the FULL instruction as `prompt`, and call():
    new_episode, episode, step     True at the first call after start_episode() (and at the very first call), the episode id
                                   (sent if given), the step
    oracle_prompt, exterior_raw, wrist_raw, gripper_closedness, hand_height    as given (frames made contiguous)
    finger_width, finger_effort, tip_xyz, tip_z    the values recorded at this step (the last record() since the previous call);
                                   tip_z is tip_xyz's z, or the given tip_z when the fingertip position is unknown
    proprio_steps                  float32 (n, 5): finger width, squeeze, fingertip x, y, z of every step recorded since the
                                   previous call, oldest first, this step last. When the first call of an episode comes at its first
                                   recorded step (the reference bench), the array is empty, (0, 5), and the values travel in the
                                   four single keys; if steps were recorded before it, they are all sent
    signal_steps                   {name: float32 (n,)}: the task's declared signals over the same steps, this step included, also
                                   at the first call of an episode (sent for a task that declares signals)
A key is sent only if its value is known and it is in `include` (default: every key of CONTROL_KEYS); a value not given is NaN
inside the per-step arrays. The reference bench (scripts/zeroshot_bench.py of the research code) builds the same keys by hand
(contract, section 5, "Adopting ControlKeys").
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

CONTROL_PREFIX = "memory/"
CONTROL_KEYS = ("new_episode", "episode", "step", "oracle_prompt", "exterior_raw", "wrist_raw", "gripper_closedness",
                "hand_height", "finger_width", "finger_effort", "tip_z", "tip_xyz", "proprio_steps", "signal_steps")
PROPRIO_COLUMNS = ("finger_width", "finger_effort", "tip_x", "tip_y", "tip_z")   # the columns of memory/proprio_steps
POINTS = {"tip": ("tip_x", "tip_y", "tip_z")}                                    # positions the rules writer measures distances of


def task_signals(task) -> tuple[str, ...]:
    """The extra signals a task file declares (`signals`: a list of names, or names with a description); () without one."""
    return tuple(getattr(task, "signals", None) or ())


def _num(x) -> float | None:
    """The value as a float; None when it was not given."""
    return None if x is None else float(x)


def _nan(x) -> float:
    """The value as a float; NaN when it was not given (the per-step arrays)."""
    return math.nan if x is None else float(x)


def _measured_now(finger_width, finger_effort, tip_xyz, tip_z) -> dict:
    """The robot's signals of one step as floats (None: not given); tip_z is tip_xyz's z when the fingertip position is known."""
    tip = None
    if tip_xyz is not None:
        tip = np.asarray(tip_xyz, float).reshape(3)
    if tip is not None:
        height = float(tip[2])
    else:
        height = _num(tip_z)
    return {"finger_width": _num(finger_width), "finger_effort": _num(finger_effort), "tip_xyz": tip, "tip_z": height}


def _proprio_row(now: dict) -> list[float]:
    """One row of memory/proprio_steps, in PROPRIO_COLUMNS order; a value not given is NaN."""
    tip = now["tip_xyz"]
    if tip is not None:
        xyz = [float(x) for x in tip]
    else:
        xyz = [math.nan, math.nan, _nan(now["tip_z"])]
    return [_nan(now["finger_width"]), _nan(now["finger_effort"]), *xyz]


class ControlKeys:
    """Builds the memory/ control keys of one environment lane (module docstring). task: the task (its `signals`) or None;
    signals: the extra signal names instead of the task's; include: the control keys to send (default all)."""

    def __init__(self, task=None, signals: Sequence[str] | None = None, include: Iterable[str] | None = None):
        self.signal_names = tuple(signals) if signals is not None else task_signals(task)
        clash = [n for n in self.signal_names if n in PROPRIO_COLUMNS or n in POINTS]
        if clash:
            raise ValueError(f"signal name(s) {clash} are robot signals of memory/proprio_steps already; choose other names")
        self.include = tuple(CONTROL_KEYS if include is None else include)
        unknown = [k for k in self.include if k not in CONTROL_KEYS]
        if unknown:
            raise ValueError(f"unknown control keys {unknown}; the contract has {CONTROL_KEYS}")
        self.start_episode()

    def start_episode(self, episode_id: Any = None):
        """A new episode: the next call() carries memory/new_episode True and the steps recorded from now on."""
        self.episode_id = episode_id
        self._first = True
        self._has_proprio = False   # whether any gripper or fingertip value was recorded in this episode
        self._reset_buffer()

    def _reset_buffer(self):
        """Forget the recorded steps: the next call() sends only the steps recorded from now on."""
        self._rows: list = []
        self._signal_rows: list = []
        self._now: dict | None = None

    def record(self, finger_width=None, finger_effort=None, tip_xyz=None, tip_z=None, signals: dict | None = None):
        """The robot's signals at one control step, observed before the step's action. Call it at EVERY control step, also at
        the steps with a policy call (before call()): every step must reach the rules writer exactly once. finger_width (m,
        both fingers), finger_effort (N, or a 1/0 grasped flag), tip_xyz (m, the fingertip midpoint) or only tip_z; signals:
        the task's declared signals by name (a missing one is NaN)."""
        now = _measured_now(finger_width, finger_effort, tip_xyz, tip_z)
        self._has_proprio = self._has_proprio or any(v is not None for v in now.values())
        self._rows.append(_proprio_row(now))
        self._signal_rows.append(self._signal_row(signals))
        self._now = now

    def _signal_row(self, signals: dict | None) -> list[float]:
        """One row of the task's declared signals, in signal_names order; a signal not given is NaN."""
        given = dict(signals or {})
        unknown = [k for k in given if k not in self.signal_names]
        if unknown:
            raise ValueError(f"unknown signal(s) {unknown}; declared: {list(self.signal_names)} (task file: signals)")
        return [_nan(given.get(name)) for name in self.signal_names]

    def call(self, step: int, *, exterior_raw=None, wrist_raw=None, gripper_closedness=None, hand_height=None,
             oracle_prompt=None) -> dict:
        """The memory/ keys of the policy call at `step` (module docstring); the next call's buffer starts empty."""
        values = {}
        values.update(self._episode_keys(step))
        values.update(self._given_keys(oracle_prompt, exterior_raw, wrist_raw, gripper_closedness, hand_height))
        values.update(self._current_step_keys())
        values.update(self._step_array_keys())
        self._first = False
        self._reset_buffer()
        return {CONTROL_PREFIX + k: x for k, x in values.items()}

    def _episode_keys(self, step: int) -> dict:
        """new_episode (True at the first call of an episode), episode (when an id was given) and step."""
        keys = {}
        if "new_episode" in self.include:
            keys["new_episode"] = self._first
        if "episode" in self.include and self.episode_id is not None:
            keys["episode"] = self.episode_id
        if "step" in self.include:
            keys["step"] = int(step)
        return keys

    def _given_keys(self, oracle_prompt, exterior_raw, wrist_raw, gripper_closedness, hand_height) -> dict:
        """The values passed to call(), as given: a value None is left out, the frames are made contiguous."""
        keys = {}
        if "oracle_prompt" in self.include and oracle_prompt is not None:
            keys["oracle_prompt"] = oracle_prompt
        if "exterior_raw" in self.include and exterior_raw is not None:
            keys["exterior_raw"] = np.ascontiguousarray(exterior_raw)
        if "wrist_raw" in self.include and wrist_raw is not None:
            keys["wrist_raw"] = np.ascontiguousarray(wrist_raw)
        if "gripper_closedness" in self.include and gripper_closedness is not None:
            keys["gripper_closedness"] = float(gripper_closedness)
        if "hand_height" in self.include and hand_height is not None:
            keys["hand_height"] = float(hand_height)
        return keys

    def _current_step_keys(self) -> dict:
        """finger_width, finger_effort, tip_z and tip_xyz of this step: the values of the last record() since the previous call."""
        now = self._now or {}
        keys = {}
        for name in ("finger_width", "finger_effort", "tip_z"):
            if name in self.include and now.get(name) is not None:
                keys[name] = now[name]
        if "tip_xyz" in self.include and now.get("tip_xyz") is not None:
            keys["tip_xyz"] = now["tip_xyz"].astype(np.float32)
        return keys

    def _step_array_keys(self) -> dict:
        """proprio_steps (when the robot has gripper or fingertip signals at all) and signal_steps (when the task declares
        signals), over every step recorded since the previous call."""
        keys = {}
        if "proprio_steps" in self.include and self._has_proprio:
            keys["proprio_steps"] = self._proprio_steps()
        if "signal_steps" in self.include and self.signal_names:
            keys["signal_steps"] = self._signal_steps()
        return keys

    def _proprio_steps(self) -> np.ndarray:
        """The recorded rows as float32 (n, 5), oldest first. The first call of an episode made at its first recorded step
        sends an empty array: the four single keys carry that row."""
        rows = self._rows
        if self._first and len(rows) <= 1:
            rows = []
        return np.asarray(rows, np.float32).reshape(-1, 5)

    def _signal_steps(self) -> dict[str, np.ndarray]:
        """The declared signals over the recorded steps: one float32 (n,) array per signal name."""
        table = np.asarray(self._signal_rows, np.float32).reshape(-1, len(self.signal_names))
        return {name: np.ascontiguousarray(table[:, column]) for column, name in enumerate(self.signal_names)}

    def observation(self, policy_keys: dict, instruction: str, step: int, **call_kw) -> dict:
        """The whole observation of a policy call: the policy's own keys, the FULL instruction as `prompt`, and call()."""
        obs = dict(policy_keys)
        obs["prompt"] = instruction
        obs.update(self.call(step, **call_kw))
        return obs
