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
    return None if x is None else float(x)


def _nan(x) -> float:
    return math.nan if x is None else float(x)


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
        self.episode_id = episode_id; self._first = True; self._grasp = False
        self._rows: list = []; self._sig: list = []; self._now: dict | None = None

    def record(self, finger_width=None, finger_effort=None, tip_xyz=None, tip_z=None, signals: dict | None = None):
        """The robot's signals at one control step, observed before the step's action. Call it at EVERY control step, also at
        the steps with a policy call (before call()): every step must reach the rules writer exactly once. finger_width (m,
        both fingers), finger_effort (N, or a 1/0 grasped flag), tip_xyz (m, the fingertip midpoint) or only tip_z; signals:
        the task's declared signals by name (a missing one is NaN)."""
        tip = None if tip_xyz is None else np.asarray(tip_xyz, float).reshape(3)
        now = {"finger_width": _num(finger_width), "finger_effort": _num(finger_effort), "tip_xyz": tip,
               "tip_z": float(tip[2]) if tip is not None else _num(tip_z)}
        self._grasp = self._grasp or any(v is not None for v in now.values())
        xyz = [float(x) for x in tip] if tip is not None else [math.nan, math.nan, _nan(now["tip_z"])]
        self._rows.append([_nan(now["finger_width"]), _nan(now["finger_effort"]), *xyz])
        sig = dict(signals or {})
        unknown = [k for k in sig if k not in self.signal_names]
        if unknown:
            raise ValueError(f"unknown signal(s) {unknown}; declared: {list(self.signal_names)} (task file: signals)")
        self._sig.append([_nan(sig.get(n)) for n in self.signal_names])
        self._now = now

    def call(self, step: int, *, exterior_raw=None, wrist_raw=None, gripper_closedness=None, hand_height=None,
             oracle_prompt=None) -> dict:
        """The memory/ keys of the policy call at `step` (module docstring); the next call's buffer starts empty."""
        inc, now, v = self.include, self._now or {}, {}
        if "new_episode" in inc: v["new_episode"] = self._first
        if "episode" in inc and self.episode_id is not None: v["episode"] = self.episode_id
        if "step" in inc: v["step"] = int(step)
        if "oracle_prompt" in inc and oracle_prompt is not None: v["oracle_prompt"] = oracle_prompt
        if "exterior_raw" in inc and exterior_raw is not None: v["exterior_raw"] = np.ascontiguousarray(exterior_raw)
        if "wrist_raw" in inc and wrist_raw is not None: v["wrist_raw"] = np.ascontiguousarray(wrist_raw)
        if "gripper_closedness" in inc and gripper_closedness is not None: v["gripper_closedness"] = float(gripper_closedness)
        if "hand_height" in inc and hand_height is not None: v["hand_height"] = float(hand_height)
        for k in ("finger_width", "finger_effort", "tip_z"):
            if k in inc and now.get(k) is not None:
                v[k] = now[k]
        if "tip_xyz" in inc and now.get("tip_xyz") is not None: v["tip_xyz"] = now["tip_xyz"].astype(np.float32)
        if "proprio_steps" in inc and self._grasp:   # the first call at the first step: the single keys carry the row
            v["proprio_steps"] = np.asarray([] if self._first and len(self._rows) <= 1 else self._rows, np.float32).reshape(-1, 5)
        if "signal_steps" in inc and self.signal_names:
            a = np.asarray(self._sig, np.float32).reshape(-1, len(self.signal_names))
            v["signal_steps"] = {n: np.ascontiguousarray(a[:, j]) for j, n in enumerate(self.signal_names)}
        self._first = False; self._rows = []; self._sig = []; self._now = None
        return {CONTROL_PREFIX + k: x for k, x in v.items()}

    def observation(self, policy_keys: dict, instruction: str, step: int, **call_kw) -> dict:
        """The whole observation of a policy call: the policy's own keys, the FULL instruction as `prompt`, and call()."""
        obs = dict(policy_keys); obs["prompt"] = instruction
        obs.update(self.call(step, **call_kw))
        return obs
