"""The environment's per-step signal arrays, in the form the rules writer reads them.

memory/proprio_steps carries one row per control step since the previous policy call: finger width, squeeze, fingertip
x, y, z. memory/signal_steps carries the task's declared signals over the same steps, one column per signal.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_NAN_ROW = np.full(5, np.nan)   # the gripper row of a control step that sent no gripper signals


def signal_rows(ctrl: dict) -> np.ndarray:
    """The environment's signals of one policy call as rows (n, 5): width, squeeze, x, y, z.
    memory/proprio_steps carries every control step since the last call; the first call of an episode may only have the
    current values (memory/finger_width, memory/finger_effort, memory/tip_xyz or memory/tip_z)."""
    steps = ctrl.get("proprio_steps")
    if steps is not None and np.size(steps):
        return np.asarray(steps, float).reshape(-1, 5)
    if "finger_width" not in ctrl:
        return np.zeros((0, 5))
    width = float(ctrl["finger_width"])
    effort = float(ctrl.get("finger_effort", np.nan))
    return np.array([[width, effort, *_current_tip(ctrl)]])


def _current_tip(ctrl: dict) -> np.ndarray:
    """The fingertip position among the current values: tip_xyz, or NaN, NaN, tip_z (NaN when neither is sent)."""
    tip = ctrl.get("tip_xyz")
    if tip is not None:
        return np.asarray(tip, float).reshape(3)
    return np.array([np.nan, np.nan, float(ctrl.get("tip_z", np.nan))])


def signal_columns(signals, names: Sequence[str]) -> dict[str, np.ndarray]:
    """The task's declared signals of one policy call as columns {name: (n,) float}: memory/signal_steps ({name: values per
    control step since the previous call}) or the same mapping from a recorded episode. A name it does not carry is NaN."""
    given = {}
    for name, values in (signals or {}).items():
        if name in names:
            given[name] = np.asarray(values, float).reshape(-1)
    if not given:
        return {}
    length = max(len(values) for values in given.values())
    if any(len(values) != length for values in given.values()):
        lengths = {name: len(values) for name, values in given.items()}
        raise ValueError(f"memory/signal_steps: the signals have different lengths {lengths}")
    return {name: given.get(name, np.full(length, np.nan)) for name in names}
