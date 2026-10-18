"""What one policy call provides for the retrieval: the raw wrist frame, the finger width and the skill.

    wrist frame    memory/wrist_raw, the raw frame the store's keys were computed on; without it the policy's own wrist image
                   (the proxy's second --image-keys entry), which the extension reports once
    finger width   m, both fingers together: the last row of memory/proprio_steps (column 0, the width of the call's own
                   step), or memory/finger_width at the first call of an episode, when proprio_steps is still empty
    skill          the prompt the proxy decided, when it is one of the task's skills (rules, oracle and vlm mode); else
                   the monitor's current skill (a method's proxy that renders a longer prompt)

The proxy hands the extension the control keys without their memory/ prefix (ctrl).
"""
from __future__ import annotations

import math

import numpy as np

from ...env_keys import PROPRIO_COLUMNS

WIDTH_COLUMN = PROPRIO_COLUMNS.index("finger_width")


def wrist_frame(obs: dict, ctrl: dict, fallback_key: str | None) -> tuple:
    """(frame or None, whether the fallback was used): memory/wrist_raw, else the policy's image under fallback_key."""
    frame = ctrl.get("wrist_raw")
    if frame is not None:
        return np.asarray(frame), False
    if fallback_key is not None and obs.get(fallback_key) is not None:
        return np.asarray(obs[fallback_key]), True
    return None, False


def finger_width(ctrl: dict) -> float | None:
    """The measured finger width (m) at this call; None when it is not sent or NaN."""
    steps = ctrl.get("proprio_steps")
    if steps is not None and np.size(steps):
        rows = np.asarray(steps, float).reshape(-1, len(PROPRIO_COLUMNS))
        width = float(rows[-1, WIDTH_COLUMN])
    elif ctrl.get("finger_width") is not None:
        width = float(ctrl["finger_width"])
    else:
        return None
    if math.isnan(width):
        return None
    return width


def skill_of(prompt: str, planner, skills) -> str | None:
    """The skill the policy works on: the prompt if it is one of the skills, else the monitor's current skill, else None."""
    if prompt in skills:
        return prompt
    current = getattr(planner, "current_skill", None)
    if current in skills:
        return current
    return None
