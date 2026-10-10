"""What the rules author may answer, shared by its prompt, its answer check and its task-file writer: the prompt variants,
the answer's parameters with their units and bounds, the gripper thresholds, and the check that the rules writer accepts a
rule set. author.py re-exports these names.
"""
from __future__ import annotations

import copy

from ..task import TaskSpec
from .rules import RuleMonitor, parse_rules

VARIANTS = ("easy", "hard", "none")

# The answer's parameters: rule-engine name -> (answer field, factor to the task file's metres, minimum, maximum, optional).
# The bounds are physical sanity limits, wider than any plausible answer: a server with structured output enforces them while
# it generates the digits, which truncates a number instead of clamping it (asked for 50 below a maximum of 30, Qwen on vLLM
# wrote 5). A value at a bound is therefore suspicious; the validator, not the schema, is the safety check.
PARAMS: dict[str, dict[str, tuple[str, float, float, float, bool]]] = {
    "lift": {"rise": ("rise_cm", 0.01, 1.0, 40.0, False),
             "at_top": ("at_top_cm_per_step", 0.01, 0.01, 2.0, True)},
    "set_down": {"release_below": ("release_below_cm", 0.01, 1.0, 40.0, False),
                 "up": ("up_cm", 0.01, 1.0, 40.0, False),
                 "away": ("away_cm", 0.01, 1.0, 40.0, True)},
}
REASON_MAX = 600
GRIPPER_DEFAULTS = {"open_width": 0.06, "empty_width": 0.015, "squeeze_force": 8.0}   # as RuleMonitor


def _cm(x_m: float) -> str:
    """Metres as a short centimetre text: 0.06 -> "6", 0.015 -> "1.5"."""
    return f"{100 * x_m:.1f}".rstrip("0").rstrip(".")


def gripper_thresholds(task: TaskSpec, **overrides) -> dict:
    """open_width, empty_width, squeeze_force: the task file's proprio section over RuleMonitor's defaults."""
    proprio = task.proprio or {}
    thresholds = dict(GRIPPER_DEFAULTS)
    for name in GRIPPER_DEFAULTS:
        if name in proprio:
            thresholds[name] = float(proprio[name])
    for name, value in overrides.items():
        if value is not None:
            thresholds[name] = float(value)
    return thresholds


def with_rules(task: TaskSpec, rules: dict) -> TaskSpec:
    """A copy of the task with history.rules replaced."""
    copied = copy.deepcopy(task)
    copied.history = {**(copied.history or {}), "rules": copy.deepcopy(rules)}
    return copied


def check_rules(task: TaskSpec, rules: dict) -> None:
    """Raises ValueError if the rules writer would reject the rule set (parse_rules, RuleMonitor)."""
    candidate = with_rules(task, rules)
    parse_rules(candidate)
    RuleMonitor(candidate)
