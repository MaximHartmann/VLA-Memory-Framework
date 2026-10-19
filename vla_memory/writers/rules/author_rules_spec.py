"""What the rules author may answer, shared by its prompt, its answer check and its task-file writer: the prompt variants,
the answer's parameters with their units and bounds, and the check that the rules writer accepts a rule set. author.py
re-exports these names.
"""
from __future__ import annotations

import copy

from ...task import TaskSpec
from .rules import GRIPPER_DEFAULTS, RuleMonitor, gripper_thresholds, parse_rules

VARIANTS = ("easy", "hard", "none")

# The answer's parameters: rule-engine name -> (answer field, factor to the task file's metres, minimum, maximum, optional).
# The bounds are physical sanity limits, wider than any plausible answer: a server with structured output enforces them while
# it generates the digits, which truncates a number instead of clamping it (asked for 50 below a maximum of 30, Qwen on vLLM
# wrote 5). A value at a bound is therefore suspicious; the validator, not the schema, is the safety check.
PARAMETERS: dict[str, dict[str, tuple[str, float, float, float, bool]]] = {
    "lift": {"rise": ("rise_cm", 0.01, 1.0, 40.0, False),
             "at_top": ("at_top_cm_per_step", 0.01, 0.01, 2.0, True)},
    "set_down": {"release_below": ("release_below_cm", 0.01, 1.0, 40.0, False),
                 "up": ("up_cm", 0.01, 1.0, 40.0, False),
                 "away": ("away_cm", 0.01, 1.0, 40.0, True)},
}
REASON_MAX = 600

__all__ = ["VARIANTS", "PARAMETERS", "REASON_MAX", "GRIPPER_DEFAULTS", "gripper_thresholds", "centimetre_text", "with_rules",
           "check_rules"]


def centimetre_text(metres: float) -> str:
    """Metres as a short centimetre text: 0.06 -> "6", 0.015 -> "1.5"."""
    return f"{100 * metres:.1f}".rstrip("0").rstrip(".")


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
