"""The task file's rules for the rules writer: `history.rules` maps every verb of the task to one trigger with its
thresholds. parse_rules() checks the section and returns {verb: (trigger, params)}.

    pick up: {trigger: lift, rise: 0.09}                                      the grasp cycle's triggers: grasp_cycle.py
    place:   {trigger: set_down, release_below: 0.08, up: 0.06, away: 0.08}
    open:    {trigger: signals, phases: [...]}                                 phases of signal conditions: signal_sequence.py
"""
from __future__ import annotations

from .signal_sequence import parse_signal_phases, rule_signals

TRIGGERS = ("lift", "set_down")             # the grasp cycle's triggers
SIGNALS = "signals"                         # the trigger of signal thresholds (task-declared phases)
ALL_TRIGGERS = TRIGGERS + (SIGNALS,)
_PARAMS = {"lift": ("rise", "at_top"), "set_down": ("release_below", "up", "away")}   # the thresholds of a grasp trigger
_OPTIONAL = ("away", "at_top")


# ---------------------------------------------------------------------------------------------------------------- task file
def parse_rules(task) -> dict[str, tuple[str, dict]]:
    """history.rules of the task file -> {verb: (trigger, params)}; checks triggers, parameters and verbs. params: the
    thresholds of a grasp trigger (lift, set_down), or {"phases": [...]} of a signals trigger."""
    rules = dict((task.history or {}).get("rules") or {})
    if not rules:
        raise ValueError("the task file has no history.rules section; the rules writer needs one")
    out = {}
    for verb, spec in rules.items():
        if verb not in task.verbs:
            raise ValueError(f"history.rules: unknown verb {verb!r} (task verbs: {task.verbs})")
        out[verb] = _parse_rule(task, verb, spec)
    uncovered = [verb for verb in task.verbs if verb not in out]
    if uncovered:
        raise ValueError(f"history.rules: no rule for verb(s) {uncovered}; every skill of a plan needs a trigger")
    return out


def _parse_rule(task, verb: str, spec) -> tuple[str, dict]:
    """One verb's rule -> (trigger, params): the phases of a signals trigger, the thresholds of a grasp trigger."""
    spec = dict(spec)
    trigger = spec.pop("trigger", None)
    where = f"history.rules[{verb!r}]"
    if trigger not in ALL_TRIGGERS:
        raise ValueError(f"{where}: trigger must be one of {ALL_TRIGGERS}, got {trigger!r}")
    if trigger == SIGNALS:
        return trigger, parse_signal_phases(spec, where, rule_signals(task))
    return trigger, _parse_grasp_params(where, trigger, spec)


def _parse_grasp_params(where: str, trigger: str, spec: dict) -> dict:
    """The thresholds of a lift or set_down rule as floats (an optional one may be None)."""
    missing = [name for name in _PARAMS[trigger] if name not in spec and name not in _OPTIONAL]
    unknown = [name for name in spec if name not in _PARAMS[trigger]]
    if missing or unknown:
        raise ValueError(f"{where}: missing {missing}, unknown {unknown} (trigger {trigger} takes {_PARAMS[trigger]})")
    params = {}
    for name, value in spec.items():
        params[name] = None if value is None else float(value)
    return params
