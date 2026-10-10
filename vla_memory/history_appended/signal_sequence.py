"""Signal thresholds: the trigger `signals`, for any skill that is not a pick or a place (push, open or close a drawer,
press a button).

The task file lists phases, each a set of conditions on the signals that must all hold at one control step; the phases are
entered in order and the skill is complete when the last one is entered. A phase's `hold` conditions must keep holding
until the next phase is entered, else the sequence starts over (the grip on a handle lost before the pull). One phase per
control step. Only the current skill's verb is followed, and its sequence starts afresh whenever the pointer moves.
    open:
      trigger: signals
      phases:
        - when: {finger_effort: {above: 8.0}, finger_width: {above: 0.01, below: 0.06}}   # the fingers squeeze the handle
          hold: {finger_effort: {above: 8.0}}
        - when: {tip_x: {below: 0.47}}                    # pulled out: an absolute position (the environment's frame)
        - when: {finger_width: {above: 0.06}}             # let go
        - when: {tip: {away: 0.03}}                       # 3 cm from where it let go (relative to the previous phase)
Signals: finger_width, finger_effort, tip_x, tip_y, tip_z (the columns of memory/proprio_steps), the point tip (the
fingertip position, for away / near), and the task's declared signals. Conditions: above / below a value; rise / fall by
at least d since a reference phase was entered; away (distance at least d) / near (less than d) from the position there.
The reference is the previous phase (for a hold, its own phase); `from: k` names another earlier phase (1-based). A signal
the environment does not send is NaN, and a condition on NaN never holds.

parse_signal_phases() checks the phases of one rule; SignalSequence follows them step by step.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..env_keys import POINTS, PROPRIO_COLUMNS

ABSOLUTE_OPS = ("above", "below")                 # the value against a threshold
RELATIVE_OPS = ("rise", "fall", "away", "near")   # the change since a reference phase was entered
_POINT_OPS = ("away", "near")                     # a position (tip) only takes the distance conditions


def rule_signals(task) -> set:
    """The signal names a `trigger: signals` rule may use: the columns of memory/proprio_steps, the point tip, and the task's
    declared signals."""
    return set(PROPRIO_COLUMNS) | set(POINTS) | set(getattr(task, "signals", None) or ())


# ---------------------------------------------------------------------------------------------------------------- parsing
def parse_signal_phases(spec: dict, where: str, names: set) -> dict:
    """The parameters of a `trigger: signals` rule -> {"phases": [{"when": {...}, "hold": {...}}, ...]}, checked."""
    unknown = [key for key in spec if key != "phases"]
    phases = spec.get("phases")
    if unknown or not isinstance(phases, list) or not phases:
        raise ValueError(f"{where}: trigger signals takes `phases`, a non-empty list (unknown: {unknown})")
    out = []
    for number, phase in enumerate(phases, 1):
        out.append(_parse_phase(phase, f"{where}.phases[{number}]", names, number))
    return {"phases": out}


def _parse_phase(phase, where: str, names: set, number: int) -> dict:
    """One phase (1-based number): `when` measures from the previous phase, `hold` from the phase itself."""
    if not isinstance(phase, dict) or "when" not in phase or any(key not in ("when", "hold") for key in phase):
        raise ValueError(f"{where}: a mapping with `when` and optionally `hold` expected, got {phase!r}")
    previous = number - 1 if number > 1 else None   # the first phase has no earlier phase to measure from
    out = {"when": parse_conditions(phase["when"], f"{where}.when", names, previous, number - 1)}
    if phase.get("hold") is not None:
        out["hold"] = parse_conditions(phase["hold"], f"{where}.hold", names, number, number)
    return out


def parse_conditions(spec, where: str, names: set, ref_default: int | None, ref_max: int) -> dict:
    """One `when` or `hold` mapping {signal: {op: value, [from: k]}} -> the normalised mapping (floats; `from` as given).
    ref_default: the phase a relative condition measures from unless `from` says otherwise (None: there is none);
    ref_max: the highest phase number `from` may name."""
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"{where}: a non-empty mapping {{signal: {{condition: value}}}} expected, got {spec!r}")
    out = {}
    for signal, operations in spec.items():
        _check_signal_name(signal, names, where)
        out[signal] = _parse_signal_conditions(signal, operations, where, ref_default, ref_max)
    return out


def _check_signal_name(signal, names: set, where: str) -> None:
    """The signal must be one the rule may use (rule_signals)."""
    if signal not in names:
        raise ValueError(f"{where}: unknown signal {signal!r} (signals: {', '.join(sorted(names))}; declare a robot signal "
                         f"of your own under `signals` in the task file)")


def _parse_signal_conditions(signal, operations, where: str, ref_default: int | None, ref_max: int) -> dict:
    """The conditions of one signal -> {op: float, ..., [from: k]}."""
    if not isinstance(operations, dict) or not operations:
        raise ValueError(f"{where}.{signal}: a mapping of conditions expected, e.g. {{above: 5.0}}, got {operations!r}")
    operations = dict(operations)
    reference = operations.pop("from", None)
    _check_operations(signal, operations, where)
    _check_reference(signal, operations, reference, where, ref_default, ref_max)
    out = {}
    for op, value in operations.items():
        out[op] = float(value)
    if reference is not None:
        out["from"] = reference
    return out


def _check_operations(signal, operations: dict, where: str) -> None:
    """Every condition must be one the signal takes, with a finite number as its value."""
    allowed = _POINT_OPS if signal in POINTS else ABSOLUTE_OPS + RELATIVE_OPS
    bad = [op for op in operations if op not in allowed]
    if bad or not operations:
        raise ValueError(f"{where}.{signal}: condition(s) {bad or 'none'}; {signal} takes {list(allowed)}")
    for op, value in operations.items():
        if not _is_number(value):
            raise ValueError(f"{where}.{signal}.{op}: a number expected, got {value!r}")


def _is_number(value) -> bool:
    """A finite int or float; a bool does not count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _check_reference(signal, operations: dict, reference, where: str, ref_default: int | None, ref_max: int) -> None:
    """`from: k` is only for relative conditions and must name an earlier phase; without it, the first phase has none."""
    relative = [op for op in operations if op in RELATIVE_OPS]
    if reference is not None:
        if not relative:
            raise ValueError(f"{where}.{signal}: `from` applies to the relative conditions {list(RELATIVE_OPS)} only")
        if isinstance(reference, bool) or not isinstance(reference, int) or not 1 <= reference <= ref_max:
            raise ValueError(f"{where}.{signal}: from must be a phase number from 1 to {ref_max}, got {reference!r}")
        return
    if relative and ref_default is None:
        raise ValueError(f"{where}.{signal}: {relative} measure from an earlier phase, and the first phase has none")


# ---------------------------------------------------------------------------------------------------------------- following
class SignalSequence:
    """The per-step state machine of one `trigger: signals` rule (module docstring); step() returns True when the last phase
    is entered, and the sequence starts over."""

    def __init__(self, phases: Sequence[dict]):
        # Each phase: its compiled `when` conditions (reference: the previous phase) and `hold` conditions (reference: the
        # phase itself). Phase indices are 0-based here.
        self.phases = []
        for index, phase in enumerate(phases):
            when = self._compile(phase["when"], index - 1)
            hold = self._compile(phase.get("hold") or {}, index)
            self.phases.append((when, hold))
        self.signals = _signals_used(phases)
        self.reset()

    @staticmethod
    def _compile(conditions: dict, ref_default: int) -> list:
        """The conditions as (signal, op, value, reference phase index) tuples; `from: k` (1-based) overrides the default."""
        out = []
        for signal, operations in conditions.items():
            reference = ref_default
            if "from" in operations:
                reference = operations["from"] - 1
            for op, value in operations.items():
                if op != "from":
                    out.append((signal, op, value, reference))
        return out

    def reset(self):
        self.k = 0                      # phases entered so far
        self.anchors: list[dict] = []   # the signals at the step each phase was entered

    @property
    def progress(self) -> str:
        return f"{self.k}/{len(self.phases)}"

    def step(self, now: dict) -> bool:
        """One control step; now: every signal's value at this step (signal -> float; tip -> (3,))."""
        if self.k and not self._holds(self.phases[self.k - 1][1], now):
            self.reset()   # a hold condition broke: start over (from the next step)
            return False
        if not self._holds(self.phases[self.k][0], now):
            return False
        self.anchors.append(now)
        self.k += 1
        if self.k < len(self.phases):
            return False
        self.reset()
        return True

    def _holds(self, conditions: list, now: dict) -> bool:
        """Whether every condition holds at this step (a NaN reading never holds)."""
        for signal, op, value, reference in conditions:
            if not self._condition_holds(signal, op, value, reference, now):
                return False
        return True

    def _condition_holds(self, signal, op, value, reference, now: dict) -> bool:
        current = now[signal]
        if op in ABSOLUTE_OPS:
            return _absolute_holds(op, current, value)
        anchor = self.anchors[reference][signal]
        if op in ("rise", "fall"):
            return _change_holds(op, current, anchor, value)
        return _distance_holds(op, current, anchor, value)


def _signals_used(phases) -> set:
    """Every signal name the phases' conditions mention."""
    used = set()
    for phase in phases:
        for part in ("when", "hold"):
            used.update(phase.get(part) or {})
    return used


def _absolute_holds(op: str, current, threshold: float) -> bool:
    """above / below: the value against the threshold."""
    if op == "above":
        return current > threshold
    return current < threshold


def _change_holds(op: str, current, anchor, amount: float) -> bool:
    """rise / fall: the change since the reference phase was entered."""
    if op == "rise":
        return current - anchor >= amount
    return anchor - current >= amount


def _distance_holds(op: str, current, anchor, distance: float) -> bool:
    """away / near: the distance from the position at the reference phase (|x - a| for a scalar)."""
    moved = float(np.linalg.norm(np.subtract(current, anchor)))
    if op == "away":
        return moved >= distance
    return moved < distance
