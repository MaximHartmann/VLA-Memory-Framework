"""The rules writer: a monitor without a model. RuleMonitor decides from the robot's own gripper and hand signals when the
current skill is done, by the rules of the task file (rule_spec.py), and advances a pointer through the plan exactly as the
planner does. It is the drop-in writer of the history-appended prompt and the sequencer of the planner loop's `rules` mode.

Signals, one row per control step: finger width (m, both fingers together), squeeze (finger force in N in the simulator, or
1/0 from a gripper that reports a grasped flag), fingertip position x, y, z (m, forward kinematics of the measured joints).
A task may declare further signals (`signals` in the task file), sent as memory/signal_steps and read by signals.py.

The thresholds must reproduce the moments at which the training labels change: a writer that is right about the event but
switches earlier than the labels loses success (docs/methods/history_appended/README.md). The same rules label recorded
episodes (label_rows), so training labels and run-time history come from one definition.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..env_keys import PROPRIO_COLUMNS
from .grasp_cycle import GraspCycle
from .rule_spec import SIGNALS, parse_rules
from .signal_sequence import SignalSequence
from .signals import _NAN_ROW, signal_columns


def _column_length(columns: dict) -> int:
    """How many control steps the signal columns carry (0 without columns)."""
    if not columns:
        return 0
    return len(next(iter(columns.values())))


def _step_count(rows, columns: dict) -> int:
    """How many control steps a call carries; rows and signals must agree when both are given."""
    signal_steps = _column_length(columns)
    if len(rows) and signal_steps and len(rows) != signal_steps:
        raise ValueError(f"{len(rows)} control steps in memory/proprio_steps but {signal_steps} in memory/signal_steps: every step "
                         f"must appear in both")
    return max(len(rows), signal_steps)


# ---------------------------------------------------------------------------------------------------------------- the writer
class RuleMonitor:
    """Drop-in writer for the history-appended prompt with the planner's pointer attributes (plan, idx, done_all,
    current_skill). The plan is the task's skill order unless one is given. Gripper thresholds come from the task's
    proprio section (open_width, empty_width, squeeze_force); keyword arguments override them (e.g. squeeze_force=0.5
    for a gripper that reports a grasped flag).

    With an identifier (identify.py), a pick names the object the wrist camera shows in the gripper instead of assuming
    the plan's: if another object of the task is held, its pending skills move to the front of the remaining plan and the
    line names it; if the held object has no pending pick, nothing is written; if the identifier cannot tell, the plan
    order holds. A set-down writes "placed" only for the object identified at the last lift; a different one (a block
    that was dropped, then another one carried) writes nothing, because a false line misleads the policy for the rest of
    the episode.

    Verbs with `trigger: signals` follow their phases (SignalSequence) while their skill is current, in the plan's order (no
    identifier); signal_names are the task's declared signals the rules use, which observe_rows() takes as `signals`. A task
    without grasp verbs keeps an idle grasp cycle (never stepped, always "free") for code that reads monitor.cycle, and its
    writer_state is the current skill's progress, "<verb> <phases entered>/<phases>"."""

    def __init__(self, task, plan: Sequence[str] | None = None, identifier=None, **gripper_overrides):
        self.task = task
        self.rules = parse_rules(task)
        self.identifier = identifier
        gripper = self._gripper_thresholds(task, gripper_overrides)
        lift = self._rule_params("lift")
        down = self._rule_params("set_down")
        self.grasp = lift is not None or down is not None   # the grasp cycle runs for a task with grasp verbs only
        if self.grasp and (lift is None or down is None):
            raise ValueError("history.rules needs one verb with trigger lift and one with trigger set_down")
        self.cycle = self._build_grasp_cycle(gripper, lift, down)
        self.sequences = self._build_sequences()
        self.signal_names = self._used_signal_names(task)
        self.default_plan = list(plan) if plan else list(task.skills)
        for skill in self.default_plan:
            task.parse_skill(skill)   # raises for strings outside the vocabulary
        self.reset()

    @staticmethod
    def _gripper_thresholds(task, overrides: dict) -> dict:
        """open_width, empty_width, squeeze_force: the defaults, then the task's proprio section, then the overrides."""
        thresholds = {"open_width": 0.06, "empty_width": 0.015, "squeeze_force": 8.0}
        proprio = task.proprio or {}
        for name in ("open_width", "empty_width", "squeeze_force"):
            if name in proprio:
                thresholds[name] = proprio[name]
        thresholds.update(overrides)
        return thresholds

    def _rule_params(self, trigger: str) -> dict | None:
        """The parameters of the first rule with this trigger, or None if no verb uses it."""
        for rule_trigger, params in self.rules.values():
            if rule_trigger == trigger:
                return params
        return None

    def _build_grasp_cycle(self, gripper: dict, lift: dict | None, down: dict | None) -> GraspCycle:
        """The grasp cycle with the lift and set_down thresholds; a task without grasp verbs gets an idle one that never
        fires."""
        if not self.grasp:
            return GraspCycle(gripper["open_width"], gripper["empty_width"], gripper["squeeze_force"], math.inf, 0.0, math.inf, None)
        return GraspCycle(gripper["open_width"], gripper["empty_width"], gripper["squeeze_force"], lift["rise"],
                          down["release_below"], down["up"], down.get("away"), lift.get("at_top"))

    def _build_sequences(self) -> dict[str, SignalSequence]:
        """One SignalSequence per verb with trigger signals."""
        sequences = {}
        for verb, (trigger, params) in self.rules.items():
            if trigger == SIGNALS:
                sequences[verb] = SignalSequence(params["phases"])
        return sequences

    def _used_signal_names(self, task) -> tuple:
        """The task's declared signals that the sequences read, in the order the task declares them."""
        used = set()
        for sequence in self.sequences.values():
            used |= sequence.signals
        declared = getattr(task, "signals", None) or ()
        return tuple(name for name in declared if name in used)

    def reset(self):
        self.plan = list(self.default_plan)
        self.idx = 0
        self.done_all = False
        self.events: list[tuple] = []
        self.held_obj = None   # the object the identifier saw at the last lift (None: unknown or no identifier)
        self.cycle.reset()
        for sequence in self.sequences.values():
            sequence.reset()

    @property
    def current_skill(self) -> str:
        return self.plan[min(self.idx, len(self.plan) - 1)]

    @property
    def writer_state(self) -> str:
        """The grasp cycle's state, or, for a task without grasp verbs, the current skill's progress through its phases."""
        if self.grasp or not self.sequences:
            return self.cycle.state
        verb = self.task.parse_skill(self.current_skill)[0]
        if verb in self.sequences:
            return f"{verb} {self.sequences[verb].progress}"
        return self.cycle.state

    def start_episode(self, instruction: str = "", episode_id: str = "") -> dict:
        self.reset()
        return {"plan": list(self.plan), "source": "task order (rules writer)", "ms": 0.0}

    def _bring_forward(self, held: str) -> bool:
        """Another object than planned was picked up: move its pending skills to the front of the remaining plan."""
        rest = self.plan[self.idx:]
        mine = [skill for skill in rest if self.task.parse_skill(skill)[1] == held]
        if not mine or self.rules[self.task.parse_skill(mine[0])[0]][0] != "lift":
            return False   # no pending pick of this object
        others = [skill for skill in rest if self.task.parse_skill(skill)[1] != held]
        self.plan = self.plan[:self.idx] + mine + others
        return True

    def _write(self, t, done: list):
        """The current skill is complete at control step t: a line, and the pointer moves on (a new skill's sequence starts
        afresh)."""
        done.append(self.plan[self.idx])
        self.events.append((t, self.plan[self.idx]))
        if self.idx + 1 >= len(self.plan):
            self.done_all = True
            self.idx = len(self.plan) - 1
        else:
            self.idx += 1
        for sequence in self.sequences.values():
            sequence.reset()

    def observe_rows(self, rows, step: int | None = None, image=None, signals=None) -> dict:
        """Feed the control steps since the last call (and the current wrist image, for the identifier); returns the
        decision record. signals: the task's declared signals over the same steps ({name: (n,)}, memory/signal_steps); when
        both are given, rows and signals must have one entry per step each."""
        rows = np.asarray(rows, float).reshape(-1, 5)
        extra = self._declared_columns(signals)
        count = _step_count(rows, extra)
        done: list[str] = []
        identified = None
        for i in range(count):
            row = rows[i] if len(rows) else _NAN_ROW
            t = None if step is None else int(step) - count + i
            if self.grasp:
                written, identification = self._step_grasp_cycle(row, t, image, done)
                if identification is not None:
                    identified = identification
                if written:
                    continue   # one line per control step
            self._step_signal_sequence(row, extra, i, t, done)
        return {"advanced": bool(done), "completed": done, "skill_after": self.current_skill, "writer_state": self.writer_state,
                "plan_complete": self.done_all, "identified": identified}

    def _declared_columns(self, signals) -> dict:
        """The declared signals the rules read, as columns per step; {} when the rules read none."""
        if not self.signal_names:
            return {}
        return signal_columns(signals, self.signal_names)

    def _step_grasp_cycle(self, row, t, image, done: list) -> tuple[bool, dict | None]:
        """One control step of the grasp cycle: a trigger that completes the current skill writes its line.
        Returns (line written, the identification made at a lift or None)."""
        trigger = self.cycle.step(row[0], row[1], row[2:5])
        if trigger is None or self.done_all:
            return False, None
        completes, identification = self._grasp_line(trigger, image)
        if completes:
            self._write(t, done)
        return completes, identification

    def _step_signal_sequence(self, row, extra: dict, i: int, t, done: list) -> None:
        """One control step of the current skill's signal sequence, if its verb has one: the last phase writes the line."""
        if not self.sequences or self.done_all:
            return
        verb = self.task.parse_skill(self.plan[self.idx])[0]
        sequence = self.sequences.get(verb)
        if sequence is None:
            return
        if sequence.step(self._values(row, extra, i)):
            self._write(t, done)

    def _grasp_line(self, trigger: str, image) -> tuple[bool, dict | None]:
        """Whether a grasp trigger completes the current skill (identifier, reordering and set-down guard of the class
        docstring), with the identification made at a lift (None without one)."""
        identification = None
        if trigger == "lift" and self.identifier is not None and image is not None:
            identification, in_plan = self._identify_at_lift(image)
            if not in_plan:
                return False, identification
        verb, obj = self.task.parse_skill(self.plan[self.idx])
        if self.rules[verb][0] != trigger:
            return False, identification   # e.g. a second lift of a block that slipped out: the pick is already written
        if trigger == "set_down":
            if self.held_obj is not None and self.held_obj != obj:
                return False, identification   # another object than this place step's was set down: write nothing
            self.held_obj = None
        return True, identification

    def _identify_at_lift(self, image) -> tuple[dict, bool]:
        """Ask the identifier which object is held; another than planned brings its skills forward. Returns the
        identification record and whether the lift is a step of the plan at all."""
        verb, expected = self.task.parse_skill(self.plan[self.idx])
        held = self.identifier.identify(image)
        identification = {"held": held, "expected": expected, "reordered": False}
        self.held_obj = held
        if held is None or held == expected:
            return identification, True
        if self.rules[verb][0] != "lift" or not self._bring_forward(held):
            return identification, False   # the held object has no pending pick: not a step of the plan, write nothing
        identification["reordered"] = True
        return identification, True

    def _values(self, row, extra: dict, i: int) -> dict:
        """Every signal's value at one step: the five columns, the point tip and the declared signals (NaN if not sent)."""
        now = {}
        for name, value in zip(PROPRIO_COLUMNS, row):
            now[name] = float(value)
        now["tip"] = np.array(row[2:5], float)
        for name in self.signal_names:
            if name in extra:
                now[name] = float(extra[name][i])
            else:
                now[name] = math.nan
        return now

    def completed(self) -> list[str]:
        """The skills completed so far, in order (what the history lists)."""
        if self.done_all:
            return list(self.plan)
        return list(self.plan[:self.idx])

    def label_rows(self, rows, signals=None) -> np.ndarray:
        """Training labels of a recorded episode: the number of completed skills after each control step (signals: the
        task's declared signals per step, {name: (T,)})."""
        self.reset()
        rows = np.asarray(rows, float).reshape(-1, 5)
        extra = self._declared_columns(signals)
        count = max(len(rows), _column_length(extra))
        labels = np.zeros(count, dtype=np.int16)
        for i in range(count):
            step_signals = {name: values[i:i + 1] for name, values in extra.items()} or None
            self.observe_rows(rows[i:i + 1], signals=step_signals)
            labels[i] = len(self.completed())
        return labels
