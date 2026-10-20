"""The rules writer: a monitor without a model. RuleMonitor decides from the robot's own gripper and hand signals when the
current skill is done, by the rules of the task file (rule_spec.py), and advances a pointer through the plan exactly as the
planner does. It is the drop-in writer of the history-appended prompt and the sequencer of the planner loop's `rules` mode.

Signals, one row per control step: finger width (m, both fingers together), squeeze (finger force in N in the simulator, or
1/0 from a gripper that reports a grasped flag), fingertip position x, y, z (m, forward kinematics of the measured joints).
A task may declare further signals (`signals` in the task file), sent as memory/signal_steps and read by signals.py.

The thresholds must reproduce the moments at which the training labels change: a writer that is right about the event but
switches earlier than the labels loses success (docs/methods/history/README.md). The same rules label recorded
episodes (label_rows), so training labels and run-time history come from one definition.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...env_keys import PROPRIO_COLUMNS
from .grasp_cycle import GraspCycle
from .rule_spec import SIGNALS, TRIGGERS, gripper_thresholds, parse_rules
from .signal_sequence import SignalSequence
from .signals import NAN_ROW, column_length, signal_columns, step_count


class RuleMonitor:
    """Drop-in writer for the history-appended prompt with the planner's pointer attributes (plan, skill_index, done_all,
    current_skill). The plan is the task's skill order unless one is given; an episode that declares its objects
    (start_episode) plans the skills of those objects only. Gripper thresholds come from the task's proprio section
    (open_width, empty_width, squeeze_force); keyword arguments override them (e.g. squeeze_force=0.5 for a gripper that
    reports a grasped flag).

    With an identifier (identify.py), a pick names the object the wrist camera shows in the gripper instead of assuming
    the plan's: if another object of the episode is held, its pending skills of the current stage (plan_stages; one stage
    covers every verb) move to the front of the remaining plan and the line names it; if the held object has no pending
    pick, nothing is written; if the identifier cannot tell, the plan order holds. A set-down writes "placed" only for the
    object identified at the last lift; a different one (a block that was dropped, then another one carried) writes
    nothing, because a false line misleads the policy for the rest of the episode.

    The thresholds of a grasp trigger are those of the current skill's verb: verbs with the same trigger may have their own
    (`release_below` of putting an object into a container and of stacking it), and the grasp cycle uses the first rule of a
    trigger while the current skill's verb has the other trigger.

    Verbs with `trigger: signals` follow their phases (SignalSequence) while their skill is current, in the plan's order (no
    identifier); signal_names are the task's declared signals the rules use, which observe_rows() takes as `signals`. A task
    without grasp verbs keeps an idle grasp cycle (never stepped, always "free") for code that reads monitor.cycle, and its
    writer_state is the current skill's progress, "<verb> <phases entered>/<phases>"."""

    def __init__(self, task, plan: Sequence[str] | None = None, identifier=None, **gripper_overrides):
        self.task = task
        self.rules = parse_rules(task)
        self.identifier = identifier
        gripper = gripper_thresholds(task, **gripper_overrides)
        lift = self._rule_parameters("lift")
        set_down = self._rule_parameters("set_down")
        self.has_grasp_verbs = lift is not None or set_down is not None   # the grasp cycle runs for grasp verbs only
        if self.has_grasp_verbs and (lift is None or set_down is None):
            raise ValueError("history.rules needs one verb with trigger lift and one with trigger set_down")
        self.cycle = self._build_grasp_cycle(gripper, lift, set_down)
        self._thresholds_skill = None     # the skill whose verb the grasp cycle's thresholds were last taken from
        self.sequences = self._build_sequences()
        self.signal_names = self._used_signal_names(task)
        self.default_plan = list(plan) if plan else list(task.skills)
        for skill in self.default_plan:
            task.parse_skill(skill)   # raises for strings outside the vocabulary
        self.reset()

    def _rule_parameters(self, trigger: str) -> dict | None:
        """The parameters of the first rule with this trigger, or None if no verb uses it."""
        for rule_trigger, parameters in self.rules.values():
            if rule_trigger == trigger:
                return parameters
        return None

    def _build_grasp_cycle(self, gripper: dict, lift: dict | None, set_down: dict | None) -> GraspCycle:
        """The grasp cycle with the lift and set_down thresholds; a task without grasp verbs gets an idle one that never
        fires."""
        if not self.has_grasp_verbs:
            return GraspCycle(gripper["open_width"], gripper["empty_width"], gripper["squeeze_force"], math.inf, 0.0, math.inf, None)
        return GraspCycle(gripper["open_width"], gripper["empty_width"], gripper["squeeze_force"], lift["rise"],
                          set_down["release_below"], set_down["up"], set_down.get("away"), lift.get("at_top"))

    def _build_sequences(self) -> dict[str, SignalSequence]:
        """One SignalSequence per verb with trigger signals."""
        sequences = {}
        for verb, (trigger, parameters) in self.rules.items():
            if trigger == SIGNALS:
                sequences[verb] = SignalSequence(parameters["phases"])
        return sequences

    def _used_signal_names(self, task) -> tuple:
        """The task's declared signals that the sequences read, in the order the task declares them."""
        used = set()
        for sequence in self.sequences.values():
            used |= sequence.signals
        declared = getattr(task, "signals", None) or ()
        return tuple(name for name in declared if name in used)

    def reset(self, objects=None):
        """A fresh episode: the default plan (of the declared objects only, when objects is given), the pointer at its first
        skill, no events, the state machines at rest."""
        self.plan = self.task.restrict_plan(self.default_plan, objects)
        self.present = None if objects is None else set(self.task.episode_objects(objects))   # None: every object of the task
        self.skill_index = 0
        self.done_all = False
        self.events: list[tuple] = []
        self.held_object = None   # the object the identifier saw at the last lift (None: unknown or no identifier)
        self.cycle.reset()
        for sequence in self.sequences.values():
            sequence.reset()

    @property
    def current_skill(self) -> str:
        """The skill the pointer is on (the last one once the plan is complete)."""
        return self.plan[min(self.skill_index, len(self.plan) - 1)]

    @property
    def writer_state(self) -> str:
        """The grasp cycle's state, or, for a task without grasp verbs, the current skill's progress through its phases."""
        if self.has_grasp_verbs or not self.sequences:
            return self.cycle.state
        verb = self.task.parse_skill(self.current_skill)[0]
        if verb in self.sequences:
            return f"{verb} {self.sequences[verb].progress}"
        return self.cycle.state

    def start_episode(self, instruction: str = "", episode_id: str = "", objects=None) -> dict:
        """Reset for a new episode (objects: those the environment declares); the plan record names the task order as the
        plan's source."""
        self.reset(objects)
        return {"plan": list(self.plan), "source": "task order (rules writer)", "ms": 0.0}

    def _bring_forward(self, held: str) -> bool:
        """Another object than planned was picked up: move its pending skills of the current stage to the front of the
        remaining plan."""
        remaining = self.plan[self.skill_index:]
        stage = self.task.stage_of(self.task.parse_skill(remaining[0])[0])
        held_skills = [skill for skill in remaining if self._is_object_in_stage(skill, held, stage)]
        if not held_skills or self.rules[self.task.parse_skill(held_skills[0])[0]][0] != "lift":
            return False   # no pending pick of this object
        other_skills = [skill for skill in remaining if skill not in held_skills]
        self.plan = self.plan[:self.skill_index] + held_skills + other_skills
        return True

    def _is_object_in_stage(self, skill: str, object_name: str, stage: int) -> bool:
        """Whether the skill is one of the object's skills in this plan stage."""
        verb, skill_object = self.task.parse_skill(skill)
        return skill_object == object_name and self.task.stage_of(verb) == stage

    def _complete_current_skill(self, control_step, completed: list):
        """The current skill is complete at this control step: a line, and the pointer moves on (a new skill's sequence
        starts afresh)."""
        completed.append(self.plan[self.skill_index])
        self.events.append((control_step, self.plan[self.skill_index]))
        if self.skill_index + 1 >= len(self.plan):
            self.done_all = True
            self.skill_index = len(self.plan) - 1
        else:
            self.skill_index += 1
        for sequence in self.sequences.values():
            sequence.reset()

    def observe_rows(self, rows, step: int | None = None, image=None, signals=None) -> dict:
        """Feed the control steps since the last call (and the current wrist image, for the identifier); returns the
        decision record. signals: the task's declared signals over the same steps ({name: (n,)}, memory/signal_steps); when
        both are given, rows and signals must have one entry per step each."""
        rows = np.asarray(rows, float).reshape(-1, 5)
        declared = self._declared_columns(signals)
        count = step_count(rows, declared)
        completed: list[str] = []
        identified = None
        for offset in range(count):
            row = rows[offset] if len(rows) else NAN_ROW
            control_step = None if step is None else int(step) - count + offset
            if self.has_grasp_verbs:
                written, identification = self._step_grasp_cycle(row, control_step, image, completed)
                if identification is not None:
                    identified = identification
                if written:
                    continue   # one line per control step
            self._step_signal_sequence(row, declared, offset, control_step, completed)
        return {"advanced": bool(completed), "completed": completed, "skill_after": self.current_skill,
                "writer_state": self.writer_state, "plan_complete": self.done_all, "identified": identified}

    def _declared_columns(self, signals) -> dict:
        """The declared signals the rules read, as columns per step; {} when the rules read none."""
        if not self.signal_names:
            return {}
        return signal_columns(signals, self.signal_names)

    def _step_grasp_cycle(self, row, control_step, image, completed: list) -> tuple[bool, dict | None]:
        """One control step of the grasp cycle: a trigger that completes the current skill writes its line.
        Returns (line written, the identification made at a lift or None)."""
        self._use_thresholds_of_current_skill()
        trigger = self.cycle.step(row[0], row[1], row[2:5])
        if trigger is None or self.done_all:
            return False, None
        completes, identification = self._grasp_line(trigger, image)
        if completes:
            self._complete_current_skill(control_step, completed)
        return completes, identification

    def _use_thresholds_of_current_skill(self) -> None:
        """The grasp cycle works with the thresholds of the current skill's verb, when that verb has a grasp trigger."""
        if self.done_all:
            return
        skill = self.plan[self.skill_index]
        if skill == self._thresholds_skill:
            return
        self._thresholds_skill = skill
        trigger, parameters = self.rules[self.task.parse_skill(skill)[0]]
        if trigger in TRIGGERS:
            self.cycle.configure(trigger, parameters)

    def _step_signal_sequence(self, row, declared: dict, offset: int, control_step, completed: list) -> None:
        """One control step of the current skill's signal sequence, if its verb has one: the last phase writes the line."""
        if not self.sequences or self.done_all:
            return
        verb = self.task.parse_skill(self.plan[self.skill_index])[0]
        sequence = self.sequences.get(verb)
        if sequence is None:
            return
        if sequence.step(self._signal_values(row, declared, offset)):
            self._complete_current_skill(control_step, completed)

    def _grasp_line(self, trigger: str, image) -> tuple[bool, dict | None]:
        """Whether a grasp trigger completes the current skill (identifier, reordering and set-down guard of the class
        docstring), with the identification made at a lift (None without one)."""
        identification = None
        if trigger == "lift" and self.identifier is not None and image is not None:
            identification, in_plan = self._identify_at_lift(image)
            if not in_plan:
                return False, identification
        verb, object_name = self.task.parse_skill(self.plan[self.skill_index])
        if self.rules[verb][0] != trigger:
            return False, identification   # e.g. a second lift of a block that slipped out: the pick is already written
        if trigger == "set_down":
            if self.held_object is not None and self.held_object != object_name:
                return False, identification   # another object than this place step's was set down: write nothing
            self.held_object = None
        return True, identification

    def _identify_at_lift(self, image) -> tuple[dict, bool]:
        """Ask the identifier which object is held; another than planned brings its skills forward. Returns the
        identification record and whether the lift is a step of the plan at all."""
        verb, expected = self.task.parse_skill(self.plan[self.skill_index])
        held = self.identifier.identify(image)
        if self.present is not None and held not in self.present:
            held = None   # an object that is not in this episode: the identifier cannot tell
        identification = {"held": held, "expected": expected, "reordered": False}
        self.held_object = held
        if held is None or held == expected:
            return identification, True
        if self.rules[verb][0] != "lift" or not self._bring_forward(held):
            return identification, False   # the held object has no pending pick: not a step of the plan, write nothing
        identification["reordered"] = True
        return identification, True

    def _signal_values(self, row, declared: dict, offset: int) -> dict:
        """Every signal's value at one step: the five columns, the point tip and the declared signals (NaN if not sent)."""
        values = {}
        for name, value in zip(PROPRIO_COLUMNS, row):
            values[name] = float(value)
        values["tip"] = np.array(row[2:5], float)
        for name in self.signal_names:
            if name in declared:
                values[name] = float(declared[name][offset])
            else:
                values[name] = math.nan
        return values

    def completed(self) -> list[str]:
        """The skills completed so far, in order (what the history lists)."""
        if self.done_all:
            return list(self.plan)
        return list(self.plan[:self.skill_index])

    def label_rows(self, rows, signals=None, objects=None) -> np.ndarray:
        """Training labels of a recorded episode: the number of completed skills after each control step (signals: the
        task's declared signals per step, {name: (T,)}; objects: the episode's objects, when it declares them)."""
        self.reset(objects)
        rows = np.asarray(rows, float).reshape(-1, 5)
        declared = self._declared_columns(signals)
        count = max(len(rows), column_length(declared))
        labels = np.zeros(count, dtype=np.int16)
        for offset in range(count):
            step_signals = {name: values[offset:offset + 1] for name, values in declared.items()} or None
            self.observe_rows(rows[offset:offset + 1], signals=step_signals)
            labels[offset] = len(self.completed())
        return labels
