"""The rules author's prompt: what the model is told, and the JSON schema its answer must follow.

build_messages() composes the chat messages of one authoring request:
    system   what the history writer does, the robot's signals, the grasp cycle and its two triggers, every parameter with
             its unit and physical meaning, the gripper thresholds of the task file's proprio section, and what a good rule
             set does (never early, never missed; up to about a second late is harmless)
    user     the task: objects, the plan's skills, the history phrases, and, by variant,
               easy   the completion criteria of the task file (texts with numbers, e.g. "the cube lifted about 9 cm")
               hard   no criteria; statistics of the writer's signals around each label switch of a few labelled
                      demonstrations (author_demos.demo_summary_text)
               none   nothing more: skill names and history phrases only
answer_schema() constrains the answer: per verb a short reason and one trigger of the engine with its parameters in cm.
"""
from __future__ import annotations

import copy
from typing import Any

from ..task import TaskSpec
from .author_rules_spec import PARAMS, REASON_MAX, VARIANTS, _cm, gripper_thresholds
from .prompt import HistoryPrompt
from .rules import TRIGGERS

SYSTEM = """\
You configure the history writer of a robot arm that is driven by a learned policy (a vision-language-action model). The \
policy receives the task instruction plus a short history of the skills completed so far, for example "<instruction> \
History: picked up the red block." The history writer adds a line when a skill is complete. It uses no model and no camera: \
it is a fixed rule engine on the robot's own gripper and hand signals, and you write its rules for one task. For each verb \
of the task you choose which of the engine's two triggers completes it, and the trigger's thresholds.

SIGNALS, one row per control step{rate}:
- finger width: the distance between the two fingers.
- squeeze: whether the fingers press on something: the finger force, or 1/0 from a gripper that only reports "grasped".
- fingertip position x, y, z (z points up): the point between the fingertips, computed from the measured joint angles.

GRASP CYCLE, a state machine the engine follows at every control step. The gripper thresholds are fixed for this robot: \
OPEN = {open_cm} cm, EMPTY = {empty_cm} cm, SQUEEZE = {force}.
1. free -> holding: the fingers stopped between EMPTY and OPEN apart and the squeeze is above SQUEEZE. The fingertip \
position at this step is the GRASP POINT. A gripper that closes on nothing stops at its commanded width without squeeze, so \
an empty grasp never starts a cycle.
2. holding -> lifted: still squeezing, the fingertips are at least RISE above the grasp point. This fires the trigger \
"lift". With AT_TOP set, the trigger also waits until the fingertips rise less than AT_TOP in one control step, i.e. for the \
top of the lift. If the fingers open, or close fully, before that, the cycle returns to free without a trigger.
3. lifted -> released: the fingers open (width at least OPEN) while the fingertips are less than RELEASE_BELOW above the \
grasp point: the object was set down, or let go just above the surface. The fingertip position at this step is the RELEASE \
POINT. Opening higher up is a drop: the cycle returns to free without a trigger.
4. released -> free: the fingertips are at least UP above the release point, or at least AWAY from it in any direction \
(straight-line distance). This fires the trigger "set_down". A touch after the release (squeeze again, then open again \
without a lift) keeps the set-down pending.

FROM TRIGGERS TO HISTORY LINES:
- Every verb of the task is completed by one trigger, "lift" or "set_down". The rule set needs at least one verb with each \
trigger.
- The writer follows the plan skill by skill. A trigger completes the current skill only if the skill's verb uses that \
trigger; otherwise nothing is written. The plan pointer only moves forward, so an object that slipped out and is lifted \
again does not produce a second line.
- The new line appears in the policy's prompt at its next call{call}.

PARAMETERS (lengths in cm, measured with the fingertip position):
- rise_cm (lift, required): how far the fingertips must rise above the grasp point. Too small: the line can appear before \
the object is really lifted. Too large: a lift that ends lower never fires.
- at_top_cm_per_step (lift, optional, null = off): fire only once the fingertips rise less than this in one control \
step, i.e. at the top of the lift{per_s}. It never fires earlier than rise_cm alone, only later.
- release_below_cm (set_down, required): an opening of the fingers counts as a set-down only while the fingertips are less \
than this above the grasp point. Too small: an object put down a little higher than where it was grasped counts as dropped \
and its line is never written. Too large: an object dropped from a height counts as set down.
- up_cm (set_down, required): how far the fingertips must rise above the release point. Too small: the line can appear \
while the hand is still at the object. Too large: a retreat that ends lower never fires.
- away_cm (set_down, optional, null = off): the alternative straight-line distance from the release point. It is never \
smaller than the rise, so an away_cm below up_cm makes up_cm irrelevant; use it to catch a sideways retreat.

WHAT A GOOD RULE SET DOES:
- The policy was trained on prompts whose history changes at fixed moments of each skill. At run time it follows its \
history: as soon as a line appears, it moves on to the next skill.
- A line that appears too early is the worst error: the policy treats the skill as done and moves on, for example it lets \
go of an object it has not lifted yet. A line that never appears is just as bad: the policy waits for it.{late}
- So every threshold must be reached only once the skill is complete, and it must be reached in every execution, also in \
imperfect ones. A learned policy executes less precisely than its demonstrations: a lift may end lower, an object may be let \
go slightly above the surface, the hand may retreat sideways. When in doubt, choose the later moment, but never one that \
some executions do not reach.

ANSWER: JSON only. For each verb of the task: "reason", one or two short sentences on how you derived the numbers, then \
"rule" with the trigger and its parameters in cm."""

EASY_HEAD = ("COMPLETION CRITERIA, the definitions the policy's training labels follow (<object> stands for the object of the "
             "skill). They were written for a camera monitor that sees the gripper's closedness (0 = open, 1 = closed) and the "
             "hand's height above the table; the history writer reads finger width, squeeze and fingertip position instead.")
HARD_HEAD = ("MEASUREMENTS FROM DEMONSTRATIONS, the recordings the policy was trained on. Their labels switch to the next "
             "history line at the moments described below. The values are the history writer's own signals, relative to the "
             "grasp and release points of the grasp cycle (min / median / max over the demonstrations; one control step = "
             "{step}).")


def build_messages(task: TaskSpec, variant: str = "easy", demo_summary: str | None = None, control_hz: float | None = None,
                   call_every: int | None = None, late_ok_s: float | None = 1.0, **gripper_overrides) -> list[dict]:
    """The chat messages of one authoring request. control_hz and call_every (the policy's call period in control steps)
    only add timing sentences; late_ok_s says how late a line may appear without cost (None: not said)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "hard" and not demo_summary:
        raise ValueError("variant 'hard' needs demo_summary (demo_summary_text of a few labelled demonstrations)")
    system = _system_text(task, control_hz, call_every, late_ok_s, gripper_overrides)
    user = _task_text(task) + _variant_text(task, variant, demo_summary, control_hz)
    user.append("Write one rule for each verb: " + ", ".join(f'"{verb}"' for verb in task.verbs) + ".")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user)}]


def _system_text(task: TaskSpec, control_hz, call_every, late_ok_s, gripper_overrides: dict) -> str:
    """SYSTEM with the gripper thresholds and the optional timing sentences filled in."""
    gripper = gripper_thresholds(task, **gripper_overrides)
    rate = f" (the controller runs at {control_hz:g} Hz)" if control_hz else ""
    call = ""
    if call_every:
        call = f", every {call_every} control steps"
        if control_hz:
            call += f" (about {call_every / control_hz:.2f} s)"
    per_s = f" (at {control_hz:g} Hz, 0.1 cm per step is {0.1 * control_hz:g} cm/s)" if control_hz else ""
    late = f" A line that appears up to about {late_ok_s:g} s late costs nothing." if late_ok_s else ""
    force = f"{gripper['squeeze_force']:g} (a force in N, or 0.5 for a grasped flag)"
    return SYSTEM.format(rate=rate, open_cm=_cm(gripper["open_width"]), empty_cm=_cm(gripper["empty_width"]), force=force,
                         call=call, per_s=per_s, late=late)


def _task_text(task: TaskSpec) -> list[str]:
    """The task: objects and verbs, the plan, the history phrase of each verb, and an example prompt."""
    history = HistoryPrompt(task)
    phrases = []
    for verb in task.verbs:
        template = history.events.get(verb, task.skill_template.format(verb=verb, object="{object}"))
        phrases.append(f'"{verb}" -> "{template.format(object="<object>")}"')
    plan = "; ".join(f"{i + 1}. {skill}" for i, skill in enumerate(task.skills))
    instruction = (task.instruction_examples or ["<instruction>"])[0]
    example = history.render(instruction, list(task.skills[:2]))
    return [f'TASK "{task.name}". Objects: {", ".join(task.objects)}. Verbs: {", ".join(task.verbs)}.',
            f"The plan, in order (the skills the policy was trained on): {plan}.",
            f"History line written when a skill is complete: {'; '.join(phrases)}.",
            f'Example prompt after the first two skills: "{example}"']


def _variant_text(task: TaskSpec, variant: str, demo_summary: str | None, control_hz) -> list[str]:
    """What the variant adds: the completion criteria (easy), the demonstration statistics (hard), nothing (none)."""
    if variant == "easy":
        lines = [EASY_HEAD]
        for verb in task.verbs:
            if verb in (task.criteria or {}):
                criterion = " ".join(str(task.criteria[verb]).replace("{object}", "<object>").split())
                lines.append(f'- "{verb}": {criterion}')
        return lines
    if variant == "hard":
        step = f"{1000 / control_hz:.0f} ms" if control_hz else "one row of the signals"
        return [HARD_HEAD.format(step=step), demo_summary.strip()]
    return []


def answer_schema(task: TaskSpec) -> dict:
    """Per verb: a short reason, then the rule: one trigger of the engine with its parameters in cm, within PARAMS' bounds
    (optional parameters may be null)."""
    verb = {"type": "object",
            "properties": {"reason": {"type": "string", "maxLength": REASON_MAX},
                           "rule": {"anyOf": [_rule_alternative(trigger) for trigger in TRIGGERS]}},
            "required": ["reason", "rule"], "additionalProperties": False}
    return {"type": "object", "properties": {v: copy.deepcopy(verb) for v in task.verbs}, "required": list(task.verbs),
            "additionalProperties": False}


def _rule_alternative(trigger: str) -> dict:
    """The schema of one trigger's rule: the trigger name and its parameters within PARAMS' bounds."""
    properties: dict[str, Any] = {"trigger": {"type": "string", "enum": [trigger]}}
    for field, _, low, high, optional in PARAMS[trigger].values():
        kind = ["number", "null"] if optional else "number"
        properties[field] = {"type": kind, "minimum": low, "maximum": high}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
