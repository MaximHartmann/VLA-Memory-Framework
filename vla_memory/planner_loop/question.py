"""The texts the planner sends to the planning model: the plan request once per episode, and the monitor question at
every policy call. Pure functions of their inputs, so the exact wording lives in one place.
"""
from __future__ import annotations

from typing import Sequence

from ..task import TaskSpec

NOW = "Now the CURRENT observation. "
COMBINE = ("the scene NOW. Combine the proprioception with what you see: the cameras tell you where the objects are, "
           "the sensors tell you whether the gripper is closed on something and how high the hand is.")


def plan_messages(task: TaskSpec, instruction: str, extra_text: str = "") -> list[dict]:
    """The chat messages of the plan request. extra_text: lines an extension adds to the request."""
    user = f"Instruction: \"{instruction}\"\n"
    if extra_text:
        user += extra_text + "\n"
    user += "Return the ordered skill list as JSON."
    return [{"role": "system", "content": task.plan_system_text()}, {"role": "user", "content": user}]


def plan_text(plan: Sequence[str], index: int) -> str:
    """The plan as a numbered list; the current skill is marked, the finished ones too."""
    lines = []
    for i, skill in enumerate(plan):
        if i == index:
            mark = "  <- CURRENT"
        elif i < index:
            mark = "  (done)"
        else:
            mark = ""
        lines.append(f"  {i + 1}. {skill}{mark}")
    return "\n".join(lines)


def pictures_text(first_picture: int, two_cameras: bool) -> str:
    """Which picture numbers show the current observation."""
    if two_cameras:
        return f"Picture {first_picture} (external camera) and Picture {first_picture + 1} (wrist camera) show"
    return f"Picture {first_picture} (external camera) shows"


def ask_text(task: TaskSpec, verdict_only: bool) -> str:
    """The question's last sentence: the verdict alone, or the object states first."""
    if verdict_only:
        return "Decide whether the current skill is complete."
    if len(task.objects) <= 2:
        objects = " and ".join(f"{o} {task.object_noun}" for o in task.objects)
    else:
        objects = "every " + task.object_noun
    return f"Report the state of the {objects} and the gripper, then decide whether the current skill is complete."


def question_text(task: TaskSpec, plan: Sequence[str], index: int, criterion_extra: str, proprio_text: str,
                  state_text: str, first_picture: int, two_cameras: bool, ask: str, has_examples: bool) -> str:
    """The text item of the monitor question (between the examples and the current pictures)."""
    skill = plan[min(index, len(plan) - 1)]
    text = NOW if has_examples else ""
    text += f"Plan:\n{plan_text(plan, index)}\n\n"
    text += f"Current skill: \"{skill}\".\nCompletion criterion: {task.criterion(skill)}{criterion_extra}\n\n"
    if proprio_text:
        text += proprio_text + "\n\n"
    if state_text:
        text += state_text + "\n\n"
    text += f"{pictures_text(first_picture, two_cameras)} {COMBINE} {ask}"
    return text
