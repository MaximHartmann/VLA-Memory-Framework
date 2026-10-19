"""The texts the planner sends to the planning model: the plan request once per episode, and the monitor question at
every policy call, composed with the retrieved examples, the extensions' content and the current pictures. Pure
functions of their inputs, so the exact wording lives in one place.
"""
from __future__ import annotations

from typing import Sequence

from ...proxy.extensions import join_texts
from ...task import TaskSpec
from ...vlm import encode_png

NOW = "Now the CURRENT observation. "
COMBINE = ("the scene NOW. Combine the proprioception with what you see: the cameras tell you where the objects are, "
           "the sensors tell you whether the gripper is closed on something and how high the hand is.")


# ---------------------------------------------------------------------- the plan request
def plan_messages(task: TaskSpec, instruction: str, extra_text: str = "") -> list[dict]:
    """The chat messages of the plan request. extra_text: lines an extension adds to the request."""
    user = f"Instruction: \"{instruction}\"\n"
    if extra_text:
        user += extra_text + "\n"
    user += "Return the ordered skill list as JSON."
    return [{"role": "system", "content": task.plan_system_text()}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------- the monitor question's text
def plan_text(plan: Sequence[str], index: int) -> str:
    """The plan as a numbered list; the current skill is marked, the finished ones too."""
    lines = []
    for position, skill in enumerate(plan):
        if position == index:
            mark = "  <- CURRENT"
        elif position < index:
            mark = "  (done)"
        else:
            mark = ""
        lines.append(f"  {position + 1}. {skill}{mark}")
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
        objects = " and ".join(f"{object_name} {task.object_noun}" for object_name in task.objects)
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


# ---------------------------------------------------------------------- the monitor question's messages
def monitor_messages(planner, pictures: list, hits: list, record: dict) -> list[dict]:
    """The chat messages of the monitor question of a planner (planner.Planner): the retrieved examples, the extensions'
    content, the text and the current pictures. The extensions' record fields go into the record."""
    skill = planner.current_skill
    items: list[dict] = []
    if hits:
        items = planner.memory.fewshot_content(hits, planner.current_label(), skill, encode_png)
    picture_count = _picture_count(items)
    extension_items = _extension_examples(planner, picture_count, len(pictures), record)
    items += extension_items
    picture_count += _picture_count(extension_items)
    two_cameras = len(pictures) >= 2
    text = question_text(planner.task, planner.plan, planner.skill_index, _extension_criterion_text(planner, skill, record),
                         planner.proprio.text(), join_texts(planner.extensions, "state_text"), picture_count + 1, two_cameras,
                         ask_text(planner.task, planner.verdict_only) + join_texts(planner.extensions, "ask_text"),
                         bool(hits) or bool(extension_items))
    items.append({"type": "text", "text": text})
    for picture in pictures:
        items.append({"type": "image_url", "image_url": {"url": planner.encode_picture(picture)}})
    return [{"role": "system", "content": planner.task.system_text(two_cameras)}, {"role": "user", "content": items}]


def _picture_count(items: list) -> int:
    """How many pictures a list of chat content items holds."""
    return sum(1 for item in items if item.get("type") == "image_url")


def _extension_examples(planner, pictures_before: int, pictures_after: int, record: dict) -> list:
    """Content the extensions show after the retrieved examples (their record fields go into the record)."""
    items: list = []
    for extension in planner.extensions:
        extension_items, picture_count, fields = extension.examples(pictures_before, pictures_after, planner.encode_picture)
        items += extension_items
        pictures_before += picture_count
        record.update(fields)
    return items


def _extension_criterion_text(planner, skill: str, record: dict) -> str:
    """The text the extensions add after the completion criterion (their record fields go into the record)."""
    text = ""
    for extension in planner.extensions:
        extension_text, fields = extension.criterion_text(skill)
        text += extension_text
        record.update(fields)
    return text
