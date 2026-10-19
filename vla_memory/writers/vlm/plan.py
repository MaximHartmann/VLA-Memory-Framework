"""The plan of an episode: the request to the planning model, the fallback to the task's skill order, and the recovery
step that inserts a put-down before a pick when another object is still held.
"""
from __future__ import annotations

from .question import plan_messages
from ...task import TaskSpec

HELD_STATES = ("held_lifted", "held_at_table")


def extension_plan_input(planner, instruction: str) -> tuple[str, dict]:
    """What the extensions add to the plan request: parameter overrides (set on the planner), text, record fields."""
    texts = []
    fields: dict = {}
    for extension in planner.extensions:
        overrides = extension.parameters()
        planner.votes = overrides.get("votes_to_advance", planner.votes)
        planner.call_every = overrides.get("call_every", planner.call_every)
        planner.min_calls = overrides.get("min_calls_per_skill", planner.min_calls)
        text, extension_fields = extension.plan_request(instruction)
        if text:
            texts.append(text)
        fields.update(extension_fields)
    return "\n".join(texts), fields


def ask_for_plan(vlm, task: TaskSpec, instruction: str, extra_text: str, record: dict):
    """The model's plan, or None when the call failed (the record keeps the error)."""
    messages = plan_messages(task, instruction, extra_text)
    try:
        reply = vlm.chat(messages, schema=task.plan_schema(), name="plan", max_tokens=256)
    except Exception as error:   # server down, timeout, bad JSON
        record["error"] = repr(error)
        return None
    plan = (reply.get("json") or {}).get("plan")
    record.update(vlm_plan=plan, ms=reply.get("ms", 0.0), reasoning=reply.get("reasoning"),
                  tokens=[reply.get("prompt_tokens"), reply.get("completion_tokens")])
    return plan


def chosen_plan(task: TaskSpec, plan, record: dict) -> list[str]:
    """The model's plan if every skill is in the vocabulary, else the task's skill order; the record notes the source."""
    if plan and all(skill in task.skills for skill in plan):
        record["source"] = "vlm"
        chosen = list(plan)
    else:
        record["source"] = "fallback_skill_order"
        chosen = list(task.skills)
    record["plan"] = chosen
    return chosen


def recovery_fix(task: TaskSpec, answer: dict, current_skill: str) -> str | None:
    """The put-down skill to insert before a pick when the answer says another object is still held, else None.
    Only for a task with exactly two verbs (pick, place)."""
    verb, object_name = task.parse_skill(current_skill)
    if len(task.verbs) != 2 or verb != task.verbs[0]:
        return None
    for other in task.objects:
        if other != object_name and answer.get(task.field(other)) in HELD_STATES:
            return task.skill(task.verbs[1], other)
    return None
