"""The plan of an episode: the request to the planning model, the fallback to the task's skill order, and the recovery
step that inserts a put-down before a pick when another object is still held. An episode that declares its objects is
planned over the skills of those objects only (TaskSpec.default_plan).
"""
from __future__ import annotations

from .question import plan_messages
from ...task import TaskSpec


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


def ask_for_plan(vlm, task: TaskSpec, instruction: str, extra_text: str, record: dict, objects=None):
    """The model's plan, or None when the call failed (the record keeps the error). objects: the episode's objects when the
    environment declares them."""
    messages = plan_messages(task, instruction, extra_text, objects)
    try:
        reply = vlm.chat(messages, schema=task.plan_schema(objects), name="plan", max_tokens=256)
    except Exception as error:   # server down, timeout, bad JSON
        record["error"] = repr(error)
        return None
    plan = (reply.get("json") or {}).get("plan")
    record.update(vlm_plan=plan, ms=reply.get("ms", 0.0), reasoning=reply.get("reasoning"),
                  tokens=[reply.get("prompt_tokens"), reply.get("completion_tokens")])
    return plan


def chosen_plan(task: TaskSpec, plan, record: dict, objects=None) -> list[str]:
    """The model's plan if every skill is one of the episode's, else the task's skill order; the record notes the source."""
    skills = task.default_plan(objects)
    if plan and all(skill in skills for skill in plan):
        record["source"] = "vlm"
        chosen = list(plan)
    else:
        record["source"] = "fallback_skill_order"
        chosen = skills
    record["plan"] = chosen
    return chosen


def recovery_fix(task: TaskSpec, answer: dict, current_skill: str, objects=None) -> str | None:
    """The put-down skill to insert before a skill whose verb the task lets recovery precede (TaskSpec.put_down_verbs) when the
    answer says another object of the episode is still held, else None."""
    verb, object_name = task.parse_skill(current_skill)
    put_down = task.put_down_verbs().get(verb)
    if put_down is None:
        return None
    for other in task.episode_objects(objects):
        if other != object_name and answer.get(task.field(other)) in task.held_states():
            return task.skill(put_down, other)
    return None
