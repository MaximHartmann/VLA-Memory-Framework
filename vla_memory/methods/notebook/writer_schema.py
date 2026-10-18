"""The writer's vocabulary: what a proposed entry may say, as JSON schemas built from the task's skills and the presets, and
the texts that describe it to the planning model.

    scopes(task)                 where an entry can act: plan, planner, skill:<each of the task's skills>
    entry_schema(task, ...)      the answer schema of a message's extraction, or of a failed episode's distillation
    merge_schema(ids, ops)       the answer schema of the merge: one operation and the id it acts on
    answer_entries(answer)       the entries of an answer (anything from a backend that does not enforce the schema)
    vocabulary(task, presets)    the skills, the scopes and the presets in words, for the system prompt
    show(text, ...)              one rule as the merge shows it: text, condition, preset, a time limit
"""
from __future__ import annotations

import json

from .entry import STRICTNESS_PRESETS, VALIDITY

KINDS = ("preference", "correction")         # kinds of stored entries
WITHDRAWAL = "withdrawal"                    # a proposal that takes an earlier wish back; never stored itself
OPS = ("ADD", "UPDATE", "DELETE", "NOOP")
PRESET_NOTES = {"double_check": "wait for two 'complete' verdicts in a row before moving on",
                "quick": "check progress only at every second policy call",
                "default": "normal checking, undoes double_check and quick"}
VALIDITY_NOTES = {"day": "today only", "session": "this session only"}   # how the merge sees an entry that does not last
ROLE = "You keep a robot's notebook about its user: short rules that the robot's planner applies in every later episode."
EXTRACT_FIELDS = ("text", "kind", "scope", "validity", "condition", "params")
DISTIL_FIELDS = ("step", "scope", "text", "condition", "params")   # a correction from a log holds until withdrawn


def scopes(task) -> list[str]:
    """Where an entry can act: the plan, the planner's parameters, or the completion rule of one of the task's skills."""
    return ["plan", "planner"] + [f"skill:{s}" for s in task.skills]


def field_schemas(task, presets) -> dict:
    """The schema of every field a proposed entry may have."""
    return {"step": {"type": "integer"},
            "scope": {"type": "string", "enum": scopes(task)},
            "text": {"type": "string", "maxLength": 200},
            "kind": {"type": "string", "enum": [*KINDS, WITHDRAWAL]},
            "validity": {"type": "string", "enum": list(VALIDITY)},
            "condition": {"type": "string", "maxLength": 120},
            "params": {"type": "string", "enum": ["none", *presets]}}


def entry_schema(task, max_items: int = 3, presets=STRICTNESS_PRESETS, distil: bool = False) -> dict:
    """Answer schema of the extraction; of the distillation with the step where it went wrong instead of the kind and the
    validity (always a correction, held until withdrawn)."""
    fields = field_schemas(task, presets)
    order = DISTIL_FIELDS if distil else EXTRACT_FIELDS
    item = {"type": "object", "properties": {k: fields[k] for k in order}, "required": list(order),
            "additionalProperties": False}
    entries = {"type": "array", "items": item, "maxItems": max_items}
    return {"type": "object", "properties": {"entries": entries}, "required": ["entries"], "additionalProperties": False}


def answer_entries(answer) -> list[dict]:
    """The answer's entries; a backend without schema enforcement may return anything."""
    items = answer.get("entries") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def merge_schema(ids, ops=OPS) -> dict:
    return {"type": "object",
            "properties": {"op": {"type": "string", "enum": list(ops)}, "id": {"type": "string", "enum": [*ids, "none"]}},
            "required": ["op", "id"], "additionalProperties": False}


def vocabulary(task, presets: dict) -> str:
    """The skills, the scopes and the presets in words."""
    skills = "; ".join(f'"{s}"' for s in task.skills)
    notes = "; ".join(f"{k}: {PRESET_NOTES.get(k, json.dumps(v))}" for k, v in presets.items())
    return (f"The robot knows exactly these skills: {skills}. For every instruction a planner puts them in order, and a "
            "monitor decides when the current skill is complete. Scopes: plan = which skills to do and in which order; "
            "skill:<skill> = an extra condition for when that skill counts as complete; planner = how carefully "
            f"progress is checked, set by a preset ({notes}).")


def show(text: str, condition: str, params: dict, validity: str, presets: dict) -> str:
    """One rule as the merge shows it: its text, its condition, its preset by name, and a time limit."""
    out = text
    if condition:
        out += f" ({condition})"
    if params:
        preset = next((k for k, v in presets.items() if v == params), json.dumps(params))
        out += f" [preset {preset}]"
    if validity in VALIDITY_NOTES:
        out += f" [{VALIDITY_NOTES[validity]}]"
    return out
