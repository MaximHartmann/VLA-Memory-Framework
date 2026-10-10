"""Task specification: everything the planner needs to know about ONE task, loaded from a YAML/JSON file.

A task is a set of objects, a set of verbs, a skill template that combines them into the exact prompt strings the
VLA policy was trained on, a written completion rule per verb, the system texts for the planner model, and the
JSON schemas that force the model's answers into the skill vocabulary. Nothing task-specific lives in the code:
see vla_memory/tasks/red_blue_blocks.yaml for the reference task (two cubes, pick and place) and
vla_memory/tasks/drawers.yaml for a task without pick and place (open and close drawers; extra signals, signal triggers and
its own monitor fields).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
# The monitor's state fields besides the object states when a task file has no monitor_fields: the gripper of the reference task
DEFAULT_MONITOR_FIELDS: dict[str, Any] = {"gripper": ["open", "closed", "unsure"], "gripper_high_above_table": "boolean"}
_FIELD_TYPES = {"boolean": {"type": "boolean"}, "number": {"type": "number"}}


def _load_file(path) -> dict:
    """The task file as a dict: YAML (needs pyyaml) or JSON, by file suffix."""
    path = pathlib.Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install pyyaml to load YAML task files, or use JSON") from e
        return yaml.safe_load(text)
    return json.loads(text)


@dataclasses.dataclass
class TaskSpec:
    name: str
    objects: list[str]                      # e.g. ["red", "blue"]
    verbs: list[str]                        # e.g. ["pick up", "place"]; this order is the default plan per object
    skill_template: str                     # e.g. "{verb} the {object} block"
    criteria: dict[str, str]                # verb -> completion rule text, may use {object}
    system_head: str
    system_ext: str                         # appended to system_head when only the external camera is used
    system_both: str                        # appended when external + wrist camera are used
    plan_system: str                        # may use {skills}
    object_states: list[str]                # enum of the per-object state field
    object_field: str = "{object}_cube"     # JSON field name of an object's state in the monitor answer
    object_noun: str = "cube"               # how objects are called in generated text ("red cube = on_table")
    max_plan_len: int = 8                   # most skills in one plan (the plan schema's maxItems)
    proprio: dict[str, Any] = dataclasses.field(default_factory=dict)      # closed_threshold, note; open_width, empty_width,
                                                                           # squeeze_force (gripper signals of the rules writer)
    fingerprint: dict[str, Any] = dataclasses.field(default_factory=dict)  # colour rules for ColourBlobFingerprint
    demo_phases: dict[str, Any] = dataclasses.field(default_factory=dict)  # phase table for labels.PhaseLabeller
    history: dict[str, Any] = dataclasses.field(default_factory=dict)      # history-appended prompt: prefix, none, events, joiner, suffix,
                                                                           # max_lines, rules (the rules writer's triggers per verb)
    instruction_examples: list[str] = dataclasses.field(default_factory=list)
    notes: str = ""
    signals: Any = dataclasses.field(default_factory=dict)  # the robot's signals beyond memory/proprio_steps that the rules use, sent
                                                            # as memory/signal_steps: names, or name -> description (unit, source)
    monitor_fields: dict[str, Any] | None = None   # the monitor's state fields besides the objects: name -> list of strings (enum),
                                                   # "boolean", "number" or a JSON schema; None = DEFAULT_MONITOR_FIELDS (gripper)

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, path) -> "TaskSpec":
        d = _load_file(path)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def default(cls) -> "TaskSpec":
        return cls.load(_HERE / "tasks" / "red_blue_blocks.yaml")

    @classmethod
    def packaged(cls, name: str) -> "TaskSpec":
        """A task file shipped with the package (vla_memory/tasks/NAME.yaml): red_blue_blocks, drawers."""
        return cls.load(_HERE / "tasks" / f"{name}.yaml")

    # ------------------------------------------------------------------ skills
    @property
    def skills(self) -> tuple[str, ...]:
        """All legal skill strings, object-major (all verbs of object 1, then object 2, ...)."""
        return tuple(self.skill(v, o) for o in self.objects for v in self.verbs)

    def skill(self, verb: str, obj: str) -> str:
        return self.skill_template.format(verb=verb, object=obj)

    def parse_skill(self, skill: str) -> tuple[str, str]:
        """skill string -> (verb, object); raises ValueError for strings outside the vocabulary."""
        for o in self.objects:
            for v in self.verbs:
                if self.skill(v, o) == skill:
                    return v, o
        raise ValueError(f"unknown skill {skill!r}")

    def field(self, obj: str) -> str:
        return self.object_field.format(object=obj)

    # ------------------------------------------------------------------ prompts and schemas
    def plan_system_text(self) -> str:
        return self.plan_system.format(skills="; ".join(f'"{s}"' for s in self.skills))

    def system_text(self, use_wrist: bool) -> str:
        return self.system_head.rstrip() + " " + (self.system_both if use_wrist else self.system_ext).lstrip()

    def criterion(self, skill: str) -> str:
        verb, obj = self.parse_skill(skill)
        return self.criteria[verb].format(object=obj)

    def plan_schema(self) -> dict:
        return {"type": "object",
                "properties": {"plan": {"type": "array", "items": {"type": "string", "enum": list(self.skills)},
                                        "minItems": 1, "maxItems": self.max_plan_len}},
                "required": ["plan"], "additionalProperties": False}

    def state_fields(self) -> dict[str, Any]:
        """The monitor's state fields besides the object states, in answer order: the task file's monitor_fields, or the
        gripper fields of the reference task (DEFAULT_MONITOR_FIELDS) when it has none."""
        return dict(DEFAULT_MONITOR_FIELDS if self.monitor_fields is None else self.monitor_fields)

    def signal_names(self) -> tuple[str, ...]:
        """The extra signals the task declares (`signals`), in order."""
        return tuple(self.signals or ())

    def monitor_schema(self, brief: bool = False, verdict_only: bool = False) -> dict:
        """The monitor's answer: every object's state, the task's state fields (state_fields(): the gripper fields unless the
        task file declares monitor_fields), a free-text reason (not with brief) and current_skill_done. verdict_only:
        current_skill_done alone (the planner's default; the shortest answer)."""
        if verdict_only:
            return {"type": "object", "properties": {"current_skill_done": {"type": "boolean"}},
                    "required": ["current_skill_done"], "additionalProperties": False}
        props = self._object_state_properties()
        for name, kind in self.state_fields().items():
            if name in props or name in ("reason", "current_skill_done"):
                raise ValueError(f"monitor_fields: {name!r} is already a field of the answer")
            props[name] = _field_schema(name, kind)
        if not brief:
            props["reason"] = {"type": "string", "maxLength": 160}
        props["current_skill_done"] = {"type": "boolean"}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def _object_state_properties(self) -> dict:
        """One string field per object (field()), its enum the task's object states, in the objects' order."""
        return {self.field(o): {"type": "string", "enum": list(self.object_states)} for o in self.objects}


def _field_schema(name: str, kind) -> dict:
    """The JSON schema of one state field of the monitor's answer: `kind` is a list of strings (an enum), "boolean",
    "number", or a JSON schema taken as it is."""
    if isinstance(kind, (list, tuple)):
        return {"type": "string", "enum": [str(x) for x in kind]}
    if isinstance(kind, dict):
        return dict(kind)
    if kind in _FIELD_TYPES:
        return dict(_FIELD_TYPES[kind])
    raise ValueError(f"monitor_fields.{name}: a list of strings, 'boolean', 'number' or a JSON schema, got {kind!r}")
