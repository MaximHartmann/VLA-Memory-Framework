"""Task specification: everything the planner needs to know about ONE task, loaded from a YAML/JSON file.

A task is a set of objects, a set of verbs, a skill template that combines them into the exact prompt strings the
VLA policy was trained on, a written completion rule per verb, the system texts for the planner model, and the
JSON schemas that force the model's answers into the skill vocabulary. Nothing task-specific lives in the code:
see vla_memory/tasks/red_blue_blocks.yaml for the reference task (two cubes, pick and place).

The task order is the order of all skills of the task: stage by stage (one stage of all verbs unless `plan_stages` says
otherwise), within a stage object by object in the order of `objects`, within an object the stage's verbs in order. An
episode may contain only some of the task's objects (the environment sends their names with the first call); its plan is
then the task order restricted to those objects (default_plan), and the answer schemas follow it.
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
# The roles the code gives to the object states when a task file has no state_roles: the reference task's names
DEFAULT_STATE_ROLES: dict[str, str] = {"rest": "on_table", "held_low": "held_at_table", "held_high": "held_lifted"}


def read_task_file(path) -> dict:
    """The task file as a dict: YAML (needs pyyaml) or JSON, by file suffix."""
    path = pathlib.Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:  # pragma: no cover
            raise ImportError("pip install pyyaml to load YAML task files, or use JSON") from error
        return yaml.safe_load(text)
    return json.loads(text)


@dataclasses.dataclass
class TaskSpec:
    """One task as its file declares it: the objects and verbs, the skill template that combines them into the prompt strings
    the policy was trained on (a verb may have its own, skill_templates), a completion rule per verb, the system texts of the
    planner model, the order of the plan (plan_stages), and the sections the writers and methods read (proprio, fingerprint,
    demo_phases, history, signals, monitor_fields, state_roles, recovery, keyframe). Every section and field after
    object_states is optional; its default reproduces the reference task."""

    name: str
    objects: list[str]                      # e.g. ["red", "blue"]; an episode may hold only some of them (episode_objects)
    verbs: list[str]                        # e.g. ["pick up", "place"]; this order is the default plan per object (in its stage)
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
    skill_templates: dict[str, str] = dataclasses.field(default_factory=dict)   # verb -> skill template of that verb, for a verb whose
                                                                                 # skills do not read like skill_template; {verb} optional
    plan_stages: list[list[str]] | None = None   # the task order: None = one stage of all verbs, object by object; else each stage runs
                                                 # over all objects in turn, e.g. [[pick up, put in box], [take out, stack]]
    state_roles: dict[str, str] = dataclasses.field(default_factory=dict)       # object state names the code needs: rest, held_low,
                                                                                 # held_high (DEFAULT_STATE_ROLES for a role not named)
    recovery: dict[str, str] | None = None       # --recovery: verb about to start -> verb that first puts down another held object;
                                                 # None: a task of two verbs puts down with its second before its first
    keyframe: dict[str, Any] = dataclasses.field(default_factory=dict)          # later placing keyframe: placed_verbs (default: the verbs
                                                                                 # of the set_down rules), rest_z (m, a placed object's centre)

    def __post_init__(self):
        unknown = [verb for verb in (*self.skill_templates, *(self.recovery or {}), *(self.recovery or {}).values())
                   if verb not in self.verbs]
        if unknown:
            raise ValueError(f"skill_templates / recovery: unknown verb(s) {sorted(set(unknown))} (task verbs: {self.verbs})")
        if self.plan_stages is not None:
            named = [verb for stage in self.plan_stages for verb in stage]
            if sorted(named) != sorted(self.verbs):
                raise ValueError(f"plan_stages {self.plan_stages} must name every verb exactly once (task verbs: {self.verbs})")

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, path) -> "TaskSpec":
        """The task of a YAML or JSON file."""
        return cls.from_dict(read_task_file(path))

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        """The task of a task file's content; keys the dataclass does not know are ignored."""
        known = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def load_or_default(cls, path=None) -> "TaskSpec":
        """The task of a --task option: the file when a path is given, else the packaged reference task."""
        if path:
            return cls.load(path)
        return cls.default()

    @classmethod
    def default(cls) -> "TaskSpec":
        """The packaged reference task: two cubes, pick and place."""
        return cls.packaged("red_blue_blocks")

    @classmethod
    def packaged(cls, name: str) -> "TaskSpec":
        """A task file shipped with the package (vla_memory/tasks/NAME.yaml), e.g. red_blue_blocks."""
        return cls.load(cls.packaged_path(name))

    @staticmethod
    def packaged_path(name: str) -> pathlib.Path:
        """The path of a task file shipped with the package (vla_memory/tasks/NAME.yaml)."""
        return _HERE / "tasks" / f"{name}.yaml"

    # ------------------------------------------------------------------ skills
    def _task_order(self) -> list[tuple[str, str]]:
        """Every (verb, object) of the task in the task order (the module docstring)."""
        stages = self.plan_stages or [self.verbs]
        return [(verb, object_name) for stage in stages for object_name in self.objects for verb in stage]

    @property
    def skills(self) -> tuple[str, ...]:
        """All legal skill strings in the task order. With one stage (the default) this is object-major: all verbs of object 1,
        then object 2, ..."""
        return tuple(self.skill(verb, object_name) for verb, object_name in self._task_order())

    def skill(self, verb: str, object_name: str) -> str:
        """The exact prompt string of one verb applied to one object: the verb's skill template (skill_template unless
        skill_templates names one) filled in."""
        return self.skill_templates.get(verb, self.skill_template).format(verb=verb, object=object_name)

    def parse_skill(self, skill: str) -> tuple[str, str]:
        """skill string -> (verb, object); raises ValueError for strings outside the vocabulary."""
        for object_name in self.objects:
            for verb in self.verbs:
                if self.skill(verb, object_name) == skill:
                    return verb, object_name
        raise ValueError(f"unknown skill {skill!r}")

    def field(self, object_name: str) -> str:
        """The JSON field of an object's state in the monitor's answer: object_field filled in."""
        return self.object_field.format(object=object_name)

    def stage_of(self, verb: str) -> int:
        """The number of the plan stage a verb belongs to (0 for a task without plan_stages)."""
        for number, stage in enumerate(self.plan_stages or [self.verbs]):
            if verb in stage:
                return number
        raise ValueError(f"unknown verb {verb!r}")

    # ------------------------------------------------------------------ the objects of an episode
    def episode_objects(self, declared=None) -> list[str]:
        """The objects of one episode in the task's order: the declared ones (what the environment sends as memory/objects), or
        all of the task's objects when none are declared. An empty list and names the task does not know are errors."""
        if declared is None:
            return list(self.objects)
        if isinstance(declared, (str, bytes)):
            declared = [declared]
        names = [name.decode() if isinstance(name, bytes) else str(name) for name in declared]   # msgpack may deliver bytes
        if not names:
            raise ValueError("the episode declares no object")
        unknown = [name for name in names if name not in self.objects]
        if unknown:
            raise ValueError(f"the episode's objects {unknown} are not among the task's objects {self.objects}")
        return [name for name in self.objects if name in names]

    def default_plan(self, objects=None) -> list[str]:
        """The plan of an episode when nothing else plans it: the task order restricted to the episode's objects (all of the
        task's objects when none are declared)."""
        present = self.episode_objects(objects)
        return [self.skill(verb, object_name) for verb, object_name in self._task_order() if object_name in present]

    def restrict_plan(self, plan, objects=None) -> list[str]:
        """The skills of a plan whose object is one of the episode's objects; the plan as it is when none are declared."""
        if objects is None:
            return list(plan)
        present = self.episode_objects(objects)
        restricted = [skill for skill in plan if self.parse_skill(skill)[1] in present]
        if plan and not restricted:
            raise ValueError(f"no skill of the plan is about the episode's objects {present}")
        return restricted

    # ------------------------------------------------------------------ what the code needs to know about the states and verbs
    def state_role(self, role: str) -> str:
        """The name this task gives an object state that the code needs: "rest", "held_low" or "held_high" (state_roles)."""
        return (self.state_roles or {}).get(role, DEFAULT_STATE_ROLES[role])

    def held_states(self) -> tuple[str, ...]:
        """The object states that say the gripper holds the object."""
        return tuple(dict.fromkeys((self.state_role("held_high"), self.state_role("held_low"))))

    def put_down_verbs(self) -> dict[str, str]:
        """For the VLM writer's --recovery: a skill whose verb is a key is preceded by the skill with the value verb on another
        object that is still held. Without a `recovery` section a task of two verbs puts down with the second before the first."""
        if self.recovery is not None:
            return dict(self.recovery)
        if len(self.verbs) == 2:
            return {self.verbs[0]: self.verbs[1]}
        return {}

    def placed_verbs(self) -> tuple[str, ...]:
        """The verbs whose events the keyframe method may take later (keyframe.placed_verbs; else the verbs the rules writer
        completes with a set-down, history.rules; else the second verb)."""
        declared = (self.keyframe or {}).get("placed_verbs")
        if declared:
            return tuple(declared)
        rules = (self.history or {}).get("rules") or {}
        set_down = tuple(verb for verb, rule in rules.items() if (rule or {}).get("trigger") == "set_down")
        return set_down or tuple(self.verbs[1:2])

    # ------------------------------------------------------------------ prompts and schemas
    def plan_system_text(self, objects=None) -> str:
        """The system text of the plan request, with the skills of the episode's objects listed in it."""
        return self.plan_system.format(skills="; ".join(f'"{skill}"' for skill in self.default_plan(objects)))

    def system_text(self, use_wrist: bool) -> str:
        """The system text of the monitor's question: the head, then the part for one camera or for both."""
        return self.system_head.rstrip() + " " + (self.system_both if use_wrist else self.system_ext).lstrip()

    def criterion(self, skill: str) -> str:
        """The written completion rule of a skill: the verb's rule with the object filled in."""
        verb, object_name = self.parse_skill(skill)
        return self.criteria[verb].format(object=object_name)

    def plan_schema(self, objects=None) -> dict:
        """The JSON schema of the plan request's answer: a list of 1 to max_plan_len skills of the episode's objects."""
        return {"type": "object",
                "properties": {"plan": {"type": "array", "items": {"type": "string", "enum": self.default_plan(objects)},
                                        "minItems": 1, "maxItems": self.max_plan_len}},
                "required": ["plan"], "additionalProperties": False}

    def state_fields(self) -> dict[str, Any]:
        """The monitor's state fields besides the object states, in answer order: the task file's monitor_fields, or the
        gripper fields of the reference task (DEFAULT_MONITOR_FIELDS) when it has none."""
        return dict(DEFAULT_MONITOR_FIELDS if self.monitor_fields is None else self.monitor_fields)

    def signal_names(self) -> tuple[str, ...]:
        """The extra signals the task declares (`signals`), in order."""
        return tuple(self.signals or ())

    def monitor_schema(self, brief: bool = False, verdict_only: bool = False, objects=None) -> dict:
        """The monitor's answer: the state of every object of the episode (objects: those declared, default all of the task's),
        the task's state fields (state_fields(): the gripper fields unless the task file declares monitor_fields), a free-text
        reason (not with brief) and current_skill_done. verdict_only: current_skill_done alone (the planner's default; the
        shortest answer)."""
        if verdict_only:
            return {"type": "object", "properties": {"current_skill_done": {"type": "boolean"}},
                    "required": ["current_skill_done"], "additionalProperties": False}
        properties = self._object_state_properties(objects)
        for name, kind in self.state_fields().items():
            if name in properties or name in ("reason", "current_skill_done"):
                raise ValueError(f"monitor_fields: {name!r} is already a field of the answer")
            properties[name] = _field_schema(name, kind)
        if not brief:
            properties["reason"] = {"type": "string", "maxLength": 160}
        properties["current_skill_done"] = {"type": "boolean"}
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    def _object_state_properties(self, objects=None) -> dict:
        """One string field per object of the episode (field()), its enum the task's object states, in the task's order."""
        return {self.field(object_name): {"type": "string", "enum": list(self.object_states)}
                for object_name in self.episode_objects(objects)}


def _field_schema(name: str, kind) -> dict:
    """The JSON schema of one state field of the monitor's answer: `kind` is a list of strings (an enum), "boolean",
    "number", or a JSON schema taken as it is."""
    if isinstance(kind, (list, tuple)):
        return {"type": "string", "enum": [str(value) for value in kind]}
    if isinstance(kind, dict):
        return dict(kind)
    if kind in _FIELD_TYPES:
        return dict(_FIELD_TYPES[kind])
    raise ValueError(f"monitor_fields.{name}: a list of strings, 'boolean', 'number' or a JSON schema, got {kind!r}")
