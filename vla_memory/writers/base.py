"""Writers: the components that signal that a step of the task is done.

A writer keeps the plan (the VLM planner's plan, or the task order of the episode's objects) and a forward-only pointer,
and moves the pointer when it decides that the current skill is complete. The methods (methods/) turn that state
into what the policy receives: the current skill as the prompt, the instruction with a line per completed skill, or the
picture of the moment a skill was completed. Every writer drives every method.

Timing: a writer decides either before the policy call, from the call's inputs (before_call), so that the prompt of the
same call shows it, or after it, from the policy's reply (after_call), so that it shows from the next call on.

    Writer       the interface: plan, skill_index, done_all, current_skill, completed(), start_episode(instruction,
                 episode_id, objects), before_call, after_call, monitor (what the extensions see)
    Call         what before_call sees of one policy call: the instruction, the control keys, the observation
    WRITERS      registered name -> class or import path; resolve_writer(name) for --writer
"""
from __future__ import annotations

import argparse
import dataclasses
from typing import Any, Sequence

import numpy as np

from ..plugins import resolve

# registered name -> class, or its import path "module:Class" (imported only when the name is given with --writer)
WRITERS: dict[str, Any] = {
    "oracle": "vla_memory.writers.oracle:OracleWriter",        # the simulator's ground truth (simulation only)
    "off": "vla_memory.writers.oracle:OffWriter",              # no writer: the policy gets the instruction
    "rules": "vla_memory.writers.rules.writer:RulesWriter",    # rules on the robot's gripper and hand signals
    "vlm": "vla_memory.writers.vlm.writer:VlmWriter",          # the VLM planner, with the exemplar store
    "head": "vla_memory.writers.head.writer:HeadWriter",       # the policy's own event head, from its reply
}
WRITER_ENTRY_POINTS = "vla_memory.writers"


@dataclasses.dataclass
class Call:
    """One policy call as a writer sees it before the call: the full instruction, the control keys (without the memory/
    prefix) and the policy's observation; image_keys name the policy's two cameras in the observation."""

    instruction: str
    control_keys: dict
    observation: dict
    image_keys: tuple = ("observation/image", "observation/wrist_image")

    @property
    def step(self) -> int:
        """The environment's step of this call (-1 when it sends none)."""
        return int(self.control_keys.get("step", -1))

    def wrist(self):
        """The raw wrist frame the environment sent, else the policy's own wrist image."""
        fallback = self.observation.get(self.image_keys[1]) if len(self.image_keys) > 1 else None
        return self.control_keys.get("wrist_raw", fallback)

    def images(self) -> list:
        """The pictures of this call: the raw frames the environment sent, else the policy's own images."""
        exterior = self.control_keys.get("exterior_raw", self.observation.get(self.image_keys[0]))
        images = [np.asarray(exterior)]
        wrist = self.wrist()
        if wrist is not None:
            images.append(np.asarray(wrist))
        return images


class Writer:
    """The interface of a writer. A subclass keeps plan, skill_index and done_all up to date and overrides the hook of its
    timing (before_call or after_call)."""

    name = "writer"
    drives_prompt = True             # False: the policy gets the instruction as it is (the writer "off")
    passthrough: str | None = None   # an environment prompt that is not a skill, forwarded as it is (the oracle only)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add the writer's own command-line options."""

    @classmethod
    def build(cls, args, task, extensions: Sequence = ()) -> "Writer":
        """Construct the writer from the parsed options; raise ValueError for an unusable combination."""
        return cls(task)

    @property
    def monitor(self):
        """The writer as the extensions see it (attach, before_policy, end_episode, the skill of on_reply): the writer
        itself, or None for a writer that observes nothing itself (the oracle, off)."""
        return self

    @property
    def current_skill(self) -> str | None:
        """The skill the pointer is on (the last one once the plan is complete)."""
        return self.plan[min(self.skill_index, len(self.plan) - 1)]

    def completed(self) -> list[str]:
        """The skills completed so far, in order."""
        if self.done_all:
            return list(self.plan)
        return list(self.plan[:self.skill_index])

    def start_episode(self, instruction: str, episode_id, objects: Sequence[str] | None = None) -> dict | None:
        """A new episode: reset, and return the plan record (plan, source, ms), or None for a writer without a plan. objects:
        the objects of this episode when the environment declares them (memory/objects; the relay passes the argument only
        then), else the task's objects: the plan covers these objects only (TaskSpec.default_plan, restrict_plan)."""
        return None

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """Before the policy call: decide from the call's inputs. Returns fields for the proxy's log."""
        return {}

    def after_call(self, reply: dict) -> dict:
        """After the policy call: decide from the policy's reply. Returns fields for the proxy's log."""
        return {}


def resolve_writer(spec):
    """The class --writer names: a registered name, an entry point of vla_memory.writers, or module:Class."""
    return resolve(spec, WRITERS, WRITER_ENTRY_POINTS, "writer")


def register_writer(name: str, factory) -> None:
    """Make `factory` (a Writer class) selectable as --writer NAME."""
    WRITERS[name] = factory
