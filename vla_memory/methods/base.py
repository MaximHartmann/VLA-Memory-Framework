"""Methods that act on a writer's signal: what the policy receives at every call, from the writer's plan and pointer.

    subgoal    the policy gets the current skill as its prompt
    history    the policy gets the instruction with a line per completed skill
    keyframe   the policy gets the instruction, and the wrist picture of the last completed skill's moment in its third
               image slot

A method's prompt() is asked after the writer's before_call, so a writer that decides before the policy call shows its
decision in the same call; before_policy() may add keys to the observation (a picture) and returns fields for the proxy's
log. Other memories plug in as extensions (proxy/extensions.py).

    Method     the interface
    METHODS    registered name -> class or import path; resolve_method(name) for --method
"""
from __future__ import annotations

import argparse
from typing import Any

from ..plugins import resolve

# registered name -> class, or its import path "module:Class" (imported only when the name is given with --method)
METHODS: dict[str, Any] = {
    "subgoal": "vla_memory.methods.subgoal:SubgoalMethod",              # the current skill as the prompt
    "history": "vla_memory.methods.history.method:HistoryMethod",       # the instruction and the completed skills
    "keyframe": "vla_memory.methods.keyframe.method:KeyframeMethod",    # the picture of the last key moment
}
METHOD_ENTRY_POINTS = "vla_memory.methods"


class Method:
    """The interface of a method: every hook has a default, a method overrides what it needs."""

    name = "method"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add the method's own command-line options."""

    @classmethod
    def build(cls, args, task) -> "Method":
        """Construct the method from the parsed options; raise ValueError for an unusable combination."""
        return cls(task)

    def __init__(self, task):
        self.task = task

    def start_episode(self, episode: str, control_keys: dict) -> None:
        """A new episode begins; control_keys: the memory/ keys of its first call, without the prefix."""

    def prompt(self, instruction: str, writer) -> str:
        """The prompt of this call, from the instruction and the writer's state."""
        raise NotImplementedError

    def before_policy(self, observation: dict, control_keys: dict, writer) -> dict:
        """Every call, after the prompt is set: may add keys to the observation. Returns fields for the proxy's log."""
        return {}

    def metadata(self) -> dict:
        """Entries for the proxy's metadata (what the environment client is told); empty: none."""
        return {}

    def close(self) -> None:
        """The proxy stops."""


def resolve_method(spec):
    """The class --method names: a registered name, an entry point of vla_memory.methods, or module:Class."""
    return resolve(spec, METHODS, METHOD_ENTRY_POINTS, "method")


def register_method(name: str, factory) -> None:
    """Make `factory` (a Method class) selectable as --method NAME."""
    METHODS[name] = factory
