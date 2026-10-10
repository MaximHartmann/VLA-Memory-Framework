"""Extensions: optional memories that plug into the planner loop without changes to the proxy or the planner.

An extension is a class with the hooks below. Every hook has a do-nothing default, so an extension implements only the
ones it needs. On the command line it is named with `--with NAME` (a registered name, an entry point of the group
vla_memory.extensions, or an import path module:Class) and adds its own options in add_arguments. The proxy and the
planner call the hooks of every extension in the order the extensions were named.

Hooks of the planner (the question to the planning model, and the switch decision):
    memory            a long-term memory store for the planner (a MemoryStore), or None
    plan_request      text added to the plan request, and fields for the plan record
    parameters        votes_to_advance, call_every, min_calls_per_skill for this episode (overrides)
    begin_call        every policy call, before the gate or the question; returns fields for the monitor record
    examples          chat content shown after the retrieved examples: items, number of pictures in them, record fields
    criterion_text    text added after the completion criterion, and record fields
    state_text        a text block after the proprioception sentence
    ask_text          text appended to the question's last sentence
    answer_schema     the monitor's answer schema, changed or returned as it is
    allow_switch      whether a "done" answer may move the plan pointer
    end_call          after a question was answered

Hooks of the proxy (the episode as a whole):
    attach            once, after the planner exists
    start_episode     a new episode begins
    before_decision   every call, before the planner decides
    after_decision    every call, after the planner decided
    on_reply          after the policy answered; may change the reply; returns a record for the proxy log, or None
    end_episode       an episode has ended; returns a record for the planner's log, or None
    log_fields        fields added to every proxy log record
    metadata          entries for the proxy's metadata (what the environment client is told)
    config            what the plan record notes about this extension
    close             the proxy stops
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence

from ..plugins import resolve

EXTENSIONS: dict[str, Any] = {}             # registered name -> class or import path "module:Class"
EXTENSION_ENTRY_POINTS = "vla_memory.extensions"


class Extension:
    """Base class of an extension: every hook does nothing. Override what the memory needs."""

    name = "extension"                       # the key of its config in the plan record and of its metadata

    # ------------------------------------------------------------------ command line and construction
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add the extension's own command-line options."""

    @classmethod
    def build(cls, args, task, built: Sequence["Extension"]) -> "Extension":
        """Construct the extension from the parsed options. `built`: the extensions named before this one (an extension
        that depends on another finds it here). Raise ValueError for an unusable combination of options."""
        return cls()

    def attach(self, planner) -> None:
        """Once, after the planner exists."""

    def config(self) -> dict:
        """What the plan record notes about this extension (empty: nothing)."""
        return {}

    # ------------------------------------------------------------------ the planner's hooks
    def memory(self):
        """A long-term memory store for the planner, or None."""
        return None

    def plan_request(self, instruction: str) -> tuple[str, dict]:
        """Text added to the plan request, and fields for the plan record."""
        return "", {}

    def parameters(self) -> dict:
        """Overrides of votes_to_advance, call_every and min_calls_per_skill for the coming episode."""
        return {}

    def begin_call(self, episode, step: int, images: Sequence, proprio: dict | None, skill: str) -> dict:
        """Every policy call, before the gate or the question. Returns fields for the monitor record."""
        return {}

    def examples(self, pictures_before: int, pictures_after: int, encode) -> tuple[list, int, dict]:
        """Chat content shown after the retrieved examples: the items, how many pictures they hold, record fields.
        pictures_before: pictures already in the question; pictures_after: the current pictures that follow."""
        return [], 0, {}

    def criterion_text(self, skill: str) -> tuple[str, dict]:
        """Text added directly after the completion criterion, and fields for the monitor record."""
        return "", {}

    def state_text(self) -> str:
        """A text block after the proprioception sentence (empty: none)."""
        return ""

    def ask_text(self) -> str:
        """Text appended to the question's last sentence (empty: none)."""
        return ""

    def answer_schema(self, schema: dict) -> dict:
        """The monitor's answer schema, changed as the extension needs, or unchanged."""
        return schema

    def allow_switch(self, skill: str, record: dict) -> bool:
        """Whether a "done" answer may move the plan pointer now."""
        return True

    def end_call(self, record: dict) -> None:
        """After a question was answered (the record holds the answer and whether the pointer moved)."""

    # ------------------------------------------------------------------ the proxy's hooks
    def start_episode(self, episode: str, ctrl: dict) -> None:
        """A new episode begins; ctrl: the control keys of its first call."""

    def before_decision(self, ctrl: dict, images: list, planner) -> None:
        """Every call, before the planner decides."""

    def after_decision(self, ctrl: dict, images: list, record: dict, planner) -> None:
        """Every call, after the planner decided."""

    def on_reply(self, obs: dict, ctrl: dict, reply: dict, skill: str | None, episode: str) -> dict | None:
        """After the policy answered. May change `reply` in place. Returns a record for the proxy log, or None."""
        return None

    def end_episode(self, planner) -> dict | None:
        """An episode has ended. Returns a record for the planner's log, or None."""
        return None

    def log_fields(self) -> dict:
        """Fields added to every proxy log record."""
        return {}

    def metadata(self) -> dict:
        """Entries for the proxy's metadata."""
        return {}

    def close(self) -> None:
        """The proxy stops."""


# ---------------------------------------------------------------------- choosing and building extensions
def register_extension(name: str, factory) -> None:
    """Make `factory` (an Extension class) selectable as --with NAME."""
    EXTENSIONS[name] = factory


def resolve_extension(spec):
    """The class --with names: a registered name, an entry point of vla_memory.extensions, or module:Class."""
    try:
        return resolve(spec, EXTENSIONS, EXTENSION_ENTRY_POINTS, "extension")
    except ValueError as error:
        if EXTENSIONS:
            raise
        raise ValueError(f"unknown extension {spec!r}: no extension is registered under a short name; give an import path "
                         f"module:Class") from error


def build_extensions(classes: Sequence[type], args, task) -> list[Extension]:
    """Build the named extensions in order; each one sees the ones built before it."""
    built: list[Extension] = []
    for cls in classes:
        built.append(cls.build(args, task, built))
    return built


def memory_from(extensions: Sequence[Extension]):
    """The first memory store an extension provides, or None."""
    for extension in extensions:
        store = extension.memory()
        if store is not None:
            return store
    return None


def collect(extensions: Sequence[Extension], hook: str, *args) -> dict:
    """Call a hook that returns a dict on every extension and merge the results."""
    fields: dict = {}
    for extension in extensions:
        fields.update(getattr(extension, hook)(*args))
    return fields


def join_texts(extensions: Sequence[Extension], hook: str, *args) -> str:
    """Call a hook that returns a text on every extension and join the texts."""
    return "".join(getattr(extension, hook)(*args) for extension in extensions)
