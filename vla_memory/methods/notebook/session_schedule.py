"""The schedule of a session: its steps as messages, clock steps and blocks of episodes, in one condition.

    schedule(session, condition, phases)   ("user", step), ("clock", step) and ("episodes", [Episode, ...]) items
    parse_seeds(value)                     "1-20", "1,3,5", a list or one number -> the seeds
    clock_spec(value)                      a clock step: next_day, new_session or {days, at, session}

In the conditions first and every the annotated messages are put in front of the instruction (session.py describes the
conditions); a message whose rules hold only for the day or the session is no longer restated after a clock step ends it.
"""
from __future__ import annotations

import dataclasses
from typing import Any

CONDITIONS = ("profile", "none", "first", "every")
CLOCK_STEPS = {"next_day": {"days": 1, "at": "09:00"}, "new_session": {"session": True}}


@dataclasses.dataclass
class Episode:
    """One episode of a session: its seed, its instruction and what its step annotates."""

    index: int
    seed: int
    instruction: str
    step: int = 0                       # index of its step in the session file
    phase: str = ""
    expect: dict = dataclasses.field(default_factory=dict)
    complaint: str = ""
    correction: list = dataclasses.field(default_factory=list)
    dry_run: dict = dataclasses.field(default_factory=dict)


def parse_seeds(value) -> list[int]:
    """The seeds of an episodes step: one number, a list, or "1-20" / "1,3,5"."""
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(seed) for seed in value]
    seeds = []
    for part in str(value).split(","):
        first, _, last = part.strip().partition("-")
        seeds += list(range(int(first), int(last) + 1)) if last else [int(first)]
    return seeds


def clock_spec(value) -> dict:
    """A clock step: "next_day" (09:00 of the next day), "new_session", or {days, at: "HH:MM", session}. A new day is
    always a new session."""
    if isinstance(value, str) and value not in CLOCK_STEPS:
        raise ValueError(f"clock step {value!r}: use {sorted(CLOCK_STEPS)} or {{days, at, session}}")
    spec = dict(CLOCK_STEPS[value]) if isinstance(value, str) else dict(value or {})
    spec["days"] = int(spec.get("days", 0))
    spec["session"] = bool(spec.get("session")) or spec["days"] > 0
    return spec


class Restated:
    """The annotated messages a condition restates in front of the instruction: all so far (every), or those not yet
    restated (first), each with how long it holds."""

    def __init__(self):
        self.said: list[tuple] = []
        self.pending: list[tuple] = []

    def add(self, step: dict) -> None:
        """A message step; one annotated `expect: []` states no rule and is never restated."""
        if step.get("expect", None) != []:
            validities = [annotation.get("validity") or "until_withdrawn" for annotation in step.get("expect") or []
                          if isinstance(annotation, dict)]
            validity = next(name for name in ("until_withdrawn", "day", "session") if name in (validities or ["until_withdrawn"]))
            self.said.append((step["user"], validity))
            self.pending.append(self.said[-1])

    def clock(self, spec) -> None:
        """A clock step ends the messages that held for the day or the session it ends."""
        clock = clock_spec(spec)
        gone = {"day", "session"} if clock["days"] else {"session"} if clock["session"] else set()
        self.said = [message for message in self.said if message[1] not in gone]
        self.pending = [message for message in self.pending if message[1] not in gone]

    def before(self, condition: str) -> list[tuple]:
        """The messages restated in front of the next instruction in this condition."""
        if condition == "every":
            return self.said
        if condition == "first":
            return self.pending
        return []


def episode_blocks(session: dict, step: dict, step_index: int, next_index: int, condition: str, restated: Restated,
                   single: bool):
    """The episodes of one step, as blocks with one instruction each (every episode its own block if `single`); returns
    the blocks and the next episode index."""
    blocks: list[list[Episode]] = []
    for seed in parse_seeds(step["episodes"]):
        said = [message for message, _ in restated.before(condition)]
        instruction = " ".join(said + [step.get("instruction") or session.get("instruction") or ""]).strip()
        episode = Episode(next_index, seed, instruction, step_index, str(step.get("phase", "")), dict(step.get("expect") or {}),
                          step.get("complaint") or "", list(step.get("correction") or []), dict(step.get("dry_run") or {}))
        next_index += 1
        if condition == "first":
            restated.pending = []
        if blocks and not single and blocks[-1][-1].instruction == episode.instruction:
            blocks[-1].append(episode)
        else:
            blocks.append([episode])
    return blocks, next_index


def schedule(session: dict, condition: str = "profile", phases=None) -> list[tuple[str, Any]]:
    """The session as ("user", step), ("clock", step) and ("episodes", [Episode, ...]) items. A block of episodes runs as one
    unit (one launcher call), never across a message or a clock step; with distillation or votes every episode is its own
    block. `phases` keeps only the episode steps of those phases (messages and clock steps are always kept; episode indices
    stay those of the whole session)."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} not in {CONDITIONS}")
    single = condition == "profile" and bool(session.get("distil") or session.get("feedback"))
    restated = Restated()
    items = []
    next_index = 0
    for step_index, step in enumerate(session["steps"]):
        if "clock" in step:
            items.append(("clock", step))
            restated.clock(step["clock"])
        elif "user" in step:
            items.append(("user", step))
            restated.add(step)
        elif phases is not None and str(step.get("phase", "")) not in phases:
            next_index += len(parse_seeds(step["episodes"]))
        else:
            blocks, next_index = episode_blocks(session, step, step_index, next_index, condition, restated, single)
            items += [("episodes", block) for block in blocks]
    return items
