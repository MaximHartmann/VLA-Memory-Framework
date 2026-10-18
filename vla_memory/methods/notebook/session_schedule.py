"""The schedule of a session: its steps as messages, clock steps and blocks of episodes, in one condition.

    schedule(session, condition, phases)   ("user", step), ("clock", step) and ("episodes", [Episode, ...]) items
    parse_seeds(v)                         "1-20", "1,3,5", a list or one number -> the seeds
    clock_spec(v)                          a clock step: next_day, new_session or {days, at, session}

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
    index: int
    seed: int
    instruction: str
    step: int = 0                       # index of its step in the session file
    phase: str = ""
    expect: dict = dataclasses.field(default_factory=dict)
    complaint: str = ""
    correction: list = dataclasses.field(default_factory=list)
    dry_run: dict = dataclasses.field(default_factory=dict)


def parse_seeds(v) -> list[int]:
    if isinstance(v, int):
        return [v]
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    out = []
    for part in str(v).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def clock_spec(v) -> dict:
    """A clock step: "next_day" (09:00 of the next day), "new_session", or {days, at: "HH:MM", session}. A new day is
    always a new session."""
    if isinstance(v, str) and v not in CLOCK_STEPS:
        raise ValueError(f"clock step {v!r}: use {sorted(CLOCK_STEPS)} or {{days, at, session}}")
    d = dict(CLOCK_STEPS[v]) if isinstance(v, str) else dict(v or {})
    d["days"] = int(d.get("days", 0))
    d["session"] = bool(d.get("session")) or d["days"] > 0
    return d


class Restated:
    """The annotated messages a condition restates in front of the instruction: all so far (every), or those not yet
    restated (first), each with how long it holds."""

    def __init__(self):
        self.said: list[tuple] = []
        self.pending: list[tuple] = []

    def add(self, step: dict) -> None:
        if step.get("expect", None) != []:
            vs = [g.get("validity") or "until_withdrawn" for g in step.get("expect") or [] if isinstance(g, dict)]
            validity = next(v for v in ("until_withdrawn", "day", "session") if v in (vs or ["until_withdrawn"]))
            self.said.append((step["user"], validity))
            self.pending.append(self.said[-1])

    def clock(self, spec) -> None:
        """A clock step ends the messages that held for the day or the session it ends."""
        c = clock_spec(spec)
        gone = {"day", "session"} if c["days"] else {"session"} if c["session"] else set()
        self.said = [m for m in self.said if m[1] not in gone]
        self.pending = [m for m in self.pending if m[1] not in gone]

    def before(self, condition: str) -> list[tuple]:
        if condition == "every":
            return self.said
        if condition == "first":
            return self.pending
        return []


def episode_blocks(session: dict, step: dict, k: int, n: int, condition: str, restated: Restated, single: bool):
    """The episodes of one step, as blocks with one instruction each (every episode its own block if `single`); returns
    the blocks and the next episode index."""
    blocks: list[list[Episode]] = []
    for seed in parse_seeds(step["episodes"]):
        said = [m for m, _ in restated.before(condition)]
        instruction = " ".join(said + [step.get("instruction") or session.get("instruction") or ""]).strip()
        ep = Episode(n, seed, instruction, k, str(step.get("phase", "")), dict(step.get("expect") or {}),
                     step.get("complaint") or "", list(step.get("correction") or []), dict(step.get("dry_run") or {}))
        n += 1
        if condition == "first":
            restated.pending = []
        if blocks and not single and blocks[-1][-1].instruction == ep.instruction:
            blocks[-1].append(ep)
        else:
            blocks.append([ep])
    return blocks, n


def schedule(session: dict, condition: str = "profile", phases=None) -> list[tuple[str, Any]]:
    """The session as ("user", step), ("clock", step) and ("episodes", [Episode, ...]) items. A block of episodes runs as one
    unit (one launcher call), never across a message or a clock step; with distillation or votes every episode is its own
    block. `phases` keeps only the episode steps of those phases (messages and clock steps are always kept; episode indices
    stay those of the whole session)."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} not in {CONDITIONS}")
    single = condition == "profile" and bool(session.get("distil") or session.get("feedback"))
    restated = Restated()
    out = []
    n = 0
    for k, step in enumerate(session["steps"]):
        if "clock" in step:
            out.append(("clock", step))
            restated.clock(step["clock"])
        elif "user" in step:
            out.append(("user", step))
            restated.add(step)
        elif phases is not None and str(step.get("phase", "")) not in phases:
            n += len(parse_seeds(step["episodes"]))
        else:
            blocks, n = episode_blocks(session, step, k, n, condition, restated, single)
            out += [("episodes", b) for b in blocks]
    return out
