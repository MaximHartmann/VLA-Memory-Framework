"""One entry of the notebook, and the vocabulary around it: the strictness presets, how long an entry holds, the task's
objects a text names.

An entry is one rule in plain language with a scope: `plan` (which skills, in which order), `skill:<skill string>` (an
extra completion rule for that skill) or `planner` (how carefully progress is checked, as parameter overrides). Its
importance starts at 2 and moves with up- and down-votes (retired at 0, as in ExpeL); a superseded or retired entry keeps
its record with `valid_to` set (as in Zep). An entry holds until withdrawn, for the day it was made, or for the session it
was made in (`validity`; `expires` is the end of that day for both).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import re

STRICTNESS_PRESETS = {                        # named parameter bundles a writer can attach to a planner-scope entry
    "double_check": {"votes_to_advance": 2, "min_calls_per_skill": 2},   # "double-check before moving on"
    "quick": {"call_every": 2},                                          # "be quick", ask only every second chunk
    "default": {"votes_to_advance": 1, "min_calls_per_skill": 2, "call_every": 1},
}
VALIDITY = ("until_withdrawn", "day", "session")          # how long an entry holds, longest first
SPAN = {name: i for i, name in enumerate(reversed(VALIDITY))}   # session 0 < day 1 < until_withdrawn 2


def now_stamp() -> str:
    """The wall clock to the microsecond: ties in "newest first" must not happen within a second."""
    return _dt.datetime.now().isoformat(timespec="microseconds")


def day_end(t: _dt.datetime) -> str:
    """Midnight after t: the end of the day a 'day' or 'session' entry was made in (a session never outlasts its day)."""
    midnight = _dt.datetime.combine(t.date(), _dt.time()) + _dt.timedelta(days=1)
    return midnight.isoformat(timespec="microseconds")


def named(text: str, objects) -> list[str]:
    """The task objects a text names (whole words, any case, plural too), in the order of their first mention."""
    hits = []
    for obj in objects:
        match = re.search(rf"\b{re.escape(obj)}(?:e?s)?\b", text or "", re.I)
        if match:
            hits.append((match.start(), obj))
    return [obj for _, obj in sorted(hits)]


@dataclasses.dataclass
class Entry:
    id: str
    kind: str                      # "preference" | "correction"
    scope: str                     # "plan" | "skill:<skill string>" | "planner"
    text: str                      # one normalised rule in plain language
    condition: str = ""            # when it applies, e.g. "when the instruction does not name an order"; empty = always
    params: dict = dataclasses.field(default_factory=dict)   # planner-scope only: parameter overrides
    source: str = "user"           # "user" (stated) | "distilled" (from a failed episode)
    evidence: str = ""             # the quote, or "episode <id> step <n>"
    created: str = dataclasses.field(default_factory=now_stamp)
    valid_to: str | None = None    # set when superseded or retired; the record stays for audit
    importance: int = 2            # ExpeL-style vote count: +1 helped, -1 harmed, retired at 0
    applied: int = 0               # how often it was injected into a prompt
    supersedes: str | None = None  # id of the entry it replaced (or suspends, when it holds for a shorter time)
    validity: str = "until_withdrawn"   # how long it holds: until_withdrawn | day | session (VALIDITY)
    expires: str | None = None     # the end of its validity (day and session: the end of the day it was made); None = never

    @property
    def active(self) -> bool:
        return self.valid_to is None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
