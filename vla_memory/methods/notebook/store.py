"""The notebook itself: a small JSON profile of one user, with the operations the planner and the writers need.

Scope decides where a rule acts, `condition` says when, `importance` moves with up- and down-votes, a superseded rule keeps
its record with `valid_to` set, and at most `max_rules` active rules per scope are injected, ranked by importance and then
recency. Applicability is a scope match, never a similarity search. The entries themselves: entry.py.

Validity: the notebook judges every entry as of its last reload, which the planner does at every episode start, so an
entry never ends in the middle of an episode; ended entries stay in the file. The notebook keeps its own clock (wall clock +
`clock_offset_s`, which a session driver moves to simulate later days) and the start of the current session
(`session_start`). A shorter-lived entry that replaces a longer-lived one suspends it instead of retiring it: "red first
today" hides "blue first from now on" until the day is over.

Precedence: an instruction that names any of the task's objects decides itself which objects are handled and in which
order, so plan-scope rules that name objects are held back for it (plan_rules_for; the planner logs them).
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import re
import uuid
from typing import Any

from .entry import SPAN, STRICTNESS_PRESETS, VALIDITY, Entry, day_end, named

__all__ = ["ProfileStore", "Entry", "STRICTNESS_PRESETS", "VALIDITY", "named"]
CLAUSE = re.compile(r"(when|if|unless|only|as long as)\b", re.I)   # a condition that needs no "when:" in front


class ProfileStore:
    PLAN_HEADING = "User profile (apply unless the instruction says otherwise):"

    def __init__(self, path=None, max_rules: int = 3):
        self.path = pathlib.Path(path) if path else None
        self.max_rules = max_rules
        self.entries: list[Entry] = []
        self.clock_offset = 0.0                 # seconds added to the wall clock (a session driver simulates later days)
        self.session_start: str | None = None   # start of the current session; "session" entries made before it have ended
        self.as_of = self._stamp()              # entries are judged as of this moment (the last reload)
        self.reload()

    # ------------------------------------------------------------------ file
    def reload(self) -> "ProfileStore":
        """Re-read the file (the session driver may have written it between episodes) and judge validity as of now."""
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.entries = [Entry.from_dict(x) for x in d.get("entries", [])]
            self.clock_offset = float(d.get("clock_offset_s", 0.0))
            self.session_start = d.get("session_start")
        self.as_of = self._stamp()
        return self

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        d: dict[str, Any] = {"entries": [e.to_dict() for e in self.entries]}
        if self.clock_offset:
            d["clock_offset_s"] = self.clock_offset
        if self.session_start:
            d["session_start"] = self.session_start
        self.path.write_text(json.dumps(d, indent=1))

    # ------------------------------------------------------------------ the notebook's clock and sessions
    def now(self) -> _dt.datetime:
        return _dt.datetime.now() + _dt.timedelta(seconds=self.clock_offset)

    def _stamp(self) -> str:
        return self.now().isoformat(timespec="microseconds")

    def move_clock(self, days: int = 0, at: str | None = None) -> None:
        """Move the notebook's clock `days` days on, to the time `at` ("HH:MM") if given; a session driver simulates "the
        next day" with it (days=1, at="09:00") or keeps a long session inside one day (at="09:00" at the start of a fresh
        notebook: a clock moved back would make older entries look newer)."""
        t = self.now() + _dt.timedelta(days=days)
        if at:
            hour, minute = (int(x) for x in str(at).split(":"))
            t = t.replace(hour=hour, minute=minute, second=0, microsecond=0)
        self.clock_offset = (t - _dt.datetime.now()).total_seconds()
        self.save()
        self.as_of = self._stamp()

    def new_session(self) -> None:
        """A new session starts now: entries that held for the session before it end."""
        self.session_start = self._stamp()
        self.save()
        self.as_of = self._stamp()

    def ended(self, e: Entry) -> bool:
        """Has e's validity run out as of the last reload (for the planner: the start of the current episode)?"""
        expired = e.expires and e.expires <= self.as_of
        session_over = e.validity == "session" and self.session_start and e.created < self.session_start
        return bool(expired or session_over)

    # ------------------------------------------------------------------ writing
    def add(self, kind: str, scope: str, text: str, condition: str = "", params: dict | None = None, source: str = "user",
            evidence: str = "", supersedes: str | None = None, validity: str = "until_withdrawn") -> Entry:
        validity = validity if validity in VALIDITY else "until_withdrawn"
        now = self.now()
        expires = None if validity == "until_withdrawn" else day_end(now)
        created = now.isoformat(timespec="microseconds")
        e = Entry(id=uuid.uuid4().hex[:8], kind=kind, scope=scope, text=text.strip(), condition=condition.strip(),
                  params=dict(params or {}), source=source, evidence=evidence, created=created, supersedes=supersedes,
                  validity=validity, expires=expires)
        self._retire_replaced(supersedes, validity)
        self.entries.append(e)
        self.save()
        return e

    def _retire_replaced(self, supersedes: str | None, validity: str) -> None:
        """A new entry that holds at least as long as the one it replaces retires it, and also what that one suspended; a
        shorter-lived one only suspends it (it stays active and is hidden while the new one holds)."""
        old = self.get(supersedes) if supersedes else None
        while old is not None and old.active and SPAN[validity] >= SPAN.get(old.validity, 2):
            self.retire(old.id)
            old = self.get(old.supersedes) if old.supersedes else None

    def extend(self, entry_id: str, validity: str) -> None:
        """A repetition that holds longer extends the entry's validity ("only red today", then "from now on, only red")."""
        e = self.get(entry_id)
        if e is not None and validity in VALIDITY and SPAN[validity] > SPAN.get(e.validity, 2):
            e.validity = validity
            e.expires = None if validity == "until_withdrawn" else day_end(self.now())
            self.save()

    def retire(self, entry_id: str) -> None:
        for e in self.entries:
            if e.id == entry_id and e.active:
                e.valid_to = self._stamp()
        self.save()

    def upvote(self, entry_id: str) -> None:
        for e in self.entries:
            if e.id == entry_id and e.active:
                e.importance += 1
        self.save()

    def downvote(self, entry_id: str) -> None:
        for e in self.entries:
            if e.id == entry_id and e.active:
                e.importance -= 1
                if e.importance <= 0:
                    e.valid_to = self._stamp()
        self.save()

    def note_applied(self, entries) -> None:
        ids = {e.id for e in entries}
        for e in self.entries:
            if e.id in ids:
                e.applied += 1
        self.save()

    # ------------------------------------------------------------------ reading
    def active(self, scope: str | None = None) -> list[Entry]:
        """Entries in force: not retired, not ended, not suspended by a shorter-lived entry that replaced them."""
        live = [e for e in self.entries if e.active and not self.ended(e)]
        held = self._suspended(live)
        out = [e for e in live if e.id not in held and (scope is None or e.scope == scope)]
        return sorted(out, key=lambda e: (e.importance, e.created), reverse=True)

    def _suspended(self, live: list[Entry]) -> set:
        """The ids suspended by live entries: reached from a live entry through the entries it replaced, down to an active
        one."""
        held = set()
        for e in live:
            s = e.supersedes
            while s and s not in held:
                held.add(s)
                prev = self.get(s)
                s = prev.supersedes if prev is not None and not prev.active else None
        return held

    def get(self, entry_id: str) -> Entry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def plan_rules(self) -> list[Entry]:
        return self.active("plan")[: self.max_rules]

    def plan_rules_for(self, instruction: str, objects) -> tuple[list[Entry], list[dict]]:
        """The plan-scope rules for this instruction, and those held back because the instruction has precedence: an
        instruction that names any of the task's objects decides itself which objects are handled and in which order, so
        a rule that names objects too could only repeat or contradict it. Rules that name no object, and instructions that
        name none ("put each block back"), are not affected. Held back: [{id, reason}], for the planner's record."""
        said = named(instruction, objects)
        keep = []
        held = []
        for e in self.active("plan"):
            about = named(f"{e.text} {e.condition}", objects)
            if said and about:
                reason = f"the instruction names {', '.join(said)}; the rule names {', '.join(about)}"
                held.append({"id": e.id, "reason": reason})
            else:
                keep.append(e)
        return keep[: self.max_rules], held

    def rules_for_skill(self, skill: str) -> list[Entry]:
        return self.active(f"skill:{skill}")[: self.max_rules]

    def planner_rules(self) -> list[Entry]:
        return self.active("planner")[: self.max_rules]

    def planner_overrides(self) -> dict[str, Any]:
        """Parameter overrides from planner-scope entries; the highest-ranked entry wins on a conflict."""
        out: dict[str, Any] = {}
        for e in reversed(self.planner_rules()):
            out.update(e.params)
        return out

    @staticmethod
    def _line(e: Entry) -> str:
        """One rule as a prompt line; a condition that is already a clause ("unless ...") gets no "when:" in front."""
        c = e.condition
        if not c:
            return f"- {e.text}"
        if CLAUSE.match(c):
            return f"- {e.text} ({c})"
        return f"- {e.text} (when: {c})"

    def plan_text(self, rules: list[Entry] | None = None) -> str:
        rules = self.plan_rules() if rules is None else rules
        if not rules:
            return ""
        return self.PLAN_HEADING + "\n" + "\n".join(self._line(e) for e in rules)

    def rules_text(self, rules: list[Entry]) -> str:
        if not rules:
            return ""
        return "Additional rules from the user for this skill:\n" + "\n".join(self._line(e) for e in rules)

    def __len__(self) -> int:
        return len(self.active())
