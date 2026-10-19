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

import datetime
import json
import pathlib
import re
import uuid
from typing import Any

from .entry import SPAN, STRICTNESS_PRESETS, VALIDITY, Entry, day_end, named

__all__ = ["ProfileStore", "Entry", "STRICTNESS_PRESETS", "VALIDITY", "named"]
CLAUSE = re.compile(r"(when|if|unless|only|as long as)\b", re.I)   # a condition that needs no "when:" in front


class ProfileStore:
    """The notebook file and its entries (module docstring); every change is saved at once. path: the JSON file (None: in
    memory only); max_rules: the most active rules injected per scope."""

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
            data = json.loads(self.path.read_text())
            self.entries = [Entry.from_dict(item) for item in data.get("entries", [])]
            self.clock_offset = float(data.get("clock_offset_s", 0.0))
            self.session_start = data.get("session_start")
        self.as_of = self._stamp()
        return self

    def save(self) -> None:
        """Write the entries, the clock offset and the session start to the file (nothing without a path)."""
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"entries": [entry.to_dict() for entry in self.entries]}
        if self.clock_offset:
            data["clock_offset_s"] = self.clock_offset
        if self.session_start:
            data["session_start"] = self.session_start
        self.path.write_text(json.dumps(data, indent=1))

    # ------------------------------------------------------------------ the notebook's clock and sessions
    def now(self) -> datetime.datetime:
        """The notebook's time: the wall clock plus the clock offset."""
        return datetime.datetime.now() + datetime.timedelta(seconds=self.clock_offset)

    def _stamp(self) -> str:
        """The notebook's time as the entries record it."""
        return self.now().isoformat(timespec="microseconds")

    def move_clock(self, days: int = 0, at: str | None = None) -> None:
        """Move the notebook's clock `days` days on, to the time `at` ("HH:MM") if given; a session driver simulates "the
        next day" with it (days=1, at="09:00") or keeps a long session inside one day (at="09:00" at the start of a fresh
        notebook: a clock moved back would make older entries look newer)."""
        moment = self.now() + datetime.timedelta(days=days)
        if at:
            hour, minute = (int(part) for part in str(at).split(":"))
            moment = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
        self.clock_offset = (moment - datetime.datetime.now()).total_seconds()
        self.save()
        self.as_of = self._stamp()

    def new_session(self) -> None:
        """A new session starts now: entries that held for the session before it end."""
        self.session_start = self._stamp()
        self.save()
        self.as_of = self._stamp()

    def ended(self, entry: Entry) -> bool:
        """Has the entry's validity run out as of the last reload (for the planner: the start of the current episode)?"""
        expired = entry.expires and entry.expires <= self.as_of
        session_over = entry.validity == "session" and self.session_start and entry.created < self.session_start
        return bool(expired or session_over)

    # ------------------------------------------------------------------ writing
    def add(self, kind: str, scope: str, text: str, condition: str = "", params: dict | None = None, source: str = "user",
            evidence: str = "", supersedes: str | None = None, validity: str = "until_withdrawn") -> Entry:
        """A new entry, made now; with `supersedes`, the replaced entry is retired or suspended (_retire_replaced)."""
        validity = validity if validity in VALIDITY else "until_withdrawn"
        now = self.now()
        expires = None if validity == "until_withdrawn" else day_end(now)
        created = now.isoformat(timespec="microseconds")
        entry = Entry(id=uuid.uuid4().hex[:8], kind=kind, scope=scope, text=text.strip(), condition=condition.strip(),
                      params=dict(params or {}), source=source, evidence=evidence, created=created, supersedes=supersedes,
                      validity=validity, expires=expires)
        self._retire_replaced(supersedes, validity)
        self.entries.append(entry)
        self.save()
        return entry

    def _retire_replaced(self, supersedes: str | None, validity: str) -> None:
        """A new entry that holds at least as long as the one it replaces retires it, and also what that one suspended; a
        shorter-lived one only suspends it (it stays active and is hidden while the new one holds)."""
        replaced = self.get(supersedes) if supersedes else None
        while replaced is not None and replaced.active and SPAN[validity] >= SPAN.get(replaced.validity, 2):
            self.retire(replaced.id)
            replaced = self.get(replaced.supersedes) if replaced.supersedes else None

    def extend(self, entry_id: str, validity: str) -> None:
        """A repetition that holds longer extends the entry's validity ("only red today", then "from now on, only red")."""
        entry = self.get(entry_id)
        if entry is not None and validity in VALIDITY and SPAN[validity] > SPAN.get(entry.validity, 2):
            entry.validity = validity
            entry.expires = None if validity == "until_withdrawn" else day_end(self.now())
            self.save()

    def retire(self, entry_id: str) -> None:
        """End the entry now; its record stays."""
        for entry in self.entries:
            if entry.id == entry_id and entry.active:
                entry.valid_to = self._stamp()
        self.save()

    def upvote(self, entry_id: str) -> None:
        """The entry helped: importance + 1."""
        for entry in self.entries:
            if entry.id == entry_id and entry.active:
                entry.importance += 1
        self.save()

    def downvote(self, entry_id: str) -> None:
        """The entry harmed: importance - 1, retired at 0."""
        for entry in self.entries:
            if entry.id == entry_id and entry.active:
                entry.importance -= 1
                if entry.importance <= 0:
                    entry.valid_to = self._stamp()
        self.save()

    def note_applied(self, entries) -> None:
        """Count one more prompt the entries were injected into."""
        ids = {entry.id for entry in entries}
        for entry in self.entries:
            if entry.id in ids:
                entry.applied += 1
        self.save()

    # ------------------------------------------------------------------ reading
    def active(self, scope: str | None = None) -> list[Entry]:
        """Entries in force: not retired, not ended, not suspended by a shorter-lived entry that replaced them."""
        live = [entry for entry in self.entries if entry.active and not self.ended(entry)]
        held = self._suspended(live)
        in_force = [entry for entry in live if entry.id not in held and (scope is None or entry.scope == scope)]
        return sorted(in_force, key=lambda entry: (entry.importance, entry.created), reverse=True)

    def _suspended(self, live: list[Entry]) -> set:
        """The ids suspended by live entries: reached from a live entry through the entries it replaced, down to an active
        one."""
        held = set()
        for entry in live:
            replaced_id = entry.supersedes
            while replaced_id and replaced_id not in held:
                held.add(replaced_id)
                replaced = self.get(replaced_id)
                replaced_id = replaced.supersedes if replaced is not None and not replaced.active else None
        return held

    def get(self, entry_id: str) -> Entry | None:
        """The entry with this id, or None."""
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def plan_rules(self) -> list[Entry]:
        """The plan-scope rules injected into the plan request, at most max_rules."""
        return self.active("plan")[: self.max_rules]

    def plan_rules_for(self, instruction: str, objects) -> tuple[list[Entry], list[dict]]:
        """The plan-scope rules for this instruction, and those held back because the instruction has precedence: an
        instruction that names any of the task's objects decides itself which objects are handled and in which order, so
        a rule that names objects too could only repeat or contradict it. Rules that name no object, and instructions that
        name none ("put each block back"), are not affected. Held back: [{id, reason}], for the planner's record."""
        said = named(instruction, objects)
        kept = []
        held = []
        for entry in self.active("plan"):
            about = named(f"{entry.text} {entry.condition}", objects)
            if said and about:
                reason = f"the instruction names {', '.join(said)}; the rule names {', '.join(about)}"
                held.append({"id": entry.id, "reason": reason})
            else:
                kept.append(entry)
        return kept[: self.max_rules], held

    def rules_for_skill(self, skill: str) -> list[Entry]:
        """The rules of scope skill:<skill>, at most max_rules."""
        return self.active(f"skill:{skill}")[: self.max_rules]

    def planner_rules(self) -> list[Entry]:
        """The planner-scope rules, at most max_rules."""
        return self.active("planner")[: self.max_rules]

    def planner_overrides(self) -> dict[str, Any]:
        """Parameter overrides from planner-scope entries; the highest-ranked entry wins on a conflict."""
        overrides: dict[str, Any] = {}
        for entry in reversed(self.planner_rules()):
            overrides.update(entry.params)
        return overrides

    @staticmethod
    def _rule_line(entry: Entry) -> str:
        """One rule as a prompt line; a condition that is already a clause ("unless ...") gets no "when:" in front."""
        condition = entry.condition
        if not condition:
            return f"- {entry.text}"
        if CLAUSE.match(condition):
            return f"- {entry.text} ({condition})"
        return f"- {entry.text} (when: {condition})"

    def plan_text(self, rules: list[Entry] | None = None) -> str:
        """The plan rules (default: plan_rules()) as the block the plan request shows; "" without rules."""
        rules = self.plan_rules() if rules is None else rules
        if not rules:
            return ""
        return self.PLAN_HEADING + "\n" + "\n".join(self._rule_line(entry) for entry in rules)

    def rules_text(self, rules: list[Entry]) -> str:
        """Skill rules as the block shown after the completion criterion; "" without rules."""
        if not rules:
            return ""
        return "Additional rules from the user for this skill:\n" + "\n".join(self._rule_line(entry) for entry in rules)

    def __len__(self) -> int:
        return len(self.active())
