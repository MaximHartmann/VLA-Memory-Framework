"""Sessions: a scripted sequence of user messages and episodes, the bench of the notebook.

A session file (YAML or JSON) lists steps in order; a step is a user message, a block of episodes or a clock step:

    name: order_conflict
    task: null                    # task file (default: the packaged reference task)
    task_override: {criteria: {"pick up": "..."}}   # optional, merged into the task one level deep (session_task)
    writer: gold                  # optional: the writer a runner uses unless told otherwise (gold | vlm)
    store: false                  # optional runner hint: planner without the exemplar store
    instruction: "..."            # default instruction of the episodes
    distil: false                 # write a correction after a failed episode: false | true (every failure) |
                                  # complaint (only failures the user remarks on)
    feedback: false               # vote the entries applied in an episode up (success) or down (failure): false |
                                  # true (corrections only) | a list of kinds, e.g. [correction, preference]
    steps:
      - clock: {at: "09:00"}      # optional: the notebook's clock to 09:00 of the current day (keeps a session in one day)
      - user: "From now on, please handle the blue block first."
        expect: [{label: order, kind: preference, scope: plan, text: "...", keywords: [blue, first]}]   # annotated rules
                                  # (validity: until_withdrawn | day | session; default until_withdrawn)
      - episodes: 1-20            # seeds: "1-20", "1,3,5", a list or one number
        phase: after              # optional label for the analysis
        instruction: "..."        # optional, replaces the default
        expect: {plan_prefix: ["pick up the blue block"], params: {votes_to_advance: 2}}   # what following means here;
                                  # also plan: [...] (the exact plan)
        complaint: "..."          # the user's remark after the first failed episode of this step (read by distil)
        correction: [{...}]       # annotated correction for a failed episode (gold writer)
        dry_run: {fail: true}     # outcome in a dry run; also fail_unless_rule / fail_if_rule: <skill>
      - clock: next_day           # 09:00 of the next day, which starts a new session; also new_session (same day) or
                                  # {days: N, at: "HH:MM", session: true}

Annotations follow gold.GoldWriter; `keywords` (all must occur in the rule text) and `expect` serve the scoring.
Conditions, from the same file: profile (the notebook is written and read), none (no memory), first (each annotated message
is put in front of the next episode's instruction only), every (all annotated messages so far in front of every
instruction: the restated, "perfect memory" upper bound). A message annotated `expect: []` states no rule and is never
restated; a message whose annotated rules hold only for the day or the session is no longer restated after a clock step ends
that day or session. In the profile condition a clock step moves the notebook's clock (the planner re-reads it at every
episode start, so the next episodes judge validity on the new day).

run_session() writes one JSON line per message, clock step, episode, distillation and vote, then a summary
(session_score.py). An episode record holds the entries active at its start and those the planner applied (plan rules, skill
rules per skill with the number of queries that carried them, planner-scope entries and their parameters), the plan, the
outcome and the following check.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

from ...task import TaskSpec, read_task_file
from .session_schedule import CLOCK_STEPS, CONDITIONS, Episode, clock_spec, parse_seeds, schedule
from .session_score import applied_entries, applied_ids, brief, check, score_writer, summarise

__all__ = ["CONDITIONS", "CLOCK_STEPS", "Episode", "Outcome", "EpisodeRunner", "load_session", "session_task",
           "parse_seeds", "clock_spec", "apply_clock", "schedule", "run_session", "applied_entries", "applied_ids", "check",
           "summarise", "score_writer"]


@dataclasses.dataclass
class Outcome:
    """How one episode ended, as a runner reports it."""

    success: bool
    reason: str = ""
    records: list = dataclasses.field(default_factory=list)   # the planner's records of this episode (plan + monitor)
    info: dict = dataclasses.field(default_factory=dict)      # whatever else the runner measured


class EpisodeRunner:
    """Runs a block of episodes in order; the planner reads the notebook at `profile` (None: no notebook)."""

    def run(self, episodes: list[Episode], profile: str | None) -> list[Outcome]:
        """One outcome per episode, in order."""
        raise NotImplementedError


# ---------------------------------------------------------------------- the session file
def load_session(path) -> dict:
    """The session file as a dict: YAML or JSON, by file suffix."""
    path = pathlib.Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def session_task(session: dict):
    """The session's task: its task file (default: the packaged reference task) with `task_override` merged in, one level
    deep (criteria: {"pick up": ...} replaces that verb's rule only). Returns the TaskSpec and the merged dict, which a
    runner can write to a file for the planner."""
    data = read_task_file(session.get("task") or TaskSpec.packaged_path("red_blue_blocks"))
    for key, value in (session.get("task_override") or {}).items():
        data[key] = {**data[key], **value} if isinstance(value, dict) and isinstance(data.get(key), dict) else value
    return TaskSpec.from_dict(data), data


def apply_clock(store, value) -> dict:
    """Carry out a clock step on the notebook; returns the new time and the entries it ended or brought back."""
    clock = clock_spec(value)
    before = [entry.id for entry in store.reload().active()]
    if clock["days"] or clock.get("at"):
        store.move_clock(clock["days"], clock.get("at"))
    if clock["session"]:
        store.new_session()
    after = [entry.id for entry in store.reload().active()]
    return {"now": store.as_of, "session_start": store.session_start,
            "ended": [entry_id for entry_id in before if entry_id not in after],
            "resumed": [entry_id for entry_id in after if entry_id not in before]}


# ---------------------------------------------------------------------- running a session
class SessionRun:
    """One run of a session in one condition: the records it writes, and the writer and the notebook in the profile
    condition."""

    def __init__(self, session: dict, runner: EpisodeRunner, writer, store, condition: str, log):
        self.session = session
        self.runner = runner
        self.writer = writer
        self.store = store
        self.uses_notebook = condition == "profile"
        if self.uses_notebook and (store is None or store.path is None or writer is None):
            raise ValueError("the profile condition needs a writer and a file-backed ProfileStore (the planner re-reads the "
                             "file)")
        self.distil = session.get("distil", False)
        self.vote = session.get("feedback", False)
        self.vote_kinds = tuple(self.vote) if isinstance(self.vote, (list, tuple)) else ("correction",)
        self.records: list[dict] = []
        self.complained: set[int] = set()
        self.file = open(log, "w") if log else None
        self.emit({"event": "session", "name": session.get("name"), "condition": condition, "distil": self.distil,
                   "feedback": self.vote, "profile": str(store.path) if self.uses_notebook else None})

    def emit(self, record: dict) -> None:
        """Keep the record and append it to the log file."""
        self.records.append(record)
        if self.file:
            self.file.write(json.dumps(record, default=str) + "\n")
            self.file.flush()

    def clock(self, step: dict) -> None:
        """A clock step: moved on the notebook in the profile condition, recorded in every condition."""
        record = {"event": "clock", "clock": step["clock"]}
        if self.uses_notebook:
            record.update(apply_clock(self.store, step["clock"]))
        self.emit(record)

    def message(self, step: dict) -> None:
        """A user message: written into the notebook in the profile condition, recorded in every condition."""
        record = {"event": "message", "text": step["user"], "expect": step.get("expect")}
        if self.uses_notebook:
            record.update(self.writer.from_message(step["user"], gold=step.get("expect")))
        self.emit(record)

    def episodes(self, block: list[Episode]) -> None:
        """Run one block; per episode its record, then the votes and the distillation of the profile condition."""
        active, planner_ids = [], []
        if self.uses_notebook:
            self.store.reload()
            active = [brief(entry) for entry in self.store.active()]
            planner_ids = [entry.id for entry in self.store.planner_rules()]
        outcomes = self.runner.run(block, str(self.store.path) if self.uses_notebook else None)
        for episode, outcome in zip(block, outcomes):
            applied = applied_entries(outcome.records, planner_ids)
            self.emit(self.episode_record(episode, outcome, active, applied))
            if self.uses_notebook:
                self.learn(episode, outcome, applied)

    @staticmethod
    def episode_record(episode: Episode, outcome: Outcome, active: list, applied: dict) -> dict:
        """The record of one episode: the plan, the entries active and applied, the held-back rules and the check."""
        plan_record = next((record for record in outcome.records if record.get("event") == "plan"), {})
        held = [item["id"] for item in plan_record.get("profile_plan_held") or []]
        following = check(episode.expect, plan_record.get("plan") or [], applied["params"])
        return {"event": "episode", "index": episode.index, "seed": episode.seed, "phase": episode.phase,
                "instruction": episode.instruction, "success": outcome.success, "reason": outcome.reason,
                "plan": plan_record.get("plan"), "plan_source": plan_record.get("source"), "active": active,
                "applied": applied, "held": held, "check": following, **outcome.info}

    def learn(self, episode: Episode, outcome: Outcome, applied: dict) -> None:
        """After one episode of the profile condition: the votes, and a correction after a failure."""
        if self.vote:
            votes = self.writer.feedback(applied_ids(applied), outcome.success, kinds=self.vote_kinds)
            if any(votes.values()):
                self.emit({"event": "feedback", "episode": episode.index, "success": outcome.success, **votes})
        remark = episode.complaint if episode.complaint and episode.step not in self.complained else ""
        if not outcome.success and (self.distil is True or (self.distil == "complaint" and remark)):
            if remark:
                self.complained.add(episode.step)
            done = self.writer.from_failure(outcome.records, outcome.reason, remark, gold=episode.correction)
            self.emit({"event": "distil", "episode": episode.index, "complaint": remark, "expect": episode.correction, **done})

    def finish(self) -> dict:
        """The summary record, written last; the log file is closed."""
        summary = summarise(self.records, self.store if self.uses_notebook else None)
        self.emit(summary)
        if self.file:
            self.file.close()
        return summary


def run_session(session: dict, runner: EpisodeRunner, writer=None, store=None, condition: str = "profile", log=None,
                phases=None) -> dict:
    """Run the steps in order and return the summary. writer and store (the notebook) are used in the profile condition
    only; log is a JSONL file for the records; phases restricts the episodes (see schedule)."""
    run = SessionRun(session, runner, writer, store, condition, log)
    for kind, item in schedule(session, condition, phases):
        if kind == "clock":
            run.clock(item)
        elif kind == "user":
            run.message(item)
        else:
            run.episodes(item)
    return run.finish()
