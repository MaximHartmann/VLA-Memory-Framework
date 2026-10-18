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

from .session_schedule import CLOCK_STEPS, CONDITIONS, Episode, clock_spec, parse_seeds, schedule
from .session_score import applied_entries, applied_ids, brief, check, score_writer, summarise

__all__ = ["CONDITIONS", "CLOCK_STEPS", "Episode", "Outcome", "EpisodeRunner", "load_session", "session_task",
           "parse_seeds", "clock_spec", "apply_clock", "schedule", "run_session", "applied_entries", "applied_ids", "check",
           "summarise", "score_writer"]


@dataclasses.dataclass
class Outcome:
    success: bool
    reason: str = ""
    records: list = dataclasses.field(default_factory=list)   # the planner's records of this episode (plan + monitor)
    info: dict = dataclasses.field(default_factory=dict)      # whatever else the runner measured


class EpisodeRunner:
    """Runs a block of episodes in order; the planner reads the notebook at `profile` (None: no notebook)."""

    def run(self, episodes: list[Episode], profile: str | None) -> list[Outcome]:
        raise NotImplementedError


# ---------------------------------------------------------------------- the session file
def load_session(path) -> dict:
    p = pathlib.Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def session_task(session: dict):
    """The session's task: its task file (default: the packaged reference task) with `task_override` merged in, one level
    deep (criteria: {"pick up": ...} replaces that verb's rule only). Returns the TaskSpec and the merged dict, which a
    runner can write to a file for the planner."""
    from ...task import _HERE, TaskSpec, _load_file
    d = _load_file(session.get("task") or _HERE / "tasks" / "red_blue_blocks.yaml")
    for k, v in (session.get("task_override") or {}).items():
        d[k] = {**d[k], **v} if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    known = {f.name for f in dataclasses.fields(TaskSpec)}
    return TaskSpec(**{k: v for k, v in d.items() if k in known}), d



def apply_clock(store, v) -> dict:
    """Carry out a clock step on the notebook; returns the new time and the entries it ended or brought back."""
    c = clock_spec(v)
    before = [e.id for e in store.reload().active()]
    if c["days"] or c.get("at"):
        store.move_clock(c["days"], c.get("at"))
    if c["session"]:
        store.new_session()
    after = [e.id for e in store.reload().active()]
    return {"now": store.as_of, "session_start": store.session_start, "ended": [i for i in before if i not in after],
            "resumed": [i for i in after if i not in before]}


# ---------------------------------------------------------------------- running a session
class SessionRun:
    """One run of a session in one condition: the records it writes, and the writer and the notebook in the profile
    condition."""

    def __init__(self, session: dict, runner: EpisodeRunner, writer, store, condition: str, log):
        self.session = session
        self.runner = runner
        self.writer = writer
        self.store = store
        self.use = condition == "profile"
        if self.use and (store is None or store.path is None or writer is None):
            raise ValueError("the profile condition needs a writer and a file-backed ProfileStore (the planner re-reads the "
                             "file)")
        self.distil = session.get("distil", False)
        self.vote = session.get("feedback", False)
        self.vote_kinds = tuple(self.vote) if isinstance(self.vote, (list, tuple)) else ("correction",)
        self.records: list[dict] = []
        self.complained: set[int] = set()
        self.file = open(log, "w") if log else None
        self.emit({"event": "session", "name": session.get("name"), "condition": condition, "distil": self.distil,
                   "feedback": self.vote, "profile": str(store.path) if self.use else None})

    def emit(self, record: dict) -> None:
        self.records.append(record)
        if self.file:
            self.file.write(json.dumps(record, default=str) + "\n")
            self.file.flush()

    def clock(self, step: dict) -> None:
        record = {"event": "clock", "clock": step["clock"]}
        if self.use:
            record.update(apply_clock(self.store, step["clock"]))
        self.emit(record)

    def message(self, step: dict) -> None:
        record = {"event": "message", "text": step["user"], "expect": step.get("expect")}
        if self.use:
            record.update(self.writer.from_message(step["user"], gold=step.get("expect")))
        self.emit(record)

    def episodes(self, block: list[Episode]) -> None:
        """Run one block; per episode its record, then the votes and the distillation of the profile condition."""
        active, planner_ids = [], []
        if self.use:
            self.store.reload()
            active = [brief(e) for e in self.store.active()]
            planner_ids = [e.id for e in self.store.planner_rules()]
        outcomes = self.runner.run(block, str(self.store.path) if self.use else None)
        for ep, oc in zip(block, outcomes):
            applied = applied_entries(oc.records, planner_ids)
            self.emit(self.episode_record(ep, oc, active, applied))
            if self.use:
                self.learn(ep, oc, applied)

    @staticmethod
    def episode_record(ep: Episode, oc: Outcome, active: list, applied: dict) -> dict:
        plan_rec = next((r for r in oc.records if r.get("event") == "plan"), {})
        held = [h["id"] for h in plan_rec.get("profile_plan_held") or []]
        following = check(ep.expect, plan_rec.get("plan") or [], applied["params"])
        return {"event": "episode", "index": ep.index, "seed": ep.seed, "phase": ep.phase, "instruction": ep.instruction,
                "success": oc.success, "reason": oc.reason, "plan": plan_rec.get("plan"),
                "plan_source": plan_rec.get("source"), "active": active, "applied": applied, "held": held,
                "check": following, **oc.info}

    def learn(self, ep: Episode, oc: Outcome, applied: dict) -> None:
        """After one episode of the profile condition: the votes, and a correction after a failure."""
        if self.vote:
            fb = self.writer.feedback(applied_ids(applied), oc.success, kinds=self.vote_kinds)
            if any(fb.values()):
                self.emit({"event": "feedback", "episode": ep.index, "success": oc.success, **fb})
        remark = ep.complaint if ep.complaint and ep.step not in self.complained else ""
        if not oc.success and (self.distil is True or (self.distil == "complaint" and remark)):
            if remark:
                self.complained.add(ep.step)
            done = self.writer.from_failure(oc.records, oc.reason, remark, gold=ep.correction)
            self.emit({"event": "distil", "episode": ep.index, "complaint": remark, "expect": ep.correction, **done})

    def finish(self) -> dict:
        summary = summarise(self.records, self.store if self.use else None)
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
