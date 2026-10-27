"""The event-head writer: the policy itself says when a skill is done (--writer head --head-dir DIR [--consecutive N]).

No writer outside the policy looks at the robot or the cameras. The policy server hands out, with the actions of every call,
summaries of the policy's own state at that call (reply["policy_state"], vla_memory/taps/reply.py). The event head, a
logistic regression fitted on one of these summaries (file.py; the fit: training/fit_event_head.py), runs here in the proxy
and says whether the current skill is complete.

The writer keeps the plan (the task's skill order; of the episode's objects only when the environment declares them), the
pointer (the skills completed) and a counter of consecutive "done" verdicts on the current skill. After `consecutive` of them
the current skill is complete: the pointer moves on and the counter starts again. A verdict arrives after its call's input
was sent, so the method shows it from the next call on. At most max_lines skills are completed this way: the last skill has
no line, as in the policy's training data.
"""
from __future__ import annotations

import time
from typing import Sequence

from ..base import Call, Writer
from .file import EventHead, load_head
from ...taps.reply import read_state

VERDICT = "event_head"        # the key of the head's record in the proxy's log


class HeadWriter(Writer):
    """The plan pointer of the self-writing policy, moved by the event head's verdicts after the policy call. consecutive:
    the verdicts in a row that complete a skill (default: the head's N); max_lines: the method's own limit (the history
    method's --max-lines, the keyframe method's --keyframe-max-events), else the task's history.max_lines."""

    name = "head"

    @classmethod
    def add_arguments(cls, parser):
        """--head-dir and --consecutive."""
        group = parser.add_argument_group("the event head (--writer head)")
        group.add_argument("--head-dir", default=None,
                           help="the head's directory (head.npz, head.json, head.sha256), from training/fit_event_head.py")
        group.add_argument("--consecutive", type=int, default=None,
                           help="consecutive 'done' verdicts that complete a skill (default: the head's N, 2)")

    @classmethod
    def build(cls, args, task, extensions: Sequence = ()):
        """The writer with the head of --head-dir, --consecutive and the chosen method's limit (--max-lines or
        --keyframe-max-events), if it has one."""
        if not args.head_dir:
            raise ValueError("--writer head needs --head-dir: a head fitted for this policy (training/fit_event_head.py)")
        limit = getattr(args, "max_lines", None)
        if limit is None:
            limit = getattr(args, "keyframe_max_events", None)
        return cls(task, load_head(args.head_dir), consecutive=args.consecutive, max_lines=limit)

    def __init__(self, task, head: EventHead, consecutive: int | None = None, max_lines: int | None = None,
                 plan: Sequence[str] | None = None):
        self.task = task
        self.head = head
        self.consecutive = int(consecutive if consecutive is not None else head.consecutive)
        self.default_plan = list(plan) if plan else list(task.skills)
        if max_lines is None:
            max_lines = (task.history or {}).get("max_lines")
        if max_lines is None:
            max_lines = len(self.default_plan) - 1
        self.max_lines = int(max_lines)
        self.start_episode()

    def start_episode(self, instruction: str = "", episode_id=None, objects=None) -> dict:
        """Back to the first skill of the plan (the task order of the episode's objects, when declared); the plan record names
        the task order as the plan's source."""
        self.plan = self.task.restrict_plan(self.default_plan, objects)
        self.skill_index = 0          # the skills completed, and so the index of the current skill
        self.counter = 0              # consecutive "done" verdicts on the current skill
        self.done_all = False         # never true: the last skill has no line
        self.unshown = False          # a skill was completed after the last reply and is not shown yet
        return {"plan": list(self.plan), "source": "task order (event-head writer)", "ms": 0.0}

    @property
    def writing(self) -> bool:
        """Whether another skill can still be completed."""
        return self.skill_index < min(self.max_lines, len(self.plan) - 1)

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """A new policy call: a skill completed after the last reply is shown from this call on. Returns the log fields."""
        fields = {"advanced": self.unshown, "skill_index": self.skill_index}
        self.unshown = False
        return fields

    def after_call(self, reply: dict) -> dict:
        """The head's verdict on the policy's state in the reply; its record goes into the proxy's log as event_head."""
        return {VERDICT: self.observe_reply(reply)}

    def observe_verdict(self, done: bool) -> str | None:
        """The head's verdict on the current skill after one policy call. Returns the skill it completes now, or None."""
        if not self.writing:
            return None
        self.counter = self.counter + 1 if done else 0
        if self.counter < self.consecutive:
            return None
        skill = self.plan[self.skill_index]
        self.skill_index += 1
        self.counter = 0
        self.unshown = True
        return skill

    def observe_reply(self, reply: dict) -> dict:
        """Score the summary the head was fitted on and act on the verdict. Returns the record for the proxy's log."""
        try:
            names, values = read_state(reply)
        except ValueError as error:
            raise RuntimeError(f"--writer head: {error}") from None
        if self.head.state not in names:
            raise RuntimeError(f"--writer head: the head reads {self.head.state!r}, the policy server hands out {names}")
        started = time.perf_counter()
        verdict = self.head.verdict(values[names.index(self.head.state)])
        milliseconds = 1000 * (time.perf_counter() - started)
        wrote = self.observe_verdict(verdict["done"])
        return {"score": verdict["score"], "done": verdict["done"], "threshold": verdict["threshold"], "ms": milliseconds,
                "counter": self.counter, "wrote": wrote}

    def metadata(self) -> dict:
        """The head the writer runs: the summary it reads, its threshold, N and its files' digests."""
        return {VERDICT: {"state": self.head.state, "threshold": self.head.threshold, "N": self.consecutive,
                          "sha256": self.head.meta.get("sha256")}}
