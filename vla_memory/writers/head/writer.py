"""The event-head writer: the history policy writes its own history (--writer head [--consecutive N]).

No writer outside the policy looks at the robot or the cameras. The policy server (server.py) returns, with the actions
of every call, the verdict of the policy's event head: a logistic regression on the policy's own representation of this
call's frames and prompt, which says whether the current skill is complete:

    reply["event_head"] = {"score": float, "done": bool, "threshold": float, "ms": time of the head pass}

The writer keeps the plan (the task's skill order), the pointer (the lines written) and a counter of consecutive "done"
verdicts on the current skill. After `consecutive` of them it writes the current skill's line: the pointer moves on and the
counter starts again. A verdict arrives after its call's prompt was sent, so a line reaches the policy in the prompt of the
next call. At most max_lines lines are written: the last skill has no line, as in the policy's training data.
"""
from __future__ import annotations

from typing import Sequence

from ..base import Call, Writer

VERDICT = "event_head"        # the entry of the policy's reply that holds the head's verdict


class HeadWriter(Writer):
    """The plan pointer of the self-writing policy, moved by the event head's verdicts after the policy call. max_lines:
    the history method's --max-lines, else the task's history.max_lines."""

    name = "head"

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group("the event head (--writer head)")
        group.add_argument("--consecutive", type=int, default=2,
                           help="consecutive 'done' verdicts of the event head that write a line (default 2)")

    @classmethod
    def build(cls, args, task, extensions: Sequence = ()):
        return cls(task, consecutive=args.consecutive, max_lines=getattr(args, "max_lines", None))

    def __init__(self, task, consecutive: int = 2, max_lines: int | None = None, plan: Sequence[str] | None = None):
        self.task = task
        self.consecutive = int(consecutive)
        self.default_plan = list(plan) if plan else list(task.skills)
        if max_lines is None:
            max_lines = (task.history or {}).get("max_lines")
        if max_lines is None:
            max_lines = len(self.default_plan) - 1
        self.max_lines = int(max_lines)
        self.start_episode()

    def start_episode(self, instruction: str = "", episode_id=None) -> dict:
        self.plan = list(self.default_plan)
        self.idx = 0                  # the lines written, and so the index of the current skill
        self.counter = 0              # consecutive "done" verdicts on the current skill
        self.done_all = False         # never true: the last skill has no line
        self.unshown = False          # a line was written after the last reply and is not in a prompt yet
        return {"plan": list(self.plan), "source": "task order (event-head writer)", "ms": 0.0}

    @property
    def writing(self) -> bool:
        """Whether another line can still be written."""
        return self.idx < min(self.max_lines, len(self.plan) - 1)

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """A new policy call: a line written after the last reply goes into this call's prompt. Returns the call's log
        fields."""
        fields = {"advanced": self.unshown, "skill_index": self.idx}
        self.unshown = False
        return fields

    def after_call(self, reply: dict) -> dict:
        """The verdict in the policy's reply; its record goes into the proxy's log as event_head."""
        return {VERDICT: self.observe_reply(reply)}

    def observe_verdict(self, done: bool) -> str | None:
        """The head's verdict on the current skill after one policy call. Returns the skill whose line it writes now, or
        None."""
        if not self.writing:
            return None
        if done:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter < self.consecutive:
            return None
        skill = self.plan[self.idx]
        self.idx += 1
        self.counter = 0
        self.unshown = True
        return skill

    def observe_reply(self, reply: dict) -> dict:
        """Read the verdict from the policy's reply and act on it. Returns the record for the proxy's log."""
        verdict = reply.get(VERDICT)
        if not isinstance(verdict, dict) or "done" not in verdict:
            raise RuntimeError("--writer head: the policy's reply has no event_head verdict; serve the policy with "
                               "python -m vla_memory.writers.head.server")
        done = bool(verdict["done"])
        wrote = self.observe_verdict(done)
        return {"score": float(verdict["score"]), "done": done, "threshold": float(verdict["threshold"]),
                "ms": float(verdict.get("ms", float("nan"))), "counter": self.counter, "wrote": wrote}
