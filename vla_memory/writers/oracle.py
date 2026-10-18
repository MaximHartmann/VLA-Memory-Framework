"""Two writers without a model: the simulator's ground truth (the reference) and none at all.

    oracle   the environment's ground-truth tracker sends the skill it considers current as memory/oracle_prompt; the
             pointer is that skill's index in the task's skill order, set before the policy call. An oracle prompt that is
             not one of the task's skills is forwarded to the policy as it is (passthrough), and the pointer stays.
             Simulation only: a real robot has no ground truth.
    off      no writer: the policy gets the environment's instruction unchanged and no step is ever completed
"""
from __future__ import annotations

import logging
from typing import Sequence

from .base import Call, Writer

log = logging.getLogger("vla_memory.writers")


def text_of(value):
    """A string that may have arrived as bytes over the wire."""
    return value.decode() if isinstance(value, bytes) else value


class OracleWriter(Writer):
    name = "oracle"
    monitor = None                 # the extensions see no monitor: the environment decides

    def __init__(self, task):
        self.task = task
        self.plan = list(task.skills)
        self._warned = False
        self.start_episode("", None)

    def start_episode(self, instruction: str, episode_id) -> None:
        self.idx = 0
        self.done_all = False      # the tracker never reports the task as finished
        self.passthrough = None
        return None

    @property
    def current_skill(self) -> str:
        return self.passthrough if self.passthrough is not None else self.plan[self.idx]

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """The skill the environment's tracker considers current (the instruction when it sends none)."""
        if "oracle_prompt" not in call.ctrl and not self._warned:
            log.warning("oracle writer without memory/oracle_prompt: forwarding the instruction")
            self._warned = True
        reported = text_of(call.ctrl.get("oracle_prompt", call.instruction))
        if reported in self.plan:
            self.idx = self.plan.index(reported)
            self.passthrough = None
        else:
            self.passthrough = reported
        return {}


class OffWriter(Writer):
    name = "off"
    drives_prompt = False
    monitor = None

    def __init__(self, task):
        self.task = task
        self.plan: list[str] = []
        self.idx = 0
        self.done_all = False

    @property
    def current_skill(self) -> None:
        return None

    def completed(self) -> list[str]:
        return []
