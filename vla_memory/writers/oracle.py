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
from ..transport import text_of

log = logging.getLogger("vla_memory.writers")


class OracleWriter(Writer):
    """The pointer follows memory/oracle_prompt, the skill the environment's tracker considers current."""

    name = "oracle"
    monitor = None                 # the extensions see no monitor: the environment decides

    def __init__(self, task):
        self.task = task
        self.plan = list(task.skills)
        self._warned = False
        self.start_episode("", None)

    def start_episode(self, instruction: str, episode_id) -> None:
        """Back to the first skill; the oracle has no plan record."""
        self.skill_index = 0
        self.done_all = False      # the tracker never reports the task as finished
        self.passthrough = None
        return None

    @property
    def current_skill(self) -> str:
        """The forwarded environment prompt when it is not a skill, else the skill the pointer is on."""
        return self.passthrough if self.passthrough is not None else self.plan[self.skill_index]

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """The skill the environment's tracker considers current (the instruction when it sends none)."""
        if "oracle_prompt" not in call.control_keys and not self._warned:
            log.warning("oracle writer without memory/oracle_prompt: forwarding the instruction")
            self._warned = True
        reported = text_of(call.control_keys.get("oracle_prompt", call.instruction))
        if reported in self.plan:
            self.skill_index = self.plan.index(reported)
            self.passthrough = None
        else:
            self.passthrough = reported
        return {}


class OffWriter(Writer):
    """No writer: an empty plan that never advances; the method forwards the instruction."""

    name = "off"
    drives_prompt = False
    monitor = None

    def __init__(self, task):
        self.task = task
        self.plan: list[str] = []
        self.skill_index = 0
        self.done_all = False

    @property
    def current_skill(self) -> None:
        """No skill: the policy works on the instruction."""
        return None

    def completed(self) -> list[str]:
        """Nothing is ever completed."""
        return []
