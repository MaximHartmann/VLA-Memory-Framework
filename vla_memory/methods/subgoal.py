"""The sub-goal prompt: the policy gets the writer's current skill ("pick up the red block") as its prompt.

The policy was trained on short skill prompts and knows nothing of the task as a whole; the writer sequences the skills.
With the VLM planner as the writer this is the planner loop (docs/methods/subgoal/README.md); with the rules writer the
skills are sequenced by the robot's own signals; with the simulator's ground truth it is the reference.
"""
from __future__ import annotations

from .base import Method


class SubgoalMethod(Method):
    name = "subgoal"

    def prompt(self, instruction: str, writer) -> str:
        return writer.current_skill
