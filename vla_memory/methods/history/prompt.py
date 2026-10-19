"""Rendering of the history-appended prompt: the unchanged instruction plus a text that lists the completed skills.

The text is built from the `history` section of the task file: a prefix (" History: "), the phrase for an empty
history ("none yet"), one past-tense phrase per verb with an {object} placeholder ("picked up the {object} block"),
a joiner (", ") and a suffix ("."). The same renderer produces the per-frame prompts of the training data, so the
strings the policy sees at run time are exactly the strings it was trained on.
"""
from __future__ import annotations

from typing import Sequence


class HistoryPrompt:
    """The renderer of the task file's `history` section: prefix, the phrase for an empty history, one phrase per verb,
    joiner and suffix."""

    def __init__(self, task):
        history = dict(task.history or {})
        self.task = task
        self.prefix = history.get("prefix", " History: ")
        self.none = history.get("none", "none yet")
        self.events = dict(history.get("events", {}))
        self.joiner = history.get("joiner", ", ")
        self.suffix = history.get("suffix", ".")

    def event(self, skill: str) -> str:
        """The past-tense phrase of one completed skill; a verb without a phrase falls back to the skill string."""
        verb, object_name = self.task.parse_skill(skill)
        return self.events.get(verb, skill).format(object=object_name)

    def history_text(self, completed_skills: Sequence[str]) -> str:
        """The phrases of the completed skills (or the phrase for an empty history), with the suffix."""
        if not completed_skills:
            return self.none + self.suffix
        return self.joiner.join(self.event(skill) for skill in completed_skills) + self.suffix

    def render(self, instruction: str, completed_skills: Sequence[str]) -> str:
        """The prompt handed to the policy: the instruction, the prefix and the history text."""
        return instruction.strip() + self.prefix + self.history_text(completed_skills)

    def frame_prompts(self, instruction: str, active_skill_index: Sequence[int], plan: Sequence[str] | None = None) -> list[str]:
        """Per-frame prompts of a recorded episode (training data) from the index of the skill active at each frame:
        the history of a frame is every skill before the active one, in the order of `plan` (default: the task's
        skill order)."""
        skills = tuple(plan) if plan else self.task.skills
        return [self.render(instruction, skills[:int(count)]) for count in active_skill_index]
