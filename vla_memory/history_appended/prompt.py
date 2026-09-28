"""Rendering of the history-appended prompt: the unchanged instruction plus a text that lists the completed skills.

The text is built from the `history` section of the task file: a prefix (" History: "), the phrase for an empty
history ("none yet"), one past-tense phrase per verb with an {object} placeholder ("picked up the {object} block"),
a joiner (", ") and a suffix ("."). The same renderer produces the per-frame prompts of the training data, so the
strings the policy sees at run time are exactly the strings it was trained on.
"""
from __future__ import annotations

from typing import Sequence


class HistoryPrompt:
    def __init__(self, task):
        h = dict(task.history or {})
        self.task = task
        self.prefix = h.get("prefix", " History: "); self.none = h.get("none", "none yet")
        self.events = dict(h.get("events", {})); self.joiner = h.get("joiner", ", "); self.suffix = h.get("suffix", ".")

    def event(self, skill: str) -> str:
        """The past-tense phrase of one completed skill; a verb without a phrase falls back to the skill string."""
        verb, obj = self.task.parse_skill(skill)
        return self.events.get(verb, skill).format(object=obj)

    def history_text(self, done_skills: Sequence[str]) -> str:
        body = self.joiner.join(self.event(s) for s in done_skills) if done_skills else self.none
        return body + self.suffix

    def render(self, instruction: str, done_skills: Sequence[str]) -> str:
        """The prompt handed to the policy."""
        return instruction.strip() + self.prefix + self.history_text(done_skills)

    def frame_prompts(self, instruction: str, active_skill_index: Sequence[int], plan: Sequence[str] | None = None) -> list[str]:
        """Per-frame prompts of a recorded episode (training data) from the index of the skill active at each frame:
        the history of a frame is every skill before the active one, in the order of `plan` (default: the task's
        skill order)."""
        skills = tuple(plan) if plan else self.task.skills
        return [self.render(instruction, skills[:int(k)]) for k in active_skill_index]
