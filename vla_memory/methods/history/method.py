"""The history-appended prompt as a method: the policy gets the full instruction with a line per completed skill.

    "Pick up the red block, place it back, and then grasp the blue block. History: picked up the red block."

The lines are the completed skills of the writer, whichever it is. An oracle prompt that is not one of the task's skills is
forwarded as it is. Two options serve timing experiments:

    --max-lines N      show at most N lines (default: the task's history.max_lines, the longest history in training)
    --delay-calls N    show the history of N policy calls ago
"""
from __future__ import annotations

from collections import deque

from ..base import Method
from .prompt import HistoryPrompt


class HistoryMethod(Method):
    """The instruction with the writer's completed skills appended (prompt.HistoryPrompt): at most max_lines of them, and
    with delay_calls the history of that many calls ago."""

    name = "history"

    @classmethod
    def add_arguments(cls, parser):
        """--max-lines and --delay-calls."""
        group = parser.add_argument_group("the history-appended prompt (--method history)")
        group.add_argument("--max-lines", type=int, default=None,
                           help="show at most N history lines (default: history.max_lines of the task file)")
        group.add_argument("--delay-calls", type=int, default=0,
                           help="show the history N policy calls late (timing-tolerance tests)")

    @classmethod
    def build(cls, args, task):
        """The method with the two options."""
        return cls(task, max_lines=args.max_lines, delay_calls=args.delay_calls)

    def __init__(self, task, max_lines: int | None = None, delay_calls: int = 0):
        super().__init__(task)
        self.history = HistoryPrompt(task)
        self.max_lines = max_lines if max_lines is not None else (task.history or {}).get("max_lines")
        self.delay_calls = max(0, int(delay_calls))
        self._shown: deque = deque()       # the histories of the last calls, for --delay-calls

    def start_episode(self, episode: str, control_keys: dict) -> None:
        """Forget the histories shown in the last episode."""
        self._shown = deque()

    def prompt(self, instruction: str, writer) -> str:
        """The instruction with the writer's completed skills; an environment prompt that is not a skill is forwarded."""
        if writer.passthrough is not None:
            return writer.passthrough      # the environment already sent a full prompt
        return self.render(instruction, writer.completed())

    def render(self, instruction: str, completed_skills) -> str:
        """Instruction plus history, at most max_lines lines (the longest history the policy was trained on)."""
        completed = list(completed_skills)
        if self.delay_calls:
            completed = self._delayed(completed)
        if self.max_lines is not None:
            completed = completed[:self.max_lines]
        return self.history.render(instruction, completed)

    def _delayed(self, completed: list) -> list:
        """The history of delay_calls calls ago (render is called once per policy call)."""
        self._shown.append(completed)
        while len(self._shown) > self.delay_calls + 1:
            self._shown.popleft()
        return self._shown[0]
