"""The history-appended relay: the planner loop's proxy with one change in what the policy receives.

The policy gets the full instruction plus the history text rendered from the completed skills, instead of the current
skill. Who decides that a skill is complete is the writer, chosen with --mode (the modes of the planner loop's proxy):

    rules    the rules writer (history_appended.rules.RuleMonitor): the robot's own gripper and hand signals, rules from
             the task file's history.rules section; no model, real time. The environment sends memory/proprio_steps
             (finger width, squeeze, fingertip x, y, z of every control step since the last call), and memory/signal_steps
             for a task whose `trigger: signals` rules use the task's declared signals. With history.object_from: colour,
             the wrist image (memory/wrist_raw) names the object at each pick
    vlm      the planner loop's VLM monitor: the planner writes the plan, monitors every policy call (with the exemplar
             memory, if given) and advances its pointer
    oracle   the environment's ground-truth tracker: it sends the skill it considers current as memory/oracle_prompt,
             and the history lists the skills before it (reference runs in simulation only)
    off      passthrough

Same command line as the planner loop's proxy:
    python -m vla_memory.history_appended.proxy --mode rules --listen-port ... --upstream-port ... [--task task.yaml]
Two switches serve the timing experiments: --delay-calls N shows every history N policy calls late; --max-lines N caps the
lines (1 = every event after the first is missed).
"""
from __future__ import annotations

from collections import deque

from ..planner_loop.cli import main as planner_main
from ..planner_loop.proxy import PlannerProxy
from .prompt import HistoryPrompt


class HistoryProxy(PlannerProxy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = HistoryPrompt(self.task) if self.task is not None else None
        self.max_lines = (self.task.history or {}).get("max_lines") if self.task is not None else None
        self.delay_calls = 0
        self._shown: deque = deque()       # the histories of the last calls, for --delay-calls
        self._shown_episode = None

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--max-lines", type=int, default=None,
                            help="show at most N history lines (default: history.max_lines of the task file)")
        parser.add_argument("--delay-calls", type=int, default=0,
                            help="show the history N policy calls late (timing-tolerance tests)")

    def configure(self, args):
        if args.max_lines is not None:
            self.max_lines = args.max_lines
        self.delay_calls = max(0, int(args.delay_calls))

    def render(self, instruction: str, done_skills) -> str:
        """Instruction plus history, at most max_lines lines (the longest history the policy was trained on)."""
        done = list(done_skills)
        if self.delay_calls:
            done = self._delayed(done)
        if self.max_lines is not None:
            done = done[:self.max_lines]
        return self.history.render(instruction, done)

    def _delayed(self, done: list) -> list:
        """The history of delay_calls calls ago (render is called once per policy call; an episode starts afresh)."""
        if self._shown_episode != self.episode_count:
            self._shown = deque()
            self._shown_episode = self.episode_count
        self._shown.append(done)
        while len(self._shown) > self.delay_calls + 1:
            self._shown.popleft()
        return self._shown[0]

    def prompt_for(self, instruction: str, record: dict) -> str:
        planner = self.planner
        done = list(planner.plan) if planner.done_all else list(planner.plan[:planner.idx])
        return self.render(instruction, done)

    def oracle_prompt_for(self, instruction: str, oracle_prompt: str) -> str:
        skills = list(self.task.skills)
        if oracle_prompt in skills:
            return self.render(instruction, skills[:skills.index(oracle_prompt)])
        return oracle_prompt   # the environment already sent a full prompt


def main(argv=None):
    return planner_main(argv, proxy_cls=HistoryProxy)


if __name__ == "__main__":
    main()
