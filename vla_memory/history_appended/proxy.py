"""The history-appended relay: the planner loop's proxy with one change in what the policy receives.

The policy gets the full instruction plus the history text rendered from the completed skills, instead of the current
skill. Who decides that a skill is complete is the writer, chosen with --mode:

    rules    the rules writer (history_appended.rules.RuleMonitor): the robot's own gripper and hand signals, rules from
             the task file's history.rules section; no model, real time. The environment sends memory/proprio_steps
             (finger width, squeeze, fingertip x, y, z of every control step since the last call; see rules.signal_rows),
             and memory/signal_steps for a task whose `trigger: signals` rules use the task's declared signals.
             With history.object_from: colour, the wrist image (memory/wrist_raw) names the object at each pick
    vlm      the planner loop's VLM monitor: the planner writes the plan, monitors every policy call (with the exemplar
             memory, if given) and advances its pointer
    oracle   the environment's ground-truth tracker: it sends the skill it considers current as memory/oracle_prompt,
             and the history lists the skills before it (reference runs in simulation only)
    off      passthrough

Same command line as the planner loop's proxy:
    python -m vla_memory.history_appended.proxy --mode rules --listen-port ... --upstream-port ... [--task task.yaml]
Evaluation switches for a writer's timing tolerance: --delay-calls N shows every history N policy calls late; --max-lines N
caps the lines (1 = every event after the first is missed).
"""
from __future__ import annotations

import logging
from collections import deque

from ..planner_loop.proxy import PlannerProxy, main as _planner_main
from .prompt import HistoryPrompt
from .identify import make_identifier
from .rules import RuleMonitor, signal_rows

log = logging.getLogger("vla_memory.history_appended.proxy")


class HistoryProxy(PlannerProxy):
    MODES = ("rules", "vlm", "oracle", "off")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = HistoryPrompt(self.task) if self.task is not None else None
        self.max_lines = (self.task.history or {}).get("max_lines") if self.task is not None else None
        self.delay_calls = 0; self._shown: deque = deque(); self._shown_episode = None

    @classmethod
    def add_arguments(cls, ap):
        ap.add_argument("--max-lines", type=int, default=None,
                        help="show at most N history lines (default: history.max_lines of the task file)")
        ap.add_argument("--delay-calls", type=int, default=0, help="show the history N policy calls late (timing-tolerance tests)")

    def configure(self, args):
        if args.max_lines is not None:
            self.max_lines = args.max_lines
        self.delay_calls = max(0, int(args.delay_calls))

    @classmethod
    def build_monitor(cls, args, task):
        if args.mode == "rules":
            return RuleMonitor(task, identifier=make_identifier(task))
        return super().build_monitor(args, task)

    def render(self, instruction: str, done_skills) -> str:
        """Instruction plus history, at most max_lines lines (the longest history the policy was trained on)."""
        done = list(done_skills)
        if self.delay_calls:   # the history of N calls ago (render is called once per policy call)
            if self._shown_episode != self.n:   # the proxy's episode counter: a repeated episode id still starts afresh
                self._shown, self._shown_episode = deque(), self.n
            self._shown.append(done)
            while len(self._shown) > self.delay_calls + 1:
                self._shown.popleft()
            done = self._shown[0]
        return self.history.render(instruction, done[:self.max_lines] if self.max_lines is not None else done)

    def prompt_for(self, instruction: str, rec: dict) -> str:
        p = self.planner
        return self.render(instruction, list(p.plan) if p.done_all else list(p.plan[:p.idx]))

    def decide(self, instruction: str, ctrl: dict, obs: dict) -> tuple[str, dict]:
        if self.mode == "rules":
            wrist = ctrl.get("wrist_raw", obs.get(self.image_keys[1]) if len(self.image_keys) > 1 else None)
            names = getattr(self.planner, "signal_names", ())   # the task's declared signals its rules use
            extra = {"signals": ctrl.get("signal_steps")} if names else {}
            if names and extra["signals"] is None and not getattr(self, "_warned_signals", False):
                log.warning("rules mode without memory/signal_steps: the rules on %s never fire", ", ".join(names))
                self._warned_signals = True
            rec = self.planner.observe_rows(signal_rows(ctrl), int(ctrl.get("step", -1)), image=wrist, **extra)
            return self.prompt_for(instruction, rec), {"advanced": rec["advanced"], "skill_index": self.planner.idx,
                                                       "completed": rec["completed"] or None, "writer_state": rec["writer_state"],
                                                       "identified": rec["identified"]}
        return super().decide(instruction, ctrl, obs)

    def oracle_prompt_for(self, instruction: str, oracle_prompt: str) -> str:
        skills = list(self.task.skills)
        if oracle_prompt in skills:
            return self.render(instruction, skills[:skills.index(oracle_prompt)])
        return oracle_prompt   # the environment already sent a full prompt


def main(argv=None):
    return _planner_main(argv, proxy_cls=HistoryProxy)


if __name__ == "__main__":
    main()
