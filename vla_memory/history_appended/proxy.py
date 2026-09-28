"""The history-appended relay: the planner loop's proxy with one change in what the policy receives.

The planner still writes the plan, monitors every policy call (with the exemplar memory, if given) and advances its
forward-only pointer; it is the event detector of this method. Instead of the current skill, the policy gets the full
instruction plus the history text rendered from the skills before the pointer. Same command line as the planner
loop's proxy:  python -m vla_memory.history_appended.proxy --listen-port ... --upstream-port ... [--memory store.npz]
"""
from __future__ import annotations

from ..planner_loop.proxy import PlannerProxy, main as _planner_main
from .prompt import HistoryPrompt


class HistoryProxy(PlannerProxy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = HistoryPrompt(self.planner.task) if self.planner is not None else None

    def prompt_for(self, instruction: str, rec: dict) -> str:
        p = self.planner
        done = list(p.plan) if p.done_all else list(p.plan[:p.idx])
        return self.history.render(instruction, done)


def main(argv=None):
    return _planner_main(argv, proxy_cls=HistoryProxy)


if __name__ == "__main__":
    main()
