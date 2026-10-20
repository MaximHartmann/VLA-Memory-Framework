"""The rules writer as a writer: rules on the robot's own gripper and hand signals, declared in the task file, move the
pointer before the policy call (rule_monitor.RuleMonitor), so the prompt of the same call shows it. No model, real time.

    --writer rules [--task task.yaml]

The environment sends memory/proprio_steps (finger width, squeeze, fingertip x, y, z of every control step since the last
call), and memory/signal_steps for a task whose `trigger: signals` rules use the task's declared signals. With
history.object_from: colour, the wrist image (memory/wrist_raw) names the object at each pick (identify.py). For a task whose
episodes hold only some of its objects the environment sends memory/objects, and the plan covers those objects. The proxy's
log record of a call: advanced, skill_index, completed (the skills completed at this call), writer_state, identified.
"""
from __future__ import annotations

import logging
from typing import Sequence

from ..base import Call, Writer
from .identify import make_identifier
from .rule_monitor import RuleMonitor
from .signals import signal_rows

log = logging.getLogger("vla_memory.writers")


class RulesWriter(RuleMonitor, Writer):
    """The RuleMonitor behind the writer interface: before_call feeds it the control steps of the call."""

    name = "rules"

    @classmethod
    def build(cls, args, task, extensions: Sequence = ()):
        """The writer with the identifier the task file asks for (history.object_from)."""
        return cls(task, identifier=make_identifier(task))

    def __init__(self, task, plan=None, identifier=None, **gripper_overrides):
        super().__init__(task, plan=plan, identifier=identifier, **gripper_overrides)
        self._warned = False

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """The control steps since the last call (and the wrist image at a pick)."""
        rows = signal_rows(call.control_keys)
        record = self.observe_rows(rows, call.step, image=call.wrist(), **self._declared_signals(call.control_keys))
        return {"advanced": record["advanced"], "skill_index": self.skill_index, "completed": record["completed"] or None,
                "writer_state": record["writer_state"], "identified": record["identified"]}

    def _declared_signals(self, control_keys: dict) -> dict:
        """The task's declared signals (memory/signal_steps) for rules that use them."""
        if not self.signal_names:
            return {}
        steps = control_keys.get("signal_steps")
        if steps is None and not self._warned:
            log.warning("rules writer without memory/signal_steps: the rules on %s never fire", ", ".join(self.signal_names))
            self._warned = True
        return {"signals": steps}
