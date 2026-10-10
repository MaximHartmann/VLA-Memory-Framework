"""The rules writer: a monitor without a model. It decides from the robot's own gripper and hand signals when the current
skill is done, by rules declared in the task file, and advances a pointer through the plan exactly as the planner does.

Signals, one row per control step:  finger width (m, both fingers together), squeeze (finger force in N in the simulator,
or 1/0 from a gripper that reports a grasped flag), fingertip position x, y, z (m, forward kinematics of the measured
joints). All of them exist on a real robot; nothing here looks at the objects. A task may declare further signals of the
robot (`signals` in the task file, e.g. the contact force at the hand), sent as memory/signal_steps.

Two kinds of triggers; the task file maps each verb to one of them (`history.rules`), and a trigger completes the current
skill only if its verb matches, so a second grasp of a block that slipped out does not write "picked up" twice.
1. The grasp cycle, for pick and place (triggers lift and set_down): grasp_cycle.py.
2. Signal thresholds, for any other skill (trigger signals: phases of conditions on the signals): signal_sequence.py.
The task file's rules section is parsed by rule_spec.py, the environment's signal arrays are read by signals.py, and the
writer itself, RuleMonitor, lives in rule_monitor.py. This module imports all of them, so that
`from vla_memory.history_appended.rules import ...` works for every name as before.
"""
from __future__ import annotations

from .grasp_cycle import GraspCycle
from .rule_monitor import RuleMonitor
from .rule_spec import ALL_TRIGGERS, SIGNALS, TRIGGERS, _OPTIONAL, _PARAMS, parse_rules
from .signal_sequence import ABSOLUTE_OPS, RELATIVE_OPS, SignalSequence, parse_signal_phases, rule_signals
from .signals import signal_columns, signal_rows

__all__ = ["TRIGGERS", "SIGNALS", "ALL_TRIGGERS", "ABSOLUTE_OPS", "RELATIVE_OPS", "parse_rules", "RuleMonitor", "GraspCycle",
           "SignalSequence", "parse_signal_phases", "rule_signals", "signal_rows", "signal_columns"]
