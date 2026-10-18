"""The rules writer: a writer without a model. It decides from the robot's own gripper and hand signals when the current
skill is done, by rules declared in the task file, and advances a pointer through the plan. See docs/writers/rules/README.md.

  writer.RulesWriter         the writer: --writer rules
  rules                      the rule set's parts, re-exported for convenience:
    rule_spec                    the task file's rules section: parse_rules and the trigger names
    rule_monitor.RuleMonitor     the writer's logic: the plan pointer, observe_rows, completed, label_rows (training labels)
    grasp_cycle.GraspCycle       the state machine of one grasp cycle (triggers lift and set_down)
    signal_sequence              phases of signal thresholds (trigger signals): parse_signal_phases, SignalSequence
    signals                      the environment's per-step signal arrays: signal_rows, signal_columns
  identify.ColourIdentifier  names the object held at a pick from the wrist image (history.object_from: colour)
  validate                   offline check of a rule set against recorded episodes and a reference
                             (python -m vla_memory.writers.rules.validate)
  author                     the planning model writes history.rules from the task description (validate before use;
                             python -m vla_memory.writers.rules.author). Its parts: author_prompt (the chat messages and the
                             answer schema), author_demos (statistics of labelled demonstrations), author_yaml (the copy
                             of the task file with the authored rules), author_rules_spec (the parameters and the check)
"""
from .rules import RuleMonitor

__all__ = ["RuleMonitor"]
