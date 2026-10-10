"""The history-appended prompt: memory inside the policy's own input.

The instruction stays the full sentence and a text is appended that lists the completed skills ("History: picked up
the red block, placed the red block."). A policy fine-tuned on such prompts can tell apart phases that look alike from
the cameras. A writer decides when a skill is complete; it is exchangeable: rules on the robot's own gripper and hand
signals declared in the task file, the planner loop's VLM monitor, or a simulator's ground truth.
See docs/methods/history_appended/README.md.

  prompt.HistoryPrompt       renders instruction + history from the task file's `history` section; also produces the
                             per-frame prompts of the training data
  rules.RuleMonitor          the rules writer: grasp-cycle triggers from width, squeeze and fingertip position, mapped to
                             the task's verbs by history.rules (parse_rules); also labels recorded episodes (label_rows).
                             Its parts, re-exported by rules:
    rule_spec                    the task file's rules section: parse_rules and the trigger names
    rule_monitor.RuleMonitor     the writer itself: the plan pointer, observe_rows, completed, label_rows
    grasp_cycle.GraspCycle       the state machine of one grasp cycle (triggers lift and set_down)
    signal_sequence              phases of signal thresholds (trigger signals): parse_signal_phases, SignalSequence
    signals                      the environment's per-step signal arrays: signal_rows, signal_columns
  identify.ColourIdentifier  names the object held at a pick from the wrist image (history.object_from: colour)
  author                     the planning model writes history.rules from the task description (validate before use;
                             python -m vla_memory.history_appended.author). Its parts, re-exported by author:
    author_prompt                the chat messages (build_messages) and the answer schema (answer_schema)
    author_demos                 statistics of labelled demonstrations for the variant hard (demo_event_summary)
    author_yaml                  the copy of the task file with the authored rules (write_task_yaml)
    author_rules_spec            the answer's parameters and bounds (PARAMS), the variants, the acceptance check (check_rules)
  validate                   offline check of a rule set against recorded episodes and a reference
                             (python -m vla_memory.history_appended.validate)
  proxy.HistoryProxy         the relay with the history prompt and the writer choice --mode rules | vlm | oracle
                             (import it from vla_memory.history_appended.proxy; run: python -m vla_memory.history_appended.proxy)
"""
from .prompt import HistoryPrompt
from .rules import RuleMonitor

__all__ = ["HistoryPrompt", "RuleMonitor"]
