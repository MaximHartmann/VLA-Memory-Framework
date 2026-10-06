"""The history-appended prompt: memory inside the policy's own input.

The instruction stays the full sentence and a text is appended that lists the completed skills ("History: picked up
the red block, placed the red block."). A policy fine-tuned on such prompts can tell apart phases that look alike from
the cameras. A writer decides when a skill is complete; it is exchangeable: rules on the robot's own gripper and hand
signals declared in the task file, the planner loop's VLM monitor, or a simulator's ground truth.
See docs/methods/history_appended/README.md.

  prompt.HistoryPrompt       renders instruction + history from the task file's `history` section; also produces the
                             per-frame prompts of the training data
  rules.RuleMonitor          the rules writer: grasp-cycle triggers from width, squeeze and fingertip position, mapped to
                             the task's verbs by history.rules; also labels recorded episodes (label_rows)
  identify.ColourIdentifier  names the object held at a pick from the wrist image (history.object_from: colour)
  author                     the planning model writes history.rules from the task description (validate before use;
                             python -m vla_memory.history_appended.author)
  validate                   offline check of a rule set against recorded episodes and a reference
                             (python -m vla_memory.history_appended.validate)
  proxy.HistoryProxy         the relay with the history prompt and the writer choice --mode rules | vlm | oracle
                             (import it from vla_memory.history_appended.proxy; run: python -m vla_memory.history_appended.proxy)
"""
from .prompt import HistoryPrompt
from .rules import RuleMonitor

__all__ = ["HistoryPrompt", "RuleMonitor"]
