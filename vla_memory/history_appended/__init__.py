"""The history-appended prompt (in development): memory inside the policy's own input.

The instruction stays the full sentence and a text is appended that lists the completed skills ("History: picked up
the red block, placed the red block."). A policy fine-tuned on such prompts can tell apart phases that look alike from
the cameras. The events come from the planner loop's monitor, so this method reuses the planner and the exemplar
memory and changes only the prompt handed to the policy. See docs/methods/history_appended/README.md.

  prompt.HistoryPrompt       renders instruction + history from the task file's `history` section; also produces the
                             per-frame prompts of the training data
  proxy.HistoryProxy         the planner loop's relay with the history prompt (import it from
                             vla_memory.history_appended.proxy; run: python -m vla_memory.history_appended.proxy)
"""
from .prompt import HistoryPrompt

__all__ = ["HistoryPrompt"]
