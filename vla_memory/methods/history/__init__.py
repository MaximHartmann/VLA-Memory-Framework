"""The history-appended prompt: memory inside the policy's own input.

The instruction stays the full sentence and a text is appended that lists the completed skills ("History: picked up the
red block, placed the red block."). A policy fine-tuned on such prompts can tell apart phases that look alike from the
cameras. Who decides that a skill is complete is the writer (writers/), chosen on the command line; the policy only sees
the text. See docs/methods/history/README.md.

  prompt.HistoryPrompt     renders instruction + history from the task file's `history` section; also produces the
                           per-frame prompts of the training data
  method.HistoryMethod     the method: --method history [--max-lines N] [--delay-calls N]
"""
from .prompt import HistoryPrompt

__all__ = ["HistoryPrompt"]
