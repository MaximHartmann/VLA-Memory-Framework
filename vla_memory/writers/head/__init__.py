"""The event head as a writer: the history policy writes its own history. A logistic regression on the policy's own
representation of each call (the residual stream after one Gemma block, averaged over the prompt's text tokens) says
whether the current skill is complete; the policy server returns the verdict with the actions, and after `consecutive`
verdicts in a row the writer completes the skill. See docs/writers/head/README.md.

  writer.HeadWriter    the writer: --writer head [--consecutive N]; it decides after the policy call, from the reply
  file                 the head's files (head.npz, head.json, head.sha256): EventHead, load_head, save_head
  rows                 the training rows of a recorded episode: probe_rows
  fit                  the fit: balanced logistic regression, L2 by cross-validation, threshold (fit_head)
  features             the policy's representation of a call, the head's input (openpi environment)
  server               the policy server with the head's verdict in every reply (openpi environment;
                       python -m vla_memory.writers.head.server)
"""
from .file import EventHead, load_head, save_head
from .writer import HeadWriter

__all__ = ["HeadWriter", "EventHead", "load_head", "save_head"]
