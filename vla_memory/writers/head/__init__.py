"""The event head as a writer: the policy itself says when a skill is done. A logistic regression on a summary of the
policy's own state at each call (for openpi's pi0.5 the residual stream after one Gemma block, averaged over the prompt's
text tokens) says whether the current skill is complete; the policy server hands out the state with the actions
(reply["policy_state"], vla_memory/taps/reply.py), the head runs in the proxy, and after `consecutive` verdicts in a row the
writer completes the skill. See docs/writers/head/README.md.

  writer.HeadWriter    the writer: --writer head --head-dir DIR [--consecutive N]; it decides after the policy call
  record.StateRecorder the proxy's recording for a head (--record-head DIR): the policy's state at every call, and after a
                       completed skill its state for the old input
  file                 the head's files (head.npz, head.json, head.sha256): EventHead, load_head, save_head
  rows                 the training rows of a recorded episode's pictures: probe_rows
  fit                  the fit: balanced logistic regression, L2 by cross-validation, threshold (fit_head)
  crop                 CenterCrop, the centre crop of the exterior picture the openpi server applies at inference
The servers that hand out the state: integrations/openpi/openpi_server.py (pi0.5, JAX or PyTorch) and
integrations/lerobot/lerobot_server.py (any LeRobot policy).
"""
from .file import EventHead, load_head, save_head
from .writer import HeadWriter

__all__ = ["HeadWriter", "EventHead", "load_head", "save_head"]
