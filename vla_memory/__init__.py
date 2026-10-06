"""VLA-Memory-Framework: knowledge bases and memory for a robotic agent built on a vision-language-action (VLA) policy.

The top level holds what every memory method shares; each method is a subpackage of its own.

  task.TaskSpec              the task specification, loaded from YAML: objects, skills, completion rules, prompt
                             texts and answer schemas
  vlm.VLMBackend             a language or vision-language model behind an OpenAI-compatible chat API, or your own
  transport.PolicyTransport  the connection to the VLA policy server (openpi websocket protocol, or your own)
  memory.MemoryStore         the long-term memory interface: what from the past resembles the current situation
  env_keys.ControlKeys       the environment's side: the memory/ control keys of every policy call, built from a simulator's
                             or robot's raw signals (docs/ENVIRONMENT_CONTRACT.md)
  vlm.make_vlm, transport.make_transport   another VLM backend or VLA transport by name: a registered name, an installed
                             entry point or module:Class (the proxies' --vlm-backend and --transport; plugins.py)

Methods (one subpackage each; docs/methods/<method>/README.md):
  planner_loop               a VLM planner decomposes the instruction into skills, monitors progress at every policy
                             call with the help of an exemplar store, and hands the policy one skill at a time
  history_appended           the policy keeps the full instruction and receives a text summary of the completed
                             skills; an exchangeable writer supplies the events: rules on the robot's own signals from
                             the task file, the planner loop's VLM monitor, or a simulator's ground truth
  (RECAP-lite, the parametric method, is a training recipe: docs/methods/recap/README.md)
"""
from .task import TaskSpec
from .vlm import VLMBackend, OpenAICompatibleVLM, encode_png, make_vlm, register_vlm_backend
from .transport import PolicyTransport, OpenPIWebsocketTransport, make_transport, register_transport
from .memory import MemoryStore
from .env_keys import ControlKeys

__version__ = "0.1.0"
__all__ = ["TaskSpec", "VLMBackend", "OpenAICompatibleVLM", "encode_png", "make_vlm", "register_vlm_backend", "PolicyTransport",
           "OpenPIWebsocketTransport", "make_transport", "register_transport", "MemoryStore", "ControlKeys"]
