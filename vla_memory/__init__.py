"""VLA-Memory-Framework: knowledge bases and memory for a robotic agent built on a vision-language-action (VLA) policy.

The top level holds what every memory method shares; each method is a subpackage of its own.

  task.TaskSpec              the task specification, loaded from YAML: objects, skills, completion rules, prompt
                             texts and answer schemas
  vlm.VLMBackend             a language or vision-language model behind an OpenAI-compatible chat API, or your own
  transport.PolicyTransport  the connection to the VLA policy server (openpi websocket protocol, or your own)
  memory.MemoryStore         the long-term memory interface: what from the past resembles the current situation
  taps                       state taps: one layer's state of a policy, read during its own forward pass (PyTorch hooks,
                             JAX hand-over; docs/taps/README.md)
  env_keys.ControlKeys       the environment's side: the memory/ control keys of every policy call, built from a simulator's
                             or robot's raw signals
  vlm.make_vlm, transport.make_transport   another VLM backend or VLA transport by name: a registered name, an installed
                             entry point or module:Class (the proxies' --vlm-backend and --transport; plugins.py)

Two layers, chosen with --writer NAME and --method NAME (docs/writers/<name>/README.md, docs/methods/<name>/README.md):
  writers                    signal that a step of the task is done:
                               oracle     the simulator's ground truth
                               rules      rules on the robot's own signals
                               vlm        a VLM planner with an exemplar store
                               head       the policy's own event head
  methods                    turn that signal into what the policy receives:
                               subgoal    the current skill
                               history    the instruction with the completed skills
                               keyframe   the instruction and the picture of the last key moment
                             other memories plug in as extensions (--with NAME)
  proxy                      the relay between the environment and the policy server that runs a writer and a method:
                             python -m vla_memory.proxy
  (RECAP-lite, the parametric method, is a training recipe: docs/methods/recap/README.md; the built-in memory lives in the
  policy, a reference implementation for pi0.5 on openpi in integrations/openpi: docs/methods/builtin_memory/README.md)
"""
from .task import TaskSpec
from .vlm import VLMBackend, OpenAICompatibleVLM, encode_png, make_vlm, register_vlm_backend
from .transport import PolicyTransport, OpenPIWebsocketTransport, make_transport, register_transport
from .memory import MemoryStore
from .env_keys import ControlKeys

__version__ = "0.1.0"
__all__ = ["TaskSpec", "VLMBackend", "OpenAICompatibleVLM", "encode_png", "make_vlm", "register_vlm_backend", "PolicyTransport",
           "OpenPIWebsocketTransport", "make_transport", "register_transport", "MemoryStore", "ControlKeys"]
