"""VLA-Memory-Framework: knowledge bases and memory for a robotic agent built on a vision-language-action (VLA) policy.

The top level holds what every memory method shares; each method is a subpackage of its own.

  task.TaskSpec              the task specification, loaded from YAML: objects, skills, completion rules, prompt
                             texts and answer schemas
  vlm.VLMBackend             a language or vision-language model behind an OpenAI-compatible chat API, or your own
  transport.PolicyTransport  the connection to the VLA policy server (openpi websocket protocol, or your own)
  memory.MemoryStore         the long-term memory interface: what from the past resembles the current situation

Methods:
  planner_loop               a VLM planner decomposes the instruction into skills, monitors progress at every policy
                             call with the help of an exemplar store, and hands the policy one skill at a time
"""
from .task import TaskSpec
from .vlm import VLMBackend, OpenAICompatibleVLM, encode_png
from .transport import PolicyTransport, OpenPIWebsocketTransport
from .memory import MemoryStore

__version__ = "0.1.0"
__all__ = ["TaskSpec", "VLMBackend", "OpenAICompatibleVLM", "encode_png", "PolicyTransport", "OpenPIWebsocketTransport",
           "MemoryStore"]
