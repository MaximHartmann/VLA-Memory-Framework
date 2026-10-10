"""The planner loop: a VLM planner drives a VLA policy that knows only short skill prompts.

The planner decomposes the user's instruction once into the task's skills, monitors progress at every policy call
from the cameras and the proprioception, and advances a forward-only pointer; the proxy relays observations between
the environment and the policy server and rewrites the prompt to the current skill. Memory: the working memory of
the episode (plan, pointer, recent proprioception) inside Planner, and the exemplar store (labelled frames of past
episodes, retrieved by fingerprint similarity) as long-term memory. See docs/methods/planner_loop/README.md.

  planner.Planner            plan once, monitor every call, hysteresis and forward-only pointer
  gate                       the memory-vote gate: the store's own verdict, which may settle a call without the model
  question                   the texts of the plan request and the monitor question
  proprio.ProprioSummary     the last few gripper / hand-height readings as one sentence
  exemplar.ExemplarStore     the long-term memory: build, save, load, retrieve, few-shot content
  exemplar_dir               the store's directory format, for a store that grows between episodes
  fingerprint.Fingerprint    the retrieval key of an observation (colour-blob key, plain thumbnail, or your own)
  labels.PhaseLabeller       per-frame labels of recorded demonstrations for building a store (parse_episode_ids: "0-19,25")
  proxy.PlannerProxy         the relay (python -m vla_memory.planner_loop.proxy); server.ProxyServer its websocket frontend
  cli                        the command line, the builders and main()
  extensions.Extension       how an optional memory plugs into the proxy and the planner (--with NAME)
"""
from .planner import Planner
from .proprio import ProprioSummary
from .exemplar import ExemplarStore
from .extensions import Extension, register_extension
from .fingerprint import Fingerprint, ColourBlobFingerprint, ThumbnailFingerprint, fingerprint_from_task
from .labels import PhaseLabeller, load_npz_episodes, parse_episode_ids, parse_ranges

__all__ = ["Planner", "ProprioSummary", "ExemplarStore", "Extension", "register_extension", "Fingerprint",
           "ColourBlobFingerprint", "ThumbnailFingerprint", "fingerprint_from_task", "PhaseLabeller", "load_npz_episodes",
           "parse_episode_ids", "parse_ranges"]
